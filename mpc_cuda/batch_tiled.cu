#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include "pir_config.h"
#include "fss_cuda_launch.h"
#include "../mpc_keys/uint128_type.h"

// One-pass batch-tiled inner product with parameterized:
//   - EntrySize   (E = D / 16): supported {2, 4, 8, 16, 32, 64} (D = 32..1024 B)
//   - TileEntries (T):          supported {16, 32, 64, 128, 256}
//   - num_partitions (P):       gridDim.x
//   - group_size (B_g):         gridDim.y = ceil(batch_size / B_g), 1 <= B_g <= 512
//
// Each block owns a contiguous DB partition (blockIdx.x) and a cooperative query
// group (blockIdx.y). Loader threads (blockDim.x, decoupled from batch_size and
// always a warp-aligned count >= k_min_batch_loader_threads) cooperatively stage
// each tile into shared memory; active query threads consume the tile via
// uniform SMEM broadcast and accumulate into private registers.
//
// Empty partitions (start >= entries) explicitly write zero partials so reused
// partial buffers cannot leak stale data.
//
// partials layout: [partition][query][entry_component]

namespace
{
constexpr size_t k_max_compiled_smem_bytes = 164 * 1024;

__device__ __forceinline__ bool batch_word_bit(const uint128_t &w, int bit)
{
    return (bit >= 64) ? bool((w.get_high() >> (bit - 64)) & 1ULL)
                       : bool((w.get_low() >> bit) & 1ULL);
}

template <bool Async, int EntrySize, int TileEntries>
__device__ __forceinline__ void batch_load_tile(uint128_t *dst, const uint128_t *db,
                                                uint64_t tile, int tid, int nthr)
{
    constexpr int total_words = TileEntries * EntrySize;
    const uint64_t db_base = tile * static_cast<uint64_t>(EntrySize);
    for (int idx = tid; idx < total_words; idx += nthr)
    {
        if constexpr (Async)
            __pipeline_memcpy_async(&dst[idx], &db[db_base + idx], sizeof(uint128_t));
        else
            dst[idx] = db[db_base + idx];
    }
}

template <int EntrySize, int TileEntries>
__device__ __forceinline__ void batch_accumulate_tile(const uint128_t *tile_smem,
                                                      const uint128_t *se,
                                                      uint128_t *acc,
                                                      int query,
                                                      uint64_t words_per_query,
                                                      uint64_t tile)
{
    if constexpr (TileEntries <= 128)
    {
        const uint128_t sw = se[static_cast<uint64_t>(query) * words_per_query + (tile >> 7)];
        const int bit_base = 127 - static_cast<int>(tile & 127ULL);

#pragma unroll 4
        for (int e = 0; e < TileEntries; ++e)
        {
            if (batch_word_bit(sw, bit_base - e))
            {
#pragma unroll
                for (int j = 0; j < EntrySize; ++j)
                    acc[j] ^= tile_smem[e * EntrySize + j];
            }
        }
    }
    else
    {
        constexpr int k_words = TileEntries / 128;
        const uint64_t se_base = static_cast<uint64_t>(query) * words_per_query + (tile >> 7);

        for (int w = 0; w < k_words; ++w)
        {
            const uint128_t sw = se[se_base + w];
#pragma unroll 4
            for (int e = 0; e < 128; ++e)
            {
                if (batch_word_bit(sw, 127 - e))
                {
                    const int row_in_tile = w * 128 + e;
#pragma unroll
                    for (int j = 0; j < EntrySize; ++j)
                        acc[j] ^= tile_smem[row_in_tile * EntrySize + j];
                }
            }
        }
    }
}

template <bool Async, int EntrySize, int TileEntries>
__device__ __forceinline__ void batch_mul_red_body(uint128_t *smem,
                                                   const uint128_t *se,
                                                   const uint128_t *db,
                                                   uint128_t *partials,
                                                   int db_bits,
                                                   uint64_t per_block,
                                                   int batch_size,
                                                   int group_size)
{
    const int tid = static_cast<int>(threadIdx.x);
    const int query = static_cast<int>(blockIdx.y) * group_size + tid;
    const bool active_query = (tid < group_size) && (query < batch_size);

    const uint64_t entries = 1ULL << db_bits;
    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * per_block;
    const uint64_t end = (start + per_block < entries) ? (start + per_block) : entries;
    const uint64_t words_per_query = entries >> 7;

    uint128_t acc[EntrySize];
#pragma unroll
    for (int j = 0; j < EntrySize; ++j)
        acc[j] = uint128_t(0, 0);

    // Block-uniform guard: all threads take the same branch, so barriers remain
    // deadlock-free and empty partitions still write zero partials below.
    if (start < entries)
    {
        if constexpr (Async)
        {
            constexpr size_t tile_words = static_cast<size_t>(TileEntries) * EntrySize;
            batch_load_tile<true, EntrySize, TileEntries>(smem, db, start, tid, blockDim.x);
            __pipeline_commit();

            int buf = 0;
            for (uint64_t tile = start; tile < end; tile += TileEntries)
            {
                const uint64_t next = tile + TileEntries;
                const int next_buf = buf ^ 1;

                if (next < end)
                {
                    batch_load_tile<true, EntrySize, TileEntries>(
                        smem + static_cast<size_t>(next_buf) * tile_words,
                        db, next, tid, blockDim.x);
                    __pipeline_commit();
                    __pipeline_wait_prior(1);
                }
                else
                {
                    __pipeline_wait_prior(0);
                }
                __syncthreads();

                if (active_query)
                {
                    batch_accumulate_tile<EntrySize, TileEntries>(
                        smem + static_cast<size_t>(buf) * tile_words,
                        se, acc, query, words_per_query, tile);
                }
                __syncthreads();
                buf = next_buf;
            }
        }
        else
        {
            for (uint64_t tile = start; tile < end; tile += TileEntries)
            {
                batch_load_tile<false, EntrySize, TileEntries>(smem, db, tile, tid, blockDim.x);
                __syncthreads();
                if (active_query)
                {
                    batch_accumulate_tile<EntrySize, TileEntries>(
                        smem, se, acc, query, words_per_query, tile);
                }
                __syncthreads();
            }
        }
    }

    if (active_query)
    {
        const uint64_t base = (static_cast<uint64_t>(blockIdx.x) * batch_size + query) * EntrySize;
#pragma unroll
        for (int j = 0; j < EntrySize; ++j)
            partials[base + j] = acc[j];
    }
}

template <bool Async, int EntrySize, int TileEntries>
__global__ __launch_bounds__(512, 1)
void batch_mul_red_tiled_kernel(const uint128_t *se,
                                const uint128_t *db,
                                uint128_t *partials,
                                int db_bits,
                                uint64_t per_block,
                                int batch_size,
                                int group_size)
{
    extern __shared__ uint128_t smem[];
    batch_mul_red_body<Async, EntrySize, TileEntries>(
        smem, se, db, partials, db_bits, per_block, batch_size, group_size);
}

template <int EntrySize>
__global__ void batch_reduce_partials_tiled_kernel(const uint128_t *partials,
                                                   uint128_t *out,
                                                   int num_partitions,
                                                   int batch_size)
{
    const int query = blockIdx.x;
    const int j = threadIdx.x;
    if (query >= batch_size || j >= EntrySize)
        return;

    uint128_t v(0, 0);
    for (int p = 0; p < num_partitions; ++p)
        v ^= partials[(static_cast<uint64_t>(p) * batch_size + query) * EntrySize + j];
    out[static_cast<uint64_t>(query) * EntrySize + j] = v;
}

int compute_loader_threads(int group_size, int batch_size)
{
    const int active = std::min(group_size, batch_size);
    const int aligned = ((active + 31) / 32) * 32;
    return std::min(k_max_batch_group_size, std::max(k_min_batch_loader_threads, aligned));
}

template <bool Async, int EntrySize, int TileEntries>
cudaError_t configure_smem_for_kernel(size_t max_smem_optin)
{
    const size_t bytes = batch_tiled_smem_bytes(Async, EntrySize, TileEntries);
    if constexpr ((Async ? 2ULL : 1ULL) * static_cast<size_t>(TileEntries) * EntrySize * sizeof(uint128_t) > k_max_compiled_smem_bytes)
    {
        return cudaErrorInvalidValue;
    }
    else
    {
        if (bytes > max_smem_optin)
            return cudaErrorInvalidValue;
        if (bytes > 48 * 1024)
        {
            return cudaFuncSetAttribute(batch_mul_red_tiled_kernel<Async, EntrySize, TileEntries>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(bytes));
        }
        return cudaSuccess;
    }
}

template <bool Async, int EntrySize, int TileEntries>
cudaError_t launch_typed_batch(int num_partitions,
                               int group_size,
                               const uint128_t *se,
                               const uint128_t *db,
                               uint128_t *partials,
                               int db_bits,
                               uint64_t per_block,
                               int batch_size,
                               cudaStream_t stream)
{
    if constexpr ((Async ? 2ULL : 1ULL) * static_cast<size_t>(TileEntries) * EntrySize * sizeof(uint128_t) > k_max_compiled_smem_bytes)
    {
        return cudaErrorInvalidValue;
    }
    else
    {
        const size_t smem = batch_tiled_smem_bytes(Async, EntrySize, TileEntries);
        const int num_groups = (batch_size + group_size - 1) / group_size;
        const int threads = compute_loader_threads(group_size, batch_size);
        const dim3 grid(static_cast<unsigned int>(num_partitions),
                        static_cast<unsigned int>(num_groups));
        batch_mul_red_tiled_kernel<Async, EntrySize, TileEntries>
            <<<grid, threads, smem, stream>>>(se, db, partials, db_bits, per_block, batch_size, group_size);
        return cudaGetLastError();
    }
}

template <bool Async, int EntrySize>
cudaError_t configure_smem_by_tile(int tile_entries, size_t max_smem_optin)
{
    switch (tile_entries)
    {
    case 16:  return configure_smem_for_kernel<Async, EntrySize, 16>(max_smem_optin);
    case 32:  return configure_smem_for_kernel<Async, EntrySize, 32>(max_smem_optin);
    case 64:  return configure_smem_for_kernel<Async, EntrySize, 64>(max_smem_optin);
    case 128: return configure_smem_for_kernel<Async, EntrySize, 128>(max_smem_optin);
    case 256: return configure_smem_for_kernel<Async, EntrySize, 256>(max_smem_optin);
    default:  return cudaErrorInvalidValue;
    }
}

template <bool Async>
cudaError_t configure_smem_by_entry(int entry_words, int tile_entries, size_t max_smem_optin)
{
    switch (entry_words)
    {
    case 2:  return configure_smem_by_tile<Async, 2>(tile_entries, max_smem_optin);
    case 4:  return configure_smem_by_tile<Async, 4>(tile_entries, max_smem_optin);
    case 8:  return configure_smem_by_tile<Async, 8>(tile_entries, max_smem_optin);
    case 16: return configure_smem_by_tile<Async, 16>(tile_entries, max_smem_optin);
    case 32: return configure_smem_by_tile<Async, 32>(tile_entries, max_smem_optin);
    case 64: return configure_smem_by_tile<Async, 64>(tile_entries, max_smem_optin);
    default: return cudaErrorInvalidValue;
    }
}

template <bool Async, int EntrySize>
cudaError_t launch_by_tile(int tile_entries,
                           int num_partitions,
                           int group_size,
                           const uint128_t *se,
                           const uint128_t *db,
                           uint128_t *partials,
                           int db_bits,
                           uint64_t per_block,
                           int batch_size,
                           cudaStream_t stream)
{
    switch (tile_entries)
    {
    case 16:  return launch_typed_batch<Async, EntrySize, 16>(num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    case 32:  return launch_typed_batch<Async, EntrySize, 32>(num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    case 64:  return launch_typed_batch<Async, EntrySize, 64>(num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    case 128: return launch_typed_batch<Async, EntrySize, 128>(num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    case 256: return launch_typed_batch<Async, EntrySize, 256>(num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    default:  return cudaErrorInvalidValue;
    }
}

template <bool Async>
cudaError_t launch_by_entry(int entry_words,
                            int tile_entries,
                            int num_partitions,
                            int group_size,
                            const uint128_t *se,
                            const uint128_t *db,
                            uint128_t *partials,
                            int db_bits,
                            uint64_t per_block,
                            int batch_size,
                            cudaStream_t stream)
{
    switch (entry_words)
    {
    case 2:  return launch_by_tile<Async, 2>(tile_entries, num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    case 4:  return launch_by_tile<Async, 4>(tile_entries, num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    case 8:  return launch_by_tile<Async, 8>(tile_entries, num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    case 16: return launch_by_tile<Async, 16>(tile_entries, num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    case 32: return launch_by_tile<Async, 32>(tile_entries, num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    case 64: return launch_by_tile<Async, 64>(tile_entries, num_partitions, group_size, se, db, partials, db_bits, per_block, batch_size, stream);
    default: return cudaErrorInvalidValue;
    }
}
} // namespace

// Backward-compatible direct kernel entry points (default entry_size & k_batch_tile_entries).
__global__ __launch_bounds__(512, 1)
void batch_mul_red_sync_kernel(const uint128_t *se,
                               const uint128_t *db,
                               uint128_t *partials,
                               int db_bits, uint64_t per_block, int batch_size)
{
    extern __shared__ uint128_t smem[];
    batch_mul_red_body<false, entry_size, k_batch_tile_entries>(
        smem, se, db, partials, db_bits, per_block, batch_size, batch_size);
}

__global__ __launch_bounds__(512, 1)
void batch_mul_red_async_kernel(const uint128_t *se,
                                const uint128_t *db,
                                uint128_t *partials,
                                int db_bits, uint64_t per_block, int batch_size)
{
    extern __shared__ uint128_t smem[];
    batch_mul_red_body<true, entry_size, k_batch_tile_entries>(
        smem, se, db, partials, db_bits, per_block, batch_size, batch_size);
}

__global__ void batch_reduce_partials_kernel(const uint128_t *partials,
                                             uint128_t *out,
                                             int num_partitions, int batch_size)
{
    const int query = blockIdx.x;
    const int j = threadIdx.x;
    if (query >= batch_size || j >= entry_size)
        return;

    uint128_t v(0, 0);
    for (int p = 0; p < num_partitions; ++p)
        v ^= partials[(static_cast<uint64_t>(p) * batch_size + query) * entry_size + j];
    out[static_cast<uint64_t>(query) * entry_size + j] = v;
}

size_t batch_tiled_smem_bytes(bool async_mode, int entry_words, int tile_entries)
{
    return (async_mode ? 2ULL : 1ULL) * static_cast<size_t>(tile_entries) *
           static_cast<size_t>(entry_words) * sizeof(uint128_t);
}

bool batch_tiled_config_supported(bool async_mode, int entry_words, int tile_entries, size_t max_smem_optin)
{
    if (!is_supported_entry_words(entry_words) || !is_supported_tile_entries(tile_entries))
        return false;
    const size_t bytes = batch_tiled_smem_bytes(async_mode, entry_words, tile_entries);
    return bytes <= k_max_compiled_smem_bytes && bytes <= max_smem_optin;
}

cudaError_t configure_batch_tiled_smem(bool async_mode, int entry_words, int tile_entries, size_t max_smem_optin)
{
    if (!batch_tiled_config_supported(async_mode, entry_words, tile_entries, max_smem_optin))
        return cudaErrorInvalidValue;
    return async_mode
               ? configure_smem_by_entry<true>(entry_words, tile_entries, max_smem_optin)
               : configure_smem_by_entry<false>(entry_words, tile_entries, max_smem_optin);
}

cudaError_t launch_batch_mul_red(bool async_mode,
                                 int entry_words,
                                 int tile_entries,
                                 int num_partitions,
                                 int group_size,
                                 const uint128_t *se,
                                 const uint128_t *db,
                                 uint128_t *partials,
                                 int db_bits,
                                 uint64_t per_block,
                                 int batch_size,
                                 cudaStream_t stream)
{
    if (!is_supported_entry_words(entry_words) || !is_supported_tile_entries(tile_entries) ||
        num_partitions <= 0 || batch_size <= 0 ||
        group_size <= 0 || group_size > k_max_batch_group_size ||
        db_bits < 7 || (uint64_t{1} << db_bits) < static_cast<uint64_t>(tile_entries) ||
        (per_block % static_cast<uint64_t>(tile_entries)) != 0)
    {
        return cudaErrorInvalidValue;
    }

    return async_mode
               ? launch_by_entry<true>(entry_words, tile_entries, num_partitions, group_size,
                                       se, db, partials, db_bits, per_block, batch_size, stream)
               : launch_by_entry<false>(entry_words, tile_entries, num_partitions, group_size,
                                        se, db, partials, db_bits, per_block, batch_size, stream);
}

cudaError_t launch_batch_reduce_partials(int entry_words,
                                         const uint128_t *partials,
                                         uint128_t *out,
                                         int num_partitions,
                                         int batch_size,
                                         cudaStream_t stream)
{
    if (num_partitions <= 0 || batch_size <= 0)
        return cudaErrorInvalidValue;

    switch (entry_words)
    {
    case 2:
        batch_reduce_partials_tiled_kernel<2><<<batch_size, 2, 0, stream>>>(partials, out, num_partitions, batch_size);
        return cudaGetLastError();
    case 4:
        batch_reduce_partials_tiled_kernel<4><<<batch_size, 4, 0, stream>>>(partials, out, num_partitions, batch_size);
        return cudaGetLastError();
    case 8:
        batch_reduce_partials_tiled_kernel<8><<<batch_size, 8, 0, stream>>>(partials, out, num_partitions, batch_size);
        return cudaGetLastError();
    case 16:
        batch_reduce_partials_tiled_kernel<16><<<batch_size, 16, 0, stream>>>(partials, out, num_partitions, batch_size);
        return cudaGetLastError();
    case 32:
        batch_reduce_partials_tiled_kernel<32><<<batch_size, 32, 0, stream>>>(partials, out, num_partitions, batch_size);
        return cudaGetLastError();
    case 64:
        batch_reduce_partials_tiled_kernel<64><<<batch_size, 64, 0, stream>>>(partials, out, num_partitions, batch_size);
        return cudaGetLastError();
    default:
        return cudaErrorInvalidValue;
    }
}
