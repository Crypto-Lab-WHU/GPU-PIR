#include <cstdint>
#include <cuda_runtime.h>

#include "pir_config.h"
#include "fss_cuda_launch.h"
#include "../mpc_keys/uint128_type.h"

// Multi-SM conditional inner product over a packed DPF selection vector.
// Parameterized over EntrySize (E = D / 16 in {2, 4, 8, 16, 32, 64}),
// LanesPerRow (L dividing gcd(32, E)), partition count (gridDim.x), and
// small-batch count (gridDim.y).
//
// partials layout: [query][partition][entry_component]
// (For single-query gridDim.y == 1, this reduces to [partition][entry_component].)

namespace
{
__device__ __forceinline__ uint128_t shfl_down_128(uint128_t value, int delta)
{
    return uint128_t(__shfl_down_sync(0xffffffffu, value.get_high(), delta),
                     __shfl_down_sync(0xffffffffu, value.get_low(), delta));
}

template <int EntrySize>
__device__ __forceinline__ void warp_xor_reduce(uint128_t (&acc)[EntrySize])
{
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
    {
#pragma unroll
        for (int j = 0; j < EntrySize; ++j)
            acc[j] ^= shfl_down_128(acc[j], delta);
    }
}

template <int EntrySize>
__device__ __forceinline__ void store_block_partial(uint128_t (&acc)[EntrySize],
                                                    uint128_t *smem,
                                                    uint128_t *partials)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;

    warp_xor_reduce<EntrySize>(acc);

    if (lane == 0)
    {
#pragma unroll
        for (int j = 0; j < EntrySize; ++j)
            smem[EntrySize * warp + j] = acc[j];
    }
    __syncthreads();

    const uint64_t partial_idx = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    for (int idx = tid; idx < EntrySize; idx += blockDim.x)
    {
        uint128_t total(0, 0);
        for (int w = 0; w < warps; ++w)
            total ^= smem[EntrySize * w + idx];
        partials[partial_idx * EntrySize + idx] = total;
    }
}

__device__ __forceinline__ bool selected(const uint128_t *se, uint64_t row)
{
    return se[row >> 7].get_bit(127 - static_cast<int>(row & 127ULL));
}

template <int EntrySize>
__device__ __forceinline__ void sector_mul_red_row_body(const uint128_t *__restrict__ se,
                                                        const uint128_t *__restrict__ db,
                                                        uint128_t *partials,
                                                        uint64_t rows,
                                                        uint64_t per_block)
{
    __shared__ uint128_t smem[8 * EntrySize];

    const int tid = threadIdx.x;
    const uint64_t words_per_query = rows >> 7;
    const uint128_t *se_query = se + static_cast<uint64_t>(blockIdx.y) * words_per_query;

    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * per_block;
    const uint64_t end = (start + per_block < rows) ? start + per_block : rows;

    uint128_t acc[EntrySize];
#pragma unroll
    for (int j = 0; j < EntrySize; ++j)
        acc[j] = uint128_t(0, 0);

    if (start < rows)
    {
        for (uint64_t row = start + tid; row < end; row += blockDim.x)
        {
            if (selected(se_query, row))
            {
#pragma unroll
                for (int j = 0; j < EntrySize; ++j)
                    acc[j] ^= db[row * EntrySize + j];
            }
        }
    }
    store_block_partial<EntrySize>(acc, smem, partials);
}

template <int EntrySize, int LanesPerRow>
__device__ __forceinline__ void sector_mul_red_lanes_body(const uint128_t *__restrict__ se,
                                                          const uint128_t *__restrict__ db,
                                                          uint128_t *partials,
                                                          uint64_t rows,
                                                          uint64_t per_block)
{
    static_assert(LanesPerRow > 0 && LanesPerRow <= 32 && (32 % LanesPerRow == 0),
                  "LanesPerRow must be a power of two in [1, 32]");
    static_assert(EntrySize % LanesPerRow == 0, "EntrySize must be divisible by LanesPerRow");

    constexpr int k_rows_per_warp = 32 / LanesPerRow;
    constexpr int k_comp_per_lane = EntrySize / LanesPerRow;

    __shared__ uint128_t smem[8 * EntrySize];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const int sub = lane & (LanesPerRow - 1);
    const int row_in_group = lane / LanesPerRow;

    const uint64_t words_per_query = rows >> 7;
    const uint128_t *se_query = se + static_cast<uint64_t>(blockIdx.y) * words_per_query;

    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * per_block;
    const uint64_t end = (start + per_block < rows) ? start + per_block : rows;

    uint128_t acc[k_comp_per_lane];
#pragma unroll
    for (int c = 0; c < k_comp_per_lane; ++c)
        acc[c] = uint128_t(0, 0);

    if (start < rows)
    {
        for (uint64_t row = start + static_cast<uint64_t>(k_rows_per_warp) * warp + row_in_group;
             row < end;
             row += static_cast<uint64_t>(k_rows_per_warp) * warps)
        {
            if (selected(se_query, row))
            {
                const uint128_t *src = db + row * EntrySize + sub * k_comp_per_lane;
#pragma unroll
                for (int c = 0; c < k_comp_per_lane; ++c)
                    acc[c] ^= src[c];
            }
        }
    }

#pragma unroll
    for (int delta = 16; delta >= LanesPerRow; delta >>= 1)
    {
#pragma unroll
        for (int c = 0; c < k_comp_per_lane; ++c)
            acc[c] ^= shfl_down_128(acc[c], delta);
    }

    if (lane < LanesPerRow)
    {
#pragma unroll
        for (int c = 0; c < k_comp_per_lane; ++c)
            smem[EntrySize * warp + sub * k_comp_per_lane + c] = acc[c];
    }
    __syncthreads();

    const uint64_t partial_idx = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    for (int idx = tid; idx < EntrySize; idx += blockDim.x)
    {
        uint128_t total(0, 0);
        for (int w = 0; w < warps; ++w)
            total ^= smem[EntrySize * w + idx];
        partials[partial_idx * EntrySize + idx] = total;
    }
}

template <int EntrySize>
__device__ __forceinline__ void sector_col_mul_red_body(const uint128_t *__restrict__ se,
                                                        const uint128_t *__restrict__ db,
                                                        uint128_t *partials,
                                                        uint64_t rows,
                                                        uint64_t per_block)
{
    __shared__ uint128_t smem[8 * EntrySize];

    const int tid = threadIdx.x;
    const uint64_t words_per_query = rows >> 7;
    const uint128_t *se_query = se + static_cast<uint64_t>(blockIdx.y) * words_per_query;

    const uint64_t start = static_cast<uint64_t>(blockIdx.x) * per_block;
    const uint64_t end = (start + per_block < rows) ? start + per_block : rows;

    uint128_t acc[EntrySize];
#pragma unroll
    for (int j = 0; j < EntrySize; ++j)
        acc[j] = uint128_t(0, 0);

    if (start < rows)
    {
        for (uint64_t row = start + tid; row < end; row += blockDim.x)
        {
            if (selected(se_query, row))
            {
#pragma unroll
                for (int j = 0; j < EntrySize; ++j)
                    acc[j] ^= db[static_cast<uint64_t>(j) * rows + row];
            }
        }
    }
    store_block_partial<EntrySize>(acc, smem, partials);
}

template <int EntrySize>
__device__ __forceinline__ void sector_reduce_partials_body(const uint128_t *__restrict__ partials,
                                                            uint128_t *__restrict__ out,
                                                            int partitions)
{
    __shared__ uint128_t smem[256];

    const int tid = threadIdx.x;
    const int component = tid % EntrySize;
    const int group = tid / EntrySize;
    const int groups = blockDim.x / EntrySize;

    const uint128_t *q_partials = partials + static_cast<uint64_t>(blockIdx.x) * partitions * EntrySize;

    uint128_t value(0, 0);
    for (int p = group; p < partitions; p += groups)
        value ^= q_partials[static_cast<uint64_t>(p) * EntrySize + component];

    smem[group * EntrySize + component] = value;
    __syncthreads();

    for (int idx = tid; idx < EntrySize; idx += blockDim.x)
    {
        uint128_t total(0, 0);
        for (int g = 0; g < groups; ++g)
            total ^= smem[g * EntrySize + idx];
        out[static_cast<uint64_t>(blockIdx.x) * EntrySize + idx] = total;
    }
}

template <int EntrySize>
__global__ __launch_bounds__(256, 1)
void sector_mul_red_row_tiled_kernel(const uint128_t *__restrict__ se,
                                     const uint128_t *__restrict__ db,
                                     uint128_t *partials,
                                     uint64_t rows, uint64_t per_block)
{
    sector_mul_red_row_body<EntrySize>(se, db, partials, rows, per_block);
}

template <int EntrySize, int LanesPerRow>
__global__ __launch_bounds__(256, 1)
void sector_mul_red_lanes_kernel(const uint128_t *__restrict__ se,
                                 const uint128_t *__restrict__ db,
                                 uint128_t *partials,
                                 uint64_t rows, uint64_t per_block)
{
    sector_mul_red_lanes_body<EntrySize, LanesPerRow>(se, db, partials, rows, per_block);
}

template <int EntrySize>
__global__ __launch_bounds__(256, 1)
void sector_col_mul_red_tiled_kernel(const uint128_t *__restrict__ se,
                                     const uint128_t *__restrict__ db,
                                     uint128_t *partials,
                                     uint64_t rows, uint64_t per_block)
{
    sector_col_mul_red_body<EntrySize>(se, db, partials, rows, per_block);
}

template <int EntrySize>
__global__ void sector_reduce_partials_tiled_kernel(const uint128_t *__restrict__ partials,
                                                    uint128_t *__restrict__ out,
                                                    int partitions)
{
    sector_reduce_partials_body<EntrySize>(partials, out, partitions);
}

template <int EntrySize>
cudaError_t launch_sector_for_entry(int lanes_per_row,
                                    int partitions,
                                    int batch_size,
                                    const uint128_t *se,
                                    const uint128_t *db,
                                    uint128_t *partials,
                                    uint64_t rows,
                                    uint64_t per_block,
                                    cudaStream_t stream)
{
    const dim3 grid(static_cast<unsigned int>(partitions),
                    static_cast<unsigned int>(batch_size));
    const int lanes = (lanes_per_row == 0) ? default_sector_lanes_per_row(EntrySize) : lanes_per_row;

    if (lanes == -1)
    {
        sector_col_mul_red_tiled_kernel<EntrySize><<<grid, 256, 0, stream>>>(se, db, partials, rows, per_block);
        return cudaGetLastError();
    }
    if (lanes == 1)
    {
        sector_mul_red_row_tiled_kernel<EntrySize><<<grid, 256, 0, stream>>>(se, db, partials, rows, per_block);
        return cudaGetLastError();
    }
    if constexpr (EntrySize >= 2)
    {
        if (lanes == 2)
        {
            sector_mul_red_lanes_kernel<EntrySize, 2><<<grid, 256, 0, stream>>>(se, db, partials, rows, per_block);
            return cudaGetLastError();
        }
    }
    if constexpr (EntrySize >= 4)
    {
        if (lanes == 4)
        {
            sector_mul_red_lanes_kernel<EntrySize, 4><<<grid, 256, 0, stream>>>(se, db, partials, rows, per_block);
            return cudaGetLastError();
        }
    }
    if constexpr (EntrySize >= 8)
    {
        if (lanes == 8)
        {
            sector_mul_red_lanes_kernel<EntrySize, 8><<<grid, 256, 0, stream>>>(se, db, partials, rows, per_block);
            return cudaGetLastError();
        }
    }
    if constexpr (EntrySize >= 16)
    {
        if (lanes == 16)
        {
            sector_mul_red_lanes_kernel<EntrySize, 16><<<grid, 256, 0, stream>>>(se, db, partials, rows, per_block);
            return cudaGetLastError();
        }
    }
    if constexpr (EntrySize >= 32)
    {
        if (lanes == 32)
        {
            sector_mul_red_lanes_kernel<EntrySize, 32><<<grid, 256, 0, stream>>>(se, db, partials, rows, per_block);
            return cudaGetLastError();
        }
    }
    return cudaErrorInvalidValue;
}
} // namespace

// Backward-compatible direct kernel entry points (entry_size == 16).
__global__ __launch_bounds__(256, 1)
void sector_mul_red_row_kernel(const uint128_t *__restrict__ se,
                               const uint128_t *__restrict__ db,
                               uint128_t *partials,
                               uint64_t rows, uint64_t per_block)
{
    sector_mul_red_row_body<entry_size>(se, db, partials, rows, per_block);
}

__global__ __launch_bounds__(256, 1)
void sector_mul_red_kernel(const uint128_t *__restrict__ se,
                           const uint128_t *__restrict__ db,
                           uint128_t *partials,
                           uint64_t rows, uint64_t per_block)
{
    sector_mul_red_lanes_body<entry_size, entry_size <= 32 ? entry_size : 32>(se, db, partials, rows, per_block);
}

__global__ __launch_bounds__(256, 1)
void sector_mul_red4_kernel(const uint128_t *__restrict__ se,
                            const uint128_t *__restrict__ db,
                            uint128_t *partials,
                            uint64_t rows, uint64_t per_block)
{
    sector_mul_red_lanes_body<entry_size, default_sector_lanes_per_row(entry_size)>(se, db, partials, rows, per_block);
}

__global__ __launch_bounds__(256, 1)
void sector_col_mul_red_kernel(const uint128_t *__restrict__ se,
                               const uint128_t *__restrict__ db,
                               uint128_t *partials,
                               uint64_t rows, uint64_t per_block)
{
    sector_col_mul_red_body<entry_size>(se, db, partials, rows, per_block);
}

__global__ void sector_reduce_partials_kernel(const uint128_t *__restrict__ partials,
                                              uint128_t *__restrict__ out,
                                              int partitions)
{
    sector_reduce_partials_body<entry_size>(partials, out, partitions);
}

cudaError_t launch_sector_mul_red(int entry_words,
                                  int lanes_per_row,
                                  int partitions,
                                  int batch_size,
                                  const uint128_t *se,
                                  const uint128_t *db,
                                  uint128_t *partials,
                                  uint64_t rows,
                                  uint64_t per_block,
                                  cudaStream_t stream)
{
    if (partitions <= 0 || batch_size <= 0 || rows == 0 || per_block == 0)
        return cudaErrorInvalidValue;

    switch (entry_words)
    {
    case 2:  return launch_sector_for_entry<2>(lanes_per_row, partitions, batch_size, se, db, partials, rows, per_block, stream);
    case 4:  return launch_sector_for_entry<4>(lanes_per_row, partitions, batch_size, se, db, partials, rows, per_block, stream);
    case 8:  return launch_sector_for_entry<8>(lanes_per_row, partitions, batch_size, se, db, partials, rows, per_block, stream);
    case 16: return launch_sector_for_entry<16>(lanes_per_row, partitions, batch_size, se, db, partials, rows, per_block, stream);
    case 32: return launch_sector_for_entry<32>(lanes_per_row, partitions, batch_size, se, db, partials, rows, per_block, stream);
    case 64: return launch_sector_for_entry<64>(lanes_per_row, partitions, batch_size, se, db, partials, rows, per_block, stream);
    default: return cudaErrorInvalidValue;
    }
}

cudaError_t launch_sector_reduce_partials(int entry_words,
                                          int batch_size,
                                          const uint128_t *partials,
                                          uint128_t *out,
                                          int partitions,
                                          cudaStream_t stream)
{
    if (partitions <= 0 || batch_size <= 0)
        return cudaErrorInvalidValue;

    switch (entry_words)
    {
    case 2:
        sector_reduce_partials_tiled_kernel<2><<<batch_size, 256, 0, stream>>>(partials, out, partitions);
        return cudaGetLastError();
    case 4:
        sector_reduce_partials_tiled_kernel<4><<<batch_size, 256, 0, stream>>>(partials, out, partitions);
        return cudaGetLastError();
    case 8:
        sector_reduce_partials_tiled_kernel<8><<<batch_size, 256, 0, stream>>>(partials, out, partitions);
        return cudaGetLastError();
    case 16:
        sector_reduce_partials_tiled_kernel<16><<<batch_size, 256, 0, stream>>>(partials, out, partitions);
        return cudaGetLastError();
    case 32:
        sector_reduce_partials_tiled_kernel<32><<<batch_size, 256, 0, stream>>>(partials, out, partitions);
        return cudaGetLastError();
    case 64:
        sector_reduce_partials_tiled_kernel<64><<<batch_size, 256, 0, stream>>>(partials, out, partitions);
        return cudaGetLastError();
    default:
        return cudaErrorInvalidValue;
    }
}
