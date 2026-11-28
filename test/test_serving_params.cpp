#include "../mpc_cuda/mpc_core.h"
#include "pir_test_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void check_cuda(cudaError_t error, const char *call, int line)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(call) + " at line " +
                                 std::to_string(line) + ": " + cudaGetErrorString(error));
}

#define TEST_CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

std::vector<uint128_t> make_random_db_words(int db_bits, int entry_words, std::mt19937_64 &rng)
{
    const size_t rows = size_t{1} << db_bits;
    std::vector<uint128_t> db(rows * static_cast<size_t>(entry_words));
    for (size_t i = 0; i < db.size(); ++i)
        db[i] = uint128_t(rng(), rng());
    return db;
}

std::vector<uint64_t> make_edge_and_random_queries(int db_bits, int batch_size, std::mt19937_64 &rng)
{
    const uint64_t rows = uint64_t{1} << db_bits;
    const uint64_t edges[] = {0, 1, 63, 64, 127, 128, rows / 2 - 1, rows / 2, rows - 2, rows - 1};
    std::vector<uint64_t> queries(batch_size);
    for (int q = 0; q < batch_size; ++q)
    {
        if (q < static_cast<int>(sizeof(edges) / sizeof(edges[0])))
            queries[q] = edges[q] & (rows - 1);
        else
            queries[q] = rng() & (rows - 1);
    }
    return queries;
}

void verify_shares_against_db(const uint128_t *res0,
                              const uint128_t *res1,
                              const std::vector<uint128_t> &db,
                              const std::vector<uint64_t> &queries,
                              int entry_words,
                              const std::string &label)
{
    if (res0 == nullptr || res1 == nullptr)
        throw std::runtime_error(label + ": null output pointer");
    for (size_t q = 0; q < queries.size(); ++q)
    {
        for (int j = 0; j < entry_words; ++j)
        {
            const uint128_t combined = res0[q * entry_words + j] ^ res1[q * entry_words + j];
            const uint128_t expected = db[queries[q] * static_cast<size_t>(entry_words) + j];
            if (combined != expected)
            {
                throw std::runtime_error(label + ": reconstruction mismatch at query=" +
                                         std::to_string(q) + " comp=" + std::to_string(j));
            }
        }
    }
}

void verify_share_exact_match(const uint128_t *lhs,
                              const uint128_t *rhs,
                              int batch_size,
                              int entry_words,
                              const std::string &label)
{
    const size_t total = static_cast<size_t>(batch_size) * entry_words;
    for (size_t i = 0; i < total; ++i)
    {
        if (lhs[i] != rhs[i])
        {
            throw std::runtime_error(label + ": share mismatch at query=" +
                                     std::to_string(i / entry_words) + " comp=" +
                                     std::to_string(i % entry_words));
        }
    }
}

// 1. Empty-partition poisoned partials buffer regression test.
void test_empty_partition_poisoning(std::mt19937_64 &rng)
{
    const int n = 10; // 1024 entries = 8 tiles of 128 -> 120 empty partitions when P = 128
    const int batch_size = 33;
    const auto db = make_random_db_words(n, 16, rng);
    auto queries = make_edge_and_random_queries(n, batch_size, rng);
    std::vector<uint8_t> k0(batch_size * dpf_key_bytes(n));
    std::vector<uint8_t> k1(batch_size * dpf_key_bytes(n));
    cudaDPFkeygen(k0.data(), k1.data(), queries.data(), n, n - 7, batch_size);

    PirBatchConfig cfg;
    cfg.entry_words = 16;
    cfg.tile_entries = 128;
    cfg.num_partitions = 128;
    cfg.group_size = 33;
    pir_batch_context ctx = init_pir_batch(batch_size, n, const_cast<uint128_t *>(db.data()), cfg);

    const size_t partial_bytes =
        static_cast<size_t>(ctx.num_blocks) * batch_size * cfg.entry_words * sizeof(uint128_t);

    uint128_t *sync0 = nullptr;
    uint128_t *sync1 = nullptr;
    for (int mode = 0; mode < 2; ++mode)
    {
        // Poison the entire partials buffer before every run to guarantee empty
        // partitions actively overwrite their slots with zero.
        TEST_CUDA_CHECK(cudaMemset(ctx.partials, 0xA5, partial_bytes));
        uint128_t *r0 = test_dpf_pir_batch(k0.data(), ctx, batch_size, mode);
        TEST_CUDA_CHECK(cudaMemset(ctx.partials, 0x5A, partial_bytes));
        uint128_t *r1 = test_dpf_pir_batch(k1.data(), ctx, batch_size, mode);

        verify_shares_against_db(r0, r1, db, queries, 16,
                                 std::string("empty_partition_poison mode=") + std::to_string(mode));
        if (mode == 0)
        {
            sync0 = r0;
            sync1 = r1;
        }
        else
        {
            verify_share_exact_match(sync0, r0, batch_size, 16, "empty_partition sync vs async p0");
            verify_share_exact_match(sync1, r1, batch_size, 16, "empty_partition sync vs async p1");
            free(r0);
            free(r1);
        }
    }
    free(sync0);
    free(sync1);
    free_cuda_memory(ctx);
    std::printf("[PASS] empty_partition_poisoning (n=10, P=128, 120 empty partitions)\n");
}

// 2. Small / unaligned batch decoupling AND large batch B > 512 with B_g grouping.
void test_batch_sizes_and_groups(std::mt19937_64 &rng)
{
    const int n = 12;
    const auto db = make_random_db_words(n, 16, rng);

    struct Case
    {
        int batch_size;
        int group_size;
        int partitions;
    };
    const Case cases[] = {
        {1, 1, 16},
        {2, 2, 16},
        {7, 7, 13},
        {31, 31, 16},
        {33, 33, 17},
        {129, 129, 32},
        {511, 511, 32},
        {513, 256, 32},
        {700, 256, 24},
        {1024, 512, 64},
        {1024, 128, 64},
        {1536, 512, 32},
    };

    for (const Case &tc : cases)
    {
        auto queries = make_edge_and_random_queries(n, tc.batch_size, rng);
        std::vector<uint8_t> k0(static_cast<size_t>(tc.batch_size) * dpf_key_bytes(n));
        std::vector<uint8_t> k1(static_cast<size_t>(tc.batch_size) * dpf_key_bytes(n));
        cudaDPFkeygen(k0.data(), k1.data(), queries.data(), n, n - 7, tc.batch_size);

        PirBatchConfig cfg;
        cfg.entry_words = 16;
        cfg.tile_entries = 128;
        cfg.num_partitions = tc.partitions;
        cfg.group_size = tc.group_size;

        pir_batch_context ctx = init_pir_batch(tc.batch_size, n, const_cast<uint128_t *>(db.data()), cfg);
        uint128_t *s0 = test_dpf_pir_batch(k0.data(), ctx, tc.batch_size, 0);
        uint128_t *s1 = test_dpf_pir_batch(k1.data(), ctx, tc.batch_size, 0);
        uint128_t *a0 = test_dpf_pir_batch(k0.data(), ctx, tc.batch_size, 1);
        uint128_t *a1 = test_dpf_pir_batch(k1.data(), ctx, tc.batch_size, 1);

        const std::string tag = "B=" + std::to_string(tc.batch_size) +
                                " Bg=" + std::to_string(tc.group_size) +
                                " P=" + std::to_string(tc.partitions);
        verify_shares_against_db(s0, s1, db, queries, 16, "sync " + tag);
        verify_shares_against_db(a0, a1, db, queries, 16, "async " + tag);
        verify_share_exact_match(s0, a0, tc.batch_size, 16, "sync==async p0 " + tag);
        verify_share_exact_match(s1, a1, tc.batch_size, 16, "sync==async p1 " + tag);

        free(s0);
        free(s1);
        free(a0);
        free(a1);
        free_cuda_memory(ctx);
    }
    std::printf("[PASS] batch_sizes_and_groups (B=1..1536, Bg=1..512, non-divisible P/B)\n");
}

// 3. Sweep P, T, and record width D (32 B .. 1024 B) on batch-tiled sync/async.
void test_parameter_sweep_batch(std::mt19937_64 &rng)
{
    cudaDeviceProp prop{};
    TEST_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int n = 12;
    const int batch_size = 65; // non-power-of-two batch
    auto queries = make_edge_and_random_queries(n, batch_size, rng);
    std::vector<uint8_t> k0(static_cast<size_t>(batch_size) * dpf_key_bytes(n));
    std::vector<uint8_t> k1(static_cast<size_t>(batch_size) * dpf_key_bytes(n));
    cudaDPFkeygen(k0.data(), k1.data(), queries.data(), n, n - 7, batch_size);

    const int entry_widths[] = {2, 4, 8, 16, 32, 64}; // D = 32, 64, 128, 256, 512, 1024 B
    const int tiles[] = {16, 32, 64, 128, 256};
    const int partitions[] = {1, 3, 16, 64, 200};

    for (int e : entry_widths)
    {
        const auto db = make_random_db_words(n, e, rng);
        std::vector<uint128_t> ref0;
        std::vector<uint128_t> ref1;

        for (int t : tiles)
        {
            if (!batch_tiled_config_supported(false, e, t, prop.sharedMemPerBlockOptin))
                continue;

            const bool async_ok = batch_tiled_config_supported(true, e, t, prop.sharedMemPerBlockOptin);
            const int p = partitions[(e + t) % 5];

            PirBatchConfig cfg;
            cfg.entry_words = e;
            cfg.tile_entries = t;
            cfg.num_partitions = p;
            cfg.group_size = 32;

            pir_batch_context ctx = init_pir_batch(batch_size, n, const_cast<uint128_t *>(db.data()), cfg);
            const std::string tag = "E=" + std::to_string(e) + " (D=" + std::to_string(e * 16) +
                                    "B) T=" + std::to_string(t) + " P=" + std::to_string(p);

            uint128_t *s0 = test_dpf_pir_batch(k0.data(), ctx, batch_size, 0);
            uint128_t *s1 = test_dpf_pir_batch(k1.data(), ctx, batch_size, 0);
            verify_shares_against_db(s0, s1, db, queries, e, "batch sync " + tag);

            if (ref0.empty())
            {
                ref0.assign(s0, s0 + static_cast<size_t>(batch_size) * e);
                ref1.assign(s1, s1 + static_cast<size_t>(batch_size) * e);
            }
            else
            {
                verify_share_exact_match(ref0.data(), s0, batch_size, e, "cross-T/P share p0 " + tag);
                verify_share_exact_match(ref1.data(), s1, batch_size, e, "cross-T/P share p1 " + tag);
            }

            if (async_ok)
            {
                uint128_t *a0 = test_dpf_pir_batch(k0.data(), ctx, batch_size, 1);
                uint128_t *a1 = test_dpf_pir_batch(k1.data(), ctx, batch_size, 1);
                verify_shares_against_db(a0, a1, db, queries, e, "batch async " + tag);
                verify_share_exact_match(s0, a0, batch_size, e, "sync==async p0 " + tag);
                verify_share_exact_match(s1, a1, batch_size, e, "sync==async p1 " + tag);
                free(a0);
                free(a1);
            }
            else
            {
                // Verify async fails gracefully with nullptr when SMEM exceeds device limit.
                uint128_t *should_be_null = test_dpf_pir_batch(k0.data(), ctx, batch_size, 1);
                if (should_be_null != nullptr)
                {
                    free(should_be_null);
                    free(s0);
                    free(s1);
                    free_cuda_memory(ctx);
                    throw std::runtime_error("expected async to be rejected for " + tag);
                }
            }

            free(s0);
            free(s1);
            free_cuda_memory(ctx);
        }
    }
    std::printf("[PASS] parameter_sweep_batch (D=32..1024B, T=16..256, P=1..200, sync/async)\n");
}

// 4. Sweep D, P, lanes_per_row L, and small-batch B on low-latency sector_view path.
void test_parameter_sweep_low_latency(std::mt19937_64 &rng)
{
    const int n = 12;
    const int entry_widths[] = {2, 4, 8, 16, 32, 64}; // D = 32..1024 B
    const int batches[] = {1, 5};
    const int lane_candidates[] = {0, 1, 2, 4, 8, 16, 32};

    for (int e : entry_widths)
    {
        const auto db = make_random_db_words(n, e, rng);
        for (int b : batches)
        {
            auto queries = make_edge_and_random_queries(n, b, rng);
            std::vector<uint8_t> k0(static_cast<size_t>(b) * dpf_key_bytes(n));
            std::vector<uint8_t> k1(static_cast<size_t>(b) * dpf_key_bytes(n));
            cudaDPFkeygen(k0.data(), k1.data(), queries.data(), n, n - 7, b);

            std::vector<uint128_t> ref0;
            std::vector<uint128_t> ref1;

            for (int lanes : lane_candidates)
            {
                if (lanes != 0 && !is_supported_sector_lanes(e, lanes))
                    continue;

                PirLowLatencyConfig cfg;
                cfg.entry_words = e;
                cfg.num_partitions = (lanes == 0) ? 0 : 19; // also exercise non-power-of-2 P
                cfg.lanes_per_row = lanes;

                PirContext ctx = init_pir_low_latency(b, n, const_cast<uint128_t *>(db.data()), cfg);
                uint128_t *r0 = test_dpf_pir_low_latency(k0.data(), ctx, b);
                uint128_t *r1 = test_dpf_pir_low_latency(k1.data(), ctx, b);

                const std::string tag = "low_latency E=" + std::to_string(e) +
                                        " B=" + std::to_string(b) +
                                        " L=" + std::to_string(lanes);
                verify_shares_against_db(r0, r1, db, queries, e, tag);
                if (ref0.empty())
                {
                    ref0.assign(r0, r0 + static_cast<size_t>(b) * e);
                    ref1.assign(r1, r1 + static_cast<size_t>(b) * e);
                }
                else
                {
                    verify_share_exact_match(ref0.data(), r0, b, e, tag + " p0");
                    verify_share_exact_match(ref1.data(), r1, b, e, tag + " p1");
                }

                free(r0);
                free(r1);
                free_cuda_memory(ctx);
            }
        }
    }
    std::printf("[PASS] parameter_sweep_low_latency (D=32..1024B, L=1..32, B=1 and B=5)\n");
}
} // namespace

int main()
{
    try
    {
        if (!has_cuda_device())
        {
            std::printf("SKIP test_serving_params: no available CUDA GPU\n");
            return 77;
        }
        std::mt19937_64 rng(0x20261128ULL);
        test_empty_partition_poisoning(rng);
        test_batch_sizes_and_groups(rng);
        test_parameter_sweep_batch(rng);
        test_parameter_sweep_low_latency(rng);
        TEST_CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("[PASS] test_serving_params all checks passed\n");
        return 0;
    }
    catch (const std::exception &ex)
    {
        std::fprintf(stderr, "[FAIL] test_serving_params: %s\n", ex.what());
        return 1;
    }
}
