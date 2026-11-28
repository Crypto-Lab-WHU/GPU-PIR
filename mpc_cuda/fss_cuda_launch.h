#pragma once

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "../mpc_keys/uint128_type.h"

#include "dcf_keys.h"

__global__ void fss_genaeskey_kernel(uint32_t key[4 * (14 + 1)]);
__global__ void dpf_gen_kernel(uint32_t *key, uint64_t *alpha, int n, DCF_Keys k0, DCF_Keys k1, int N, int maxlayer);
__global__ void EvalAll_LBLEvaluation(uint32_t *key, int layer, int maxlayer, int N, unsigned char *k, uint128_t *s, uint32_t *t);
__global__ void EvalAll_BlockEvaluation(uint32_t *key, int layer_len, int layer, unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *res_s, uint32_t *res_t);
__global__ void EvalAll_BlockEvaluation_Se(uint32_t *key, int layer_len, int layer, unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *se);
__global__ void EvalAll_lastlayer(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *res);
__global__ void MultiplicationReduction_grid_col(unsigned char *k, uint128_t *se, const uint128_t *d_db, uint128_t *blocksum_cuda, size_t block_num, int N);
__global__ void MultiplicationReduction_grid_row(unsigned char *k, uint128_t *se, const uint128_t *d_db, uint128_t *blocksum_cuda, size_t block_num, int N);
__global__ void batch_mul_red_sync_kernel(const uint128_t *se, const uint128_t *db, uint128_t *partials, int db_bits, uint64_t per_block, int batch_size);
__global__ void batch_mul_red_async_kernel(const uint128_t *se, const uint128_t *db, uint128_t *partials, int db_bits, uint64_t per_block, int batch_size);
__global__ void batch_reduce_partials_kernel(const uint128_t *partials, uint128_t *out, int num_partitions, int batch_size);
__global__ void sector_mul_red_row_kernel(const uint128_t *se, const uint128_t *db, uint128_t *partials, uint64_t rows, uint64_t per_block);
__global__ void sector_mul_red_kernel(const uint128_t *se, const uint128_t *db, uint128_t *partials, uint64_t rows, uint64_t per_block);
__global__ void sector_mul_red4_kernel(const uint128_t *se, const uint128_t *db, uint128_t *partials, uint64_t rows, uint64_t per_block);
__global__ void sector_col_mul_red_kernel(const uint128_t *se, const uint128_t *db, uint128_t *partials, uint64_t rows, uint64_t per_block);
__global__ void sector_reduce_partials_kernel(const uint128_t *partials, uint128_t *out, int partitions);
__global__ void MultiplicationReduction_LUT_new(unsigned char *k, uint128_t *se, const uint32_t *d_db, uint32_t *blocksum_cuda, size_t block_num, int N);
__global__ void MultiplicationReduction_LUT(unsigned char *k, uint128_t *se, const uint32_t *d_db, uint32_t *blocksum_cuda, size_t block_num);
__global__ void EVAL_Pack_last_layer_gense(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *se);
__global__ void EvalAll_SeGeneration(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *se);
__global__ void BlockReduction(uint128_t *blocksum_cuda, uint128_t *k_res);
__global__ void BlockReduction_LUT(uint32_t *blocksum_cuda, uint32_t *k_res);
__global__ void seed_copy(uint128_t *d_s, uint32_t *d_t, uint8_t *key, int maxlayer, int N);

// Parameterized host dispatchers for batch-tiled (P, T, B_g, D) and sector-view (P, L, B, D) kernels.
size_t batch_tiled_smem_bytes(bool async_mode, int entry_words, int tile_entries);
bool batch_tiled_config_supported(bool async_mode, int entry_words, int tile_entries, size_t max_smem_optin);
cudaError_t configure_batch_tiled_smem(bool async_mode, int entry_words, int tile_entries, size_t max_smem_optin);
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
                                 cudaStream_t stream = nullptr);
cudaError_t launch_batch_reduce_partials(int entry_words,
                                         const uint128_t *partials,
                                         uint128_t *out,
                                         int num_partitions,
                                         int batch_size,
                                         cudaStream_t stream = nullptr);

cudaError_t launch_sector_mul_red(int entry_words,
                                  int lanes_per_row,
                                  int partitions,
                                  int batch_size,
                                  const uint128_t *se,
                                  const uint128_t *db,
                                  uint128_t *partials,
                                  uint64_t rows,
                                  uint64_t per_block,
                                  cudaStream_t stream = nullptr);
cudaError_t launch_sector_reduce_partials(int entry_words,
                                          int batch_size,
                                          const uint128_t *partials,
                                          uint128_t *out,
                                          int partitions,
                                          cudaStream_t stream = nullptr);
