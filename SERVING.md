# Serving Extension: Low-Latency and High-Throughput PIR

This document describes the post-paper extension included in this repository. It builds on
the DPF-PIR artifact below and is **not** part of the paper's evaluated results; the
benchmarks in `README.md` (Table 2, Figure 11) are unaffected.

The extension adds:

- an optimized GGM tree evaluation that generates selection-vector shares in the final
  tree layer instead of a separate pass;
- a **low-latency** kernel path for a single query, which spreads the database scan over
  all SMs using sector-view conditional loads;
- a **high-throughput** kernel path for batched queries, which stages the database in
  shared memory and reuses each tile across the whole batch (synchronous or `cp.async`
  double-buffered loading).

Both paths keep the row-major database layout, the DPF key format and the API shape of
the original artifact.

## Design

```text
                     Optimized GGM tree evaluation
                                  │
                       generate selection shares
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
        single query / small batch        medium / large batch
                  │                               │
      multi-SM sector-view loads        shared-memory DB tiling
        (only selected rows)                (128-entry tiles)
                  │                               │
     per-block partials + reduce       sync / async load & reuse
                  │                               │
           low execution latency            high throughput
```

### Optimized GGM tree evaluation

The original flow evaluated the whole GGM tree, wrote the final `(s, t)` nodes, and then
packed them into the selection-vector shares `se`. The serving paths fuse the two steps:
the final tree layer applies the leaf block directly and XORs it into `se` while
expanding (`EvalAll_BlockEvaluation_Se`). The single-query and batch paths therefore no
longer allocate the `s_res` / `t_res` intermediate buffers and no longer run a separate
selection-generation pass; the pipeline and LUT paths keep the original two-step flow.

`test_fused_eval` checks the fused output against a CPU reference over edge-case indices,
single-query repeats, and both prefix/final-layer splits (`n = 8..20`, batch up to 129).

### Low-latency path: multi-SM sector-view conditional access

For a single query (`N == 1`), the database is partitioned into contiguous row ranges,
one CUDA block per range, and the number of partitions is derived from the device's SM
count (capped by the number of 128-row groups). Each block only loads rows whose
selection bit is set, emits one partial per block, and a small second kernel XOR-reduces
the partials. The default kernel (`sector_mul_red4_kernel`, four lanes per row at
`entry_size = 16`) reads 64 contiguous bytes (two 32-byte sectors) per lane and advances
eight rows per warp. With the row-major layout a selected row costs exactly its payload
(1.0x); the column-major variant is kept for comparison but transfers about 1.5x at 50%
row density.

The parameterized low-latency entry points (`init_pir_low_latency` /
`test_dpf_pir_low_latency` with `PirLowLatencyConfig`) generalize this path across record
widths (`entry_words` in `{2, 4, 8, 16, 32, 64}`, i.e., 32–1024 B), partition counts `P`,
row-to-lane mappings `lanes_per_row`, and small batches via `gridDim.y`.

### High-throughput path: shared-memory database tiling

For batched queries (`init_pir_batch`), each block owns a contiguous database partition
(`blockIdx.x`) and a cooperative query group (`blockIdx.y`, up to `group_size <= 512`
queries per group), walking its partition in tiles of `tile_entries` rows (default 128
for `entry_words <= 16`, 64 for 512 B entries, 32 for 1024 B entries). Loader threads are
decoupled from `batch_size` (`clamp(round_up_32(min(B, B_g)), 128, 512)`) so small and
non-warp-aligned batches still issue full coalesced tile loads, while active query
threads consume each staged tile via uniform shared-memory broadcast. Empty partitions
explicitly write zero partials so reused buffers cannot retain stale values. Two variants
are provided:

- **sync** (`mode = 0`): one shared-memory tile, blocking loads;
- **async** (`mode = 1`): double-buffered tiles prefetched with `cp.async`
  (`cuda_pipeline`), automatically validated against `sharedMemPerBlockOptin`.

Both variants write `[partition][query][component]` partials and share one reduce kernel.
The batch path uses its own context (`pir_batch_context`), supports arbitrary
`batch_size >= 1` via cooperative groups, and exposes `(entry_words, tile_entries,
num_partitions, group_size)` through `PirBatchConfig`.

### Choosing a path

- single query or small batch, latency matters → low-latency path (`init_pir` with
  `N = 1`, or `init_pir_low_latency`);
- medium / large batch, throughput matters → batch path (`init_pir_batch` with
  `batch_size` queries, sync or async);
- `bench_serving` measures both on the same database and prints the latency percentiles,
  the batched throughput, and the per-request throughput ratio.

## Added Files

| File | Contents |
|------|----------|
| `mpc_cuda/sector_view.cu` | parameterized low-latency row/sector/lanes/column conditional-access kernels and partial reduce |
| `mpc_cuda/batch_tiled.cu` | parameterized sync/async tiled kernels (`P`, `T`, `B_g`, `D`) and batch reduce |
| `test/test_fused_eval.cu` | fused selection generation vs CPU reference |
| `test/test_serving_params.cpp` | regression sweep over empty partitions, `B > 512`, `B_g`, `P`, `T`, `D`, and `lanes_per_row` |
| `test/bench_sector_view.cu` | single-query kernel study from one SM to multi-SM |
| `test/bench_serving.cpp` | low-latency vs high-throughput comparison |

`mpc_cuda/fss_cuda_kernels.cu` gains the fused `EvalAll_BlockEvaluation_Se` path and
64-bit batch index arithmetic, and `mpc_cuda/pir_config.h` holds default constants and
parameter validation helpers.

## Building

The extension uses the same build as the rest of the repository; see
[Build Instructions](README.md#build-instructions) in `README.md`. New targets are
picked up automatically by `test/CMakeLists.txt`.

## Running the New Drivers

All drivers are built under `build/test` and skip cleanly with a `SKIP` line when no CUDA
device is visible.

| Binary | Purpose | Usage | Defaults |
|--------|---------|-------|----------|
| `test_fused_eval` | fused selection generation vs CPU reference | `./build/test/test_fused_eval` | fixed case matrix |
| `test_serving_params` | parameterized batch & low-latency regression suite | `./build/test/test_serving_params` | fixed sweep matrix |
| `bench_sector_view` | single-query kernel variants from one SM to multi-SM | `./build/test/bench_sector_view [n] [samples]` | `20`, `31` |
| `bench_serving` | low-latency vs high-throughput comparison | `./build/test/bench_serving [n] [batch] [low_samples] [batch_samples]` | `20`, `512`, `31`, `5` |

Examples:

```bash
./build/test/test_serving_params
./build/test/bench_sector_view 24 11
./build/test/bench_serving 24 512 51 7
```

The existing drivers also cover the new paths: `test_pir` adds `test_pir_single`,
`test_pir_batch_sync` and `test_pir_batch_async`, and `bench_pir` adds the
`DPF-PIR batch sync` and `DPF-PIR batch async` labels next to the original modes.

### Public API

```cpp
// Low-latency path: N == 1 uses the multi-SM sector-view kernels.
PirContext ctx = init_pir(N, n, db);
uint128_t *out = test_dpf_pir(key, ctx, N);
free_cuda_memory(ctx);

// Parameterized low-latency path (supports D = 32..1024 B, custom P, lanes_per_row, and small B).
PirLowLatencyConfig lcfg{/*entry_words=*/16, /*num_partitions=*/0, /*lanes_per_row=*/4};
PirContext lctx = init_pir_low_latency(batch_size, n, db, lcfg);
uint128_t *lout = test_dpf_pir_low_latency(key, lctx, batch_size);
free_cuda_memory(lctx);

// High-throughput path: batch of queries (mode 0 = sync, 1 = async; supports B > 512).
PirBatchConfig bcfg{/*entry_words=*/16, /*tile_entries=*/128, /*num_partitions=*/0, /*group_size=*/512};
pir_batch_context bctx = init_pir_batch(batch_size, n, db, bcfg);
uint128_t *out = test_dpf_pir_batch(key, bctx, batch_size, mode);
free_cuda_memory(bctx);
```

`init_pir_batch` allocates the batch partials/output buffers and configures opt-in shared
memory when a sync or async configuration exceeds 48 KB; the row-major and batch paths
are benchmarked with separate contexts because their memory footprints differ.

## Output Labels

`bench_sector_view` prints one `RESULT` line per variant (baseline single SM, multi-SM
row, sector, sector4, column-major, and the batched shared-memory sync/async kernels at
`batch = 1`), plus `SAMPLE,n,variant,ms` lines for plotting.

`bench_serving` prints:

```text
LOW_LATENCY n=... samples=... p50_ms=... p95_ms=... p99_ms=... min_ms=... max_ms=...
THROUGHPUT n=... B=... mode=sync  samples=... median_ms=... p99_ms=... pirs_per_s=... per_request_us=...
THROUGHPUT n=... B=... mode=async samples=... median_ms=... p99_ms=... pirs_per_s=... per_request_us=...
COMPARISON n=... single_p50_ms=... single_p99_ms=... batch_async_p50_ms=... batch_async_p99_ms=... async_per_request_us=... throughput_ratio=...
```

Both paths are verified against the reconstructed database before timing;
`throughput_ratio` compares batched async throughput with single-query throughput at the
same `n`.

## Constraints

- `n >= 8`; the packed DPF depth is `maxlayer = n - 7`.
- Batch path: supports any `batch_size >= 1` via cooperative groups (`1 <= group_size <= 512`),
  `entry_words` in `{2, 4, 8, 16, 32, 64}` (32–1024 B), and `tile_entries` in
  `{16, 32, 64, 128, 256}` (with `2^n >= tile_entries`).
- Shared memory per batch block: `tile_entries * entry_words * 16` bytes (`sync`) or
  `2 * tile_entries * entry_words * 16` bytes (`async`, double-buffered). `init_pir_batch`
  validates against `sharedMemPerBlockOptin` and requests opt-in shared memory when > 48 KB;
  if only `sync` fits, `mode = 1` returns `nullptr` cleanly instead of aborting.
- `bench_serving` accepts `n = 8..24`, `batch <= 512` and `batch <= 2^n`.
- Default constants: `entry_size = 16` and `k_batch_tile_entries = 128` in
  `mpc_cuda/pir_config.h`; `NUM_STREAMS = 10` and `NUM_CHUNKS = 32768` in
  `test/CMakeLists.txt`.
