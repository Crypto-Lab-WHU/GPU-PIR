#pragma once

#include "pir_config.h"

struct PirBatchConfig
{
    int entry_words = entry_size; // E = D / 16; supported: 2, 4, 8, 16, 32, 64 (D = 32..1024 B)
    int tile_entries = 0;         // T; 0 = auto; supported: 16, 32, 64, 128, 256
    int num_partitions = 0;       // P; 0 = device SM count; >= 1
    int group_size = 0;           // B_g; 0 = min(batch_size, 512); 1..512
};

struct PirLowLatencyConfig
{
    int entry_words = entry_size; // E = D / 16; supported: 2, 4, 8, 16, 32, 64 (D = 32..1024 B)
    int num_partitions = 0;       // P; 0 = auto (min(SM count, max(1, rows / 128)))
    int lanes_per_row = 0;        // L; 0 = auto, -1 = column-major, 1 = row, or power-of-2 dividing gcd(32, E)
};

struct PirContext
{
    void *db = nullptr;
    void *output = nullptr;
    void *partials = nullptr; // [batch_size][num_partitions][entry_words], low-latency path
    int num_partitions = 0;
    void *key = nullptr;
    void *seed = nullptr;
    void *bit = nullptr;
    void *seed_intermediate = nullptr;
    void *bit_intermediate = nullptr;
    void *selection = nullptr;
    void *aes_key = nullptr;
    int entry_words = entry_size;
    int lanes_per_row = 0;
    int max_batch_size = 0;
    int db_bits = 0;
};

struct pir_batch_context
{
    void *db = nullptr;
    void *partials = nullptr; // [num_blocks][batch_size][entry_words]
    void *output = nullptr;   // [batch_size][entry_words]
    void *key = nullptr;
    void *seed = nullptr;
    void *bit = nullptr;
    void *seed_intermediate = nullptr;
    void *bit_intermediate = nullptr;
    void *selection = nullptr;
    void *aes_key = nullptr;
    int num_blocks = 0;
    int entry_words = entry_size;
    int tile_entries = k_batch_tile_entries;
    int group_size = 0;
    int max_batch_size = 0;
    int db_bits = 0;
    bool async_supported = true;
};

struct PirPipelineContext
{
    void *db = nullptr;
    void *partials = nullptr;
    void *output = nullptr;
    void *seed = nullptr;
    void *bit = nullptr;
    void *seed_intermediate = nullptr;
    void *bit_intermediate = nullptr;
    void *seed_result = nullptr;
    void *bit_result = nullptr;
    void *selection = nullptr;
    void *aes_key = nullptr;
};

struct PirLutContext
{
    void *db = nullptr;
    void *input = nullptr;
    void *intermediate = nullptr;
    void *output = nullptr;
    void *seed = nullptr;
    void *bit = nullptr;
    void *seed_intermediate = nullptr;
    void *bit_intermediate = nullptr;
    void *seed_result = nullptr;
    void *bit_result = nullptr;
    void *selection = nullptr;
    void *aes_key = nullptr;
};

struct PirStreamContext
{
    void *streams = nullptr;
    void *events = nullptr;
};
