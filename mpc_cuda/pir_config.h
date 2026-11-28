#pragma once

#ifndef PIR_ENTRY_SIZE
#define PIR_ENTRY_SIZE 16
#endif

constexpr int entry_size = PIR_ENTRY_SIZE;
constexpr int k_batch_tile_entries = 128;
constexpr int k_max_batch_group_size = 512;
constexpr int k_min_batch_loader_threads = 128;

constexpr bool is_supported_entry_words(int e)
{
    return e == 2 || e == 4 || e == 8 || e == 16 || e == 32 || e == 64;
}

constexpr bool is_supported_tile_entries(int t)
{
    return t == 16 || t == 32 || t == 64 || t == 128 || t == 256;
}

constexpr int default_batch_tile_entries(int e)
{
    return e <= 16 ? 128 : (e == 32 ? 64 : 32);
}

constexpr int default_sector_lanes_per_row(int e)
{
    return (e >= 16) ? (e / 4) : ((e >= 4) ? 2 : e);
}

constexpr bool is_supported_sector_lanes(int e, int lanes)
{
    return lanes > 0 && lanes <= 32 && ((lanes & (lanes - 1)) == 0) && (e % lanes == 0);
}
