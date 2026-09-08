#ifndef PREFIX_GROUPER_NPU_SHARED_PREFIX_ATTENTION_TILING_DATA_H
#define PREFIX_GROUPER_NPU_SHARED_PREFIX_ATTENTION_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

constexpr uint32_t kSharedPrefixRowAlignment = 16; // Independent 64-byte FP32 rows.
constexpr uint32_t kSharedPrefixSlots = 2;

struct SharedPrefixAttentionTilingData {
    uint64_t total_tokens;
    uint64_t q_heads;
    uint64_t kv_heads;
    uint64_t head_dim;
    uint64_t padded_dim;
    uint64_t task_count;
    uint64_t core_workspace_bytes;
    uint32_t tile;
    uint32_t vector_cores;
    float scale;
    uint32_t reserved;
    AscendC::tiling::TCubeTiling score_mm;
    AscendC::tiling::TCubeTiling value_mm;
    AscendC::tiling::TCubeTiling transpose_mm;
};

struct SharedPrefixVectorTilingData {
    uint64_t rows;
    uint64_t head_dim;
    uint64_t padded_dim;
    uint64_t tasks;
    uint32_t tile;
    uint32_t cores;
};

#endif
