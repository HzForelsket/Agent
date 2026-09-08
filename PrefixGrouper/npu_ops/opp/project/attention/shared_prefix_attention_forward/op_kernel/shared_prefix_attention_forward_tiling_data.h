#ifndef PREFIX_GROUPER_FORWARD_TILING_DATA_H
#define PREFIX_GROUPER_FORWARD_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"
#include <cstdint>
constexpr uint32_t kForwardMaxVectorCores = 64;
struct ForwardTaskRange {
    uint64_t query_start, query_head, task_count;
};
struct SharedPrefixAttentionForwardTilingData {
    uint64_t total_tokens, q_heads, kv_heads, head_dim, padded_dim;
    uint64_t task_count, core_workspace_bytes;
    uint32_t q_tile, kv_tile, vector_cores, softmax_tmp_bytes;
    float scale;
    uint32_t reserved;
    AscendC::tiling::TCubeTiling score_mm, value_mm;
    ForwardTaskRange ranges[kForwardMaxVectorCores];
};
#endif
