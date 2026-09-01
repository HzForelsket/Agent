#ifndef PREFIX_GROUPER_NPU_SHARED_PREFIX_ATTENTION_TILING_H
#define PREFIX_GROUPER_NPU_SHARED_PREFIX_ATTENTION_TILING_H

#include <cstdint>

struct SharedPrefixAttentionTilingData {
    uint32_t total_tokens;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t task_count;
    float scale;
};

#endif
