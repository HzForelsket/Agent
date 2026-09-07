#ifndef PREFIX_GROUPER_NPU_SHARED_PREFIX_ATTENTION_HOST_H
#define PREFIX_GROUPER_NPU_SHARED_PREFIX_ATTENTION_HOST_H

#include "../op_kernel/shared_prefix_attention_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace shared_prefix_host {
inline bool Multiply(uint64_t a, uint64_t b, uint64_t& result)
{
    if (b && a > static_cast<uint64_t>(INT64_MAX) / b) return false;
    result = a * b;
    return true;
}
inline uint64_t Align(uint64_t n, uint64_t alignment = kSharedPrefixRowAlignment)
{
    return (n + alignment - 1) / alignment * alignment;
}
inline bool Shape(const gert::Shape& q, const gert::Shape& k)
{
    return q.GetDimNum() == 3 && k.GetDimNum() == 3 && q.GetDim(0) > 0 &&
        q.GetDim(0) <= INT32_MAX && q.GetDim(0) == k.GetDim(0) &&
        q.GetDim(1) > 0 && k.GetDim(1) > 0 && q.GetDim(1) % k.GetDim(1) == 0 &&
        q.GetDim(2) > 0 && q.GetDim(2) <= INT64_MAX - 15 && q.GetDim(2) == k.GetDim(2);
}
inline bool Matmul(platform_ascendc::PlatformAscendC& platform, uint32_t tile,
                   bool transA, bool transB, AscendC::tiling::TCubeTiling& result)
{
    matmul_tiling::MatmulApiTiling mm(platform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BF16, transA);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BF16, transB);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT);
    mm.SetShape(tile, tile, tile);
    mm.SetOrgShape(tile, tile, tile);
    mm.SetFixSplit(tile, tile, tile);
    mm.EnableBias(false);
    uint64_t ub = 0, l1 = 0, l0c = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    const uint64_t liveUb = 34ULL * tile * tile + 200ULL * tile;
    if (ub <= liveUb + 4096) return false;
    mm.SetBufferSpace(l1 / 3, l0c / 3, (ub - liveUb - 4096) / 3);
    return mm.GetTiling(result) == 0;
}
inline ge::graphStatus AttentionTiling(gert::TilingContext* context, bool backward)
{
    const uint32_t qi = backward ? 1 : 0, ki = qi + 1;
    auto* q = context->GetInputShape(qi);
    auto* k = context->GetInputShape(ki);
    auto* attrs = context->GetAttrs();
    auto* info = context->GetPlatformInfo();
    if (!q || !k || !attrs || !info || !Shape(q->GetStorageShape(), k->GetStorageShape()))
        return ge::GRAPH_FAILED;
    auto* v = context->GetInputShape(ki + 1);
    if (!v || v->GetStorageShape() != k->GetStorageShape()) return ge::GRAPH_FAILED;
    for (uint32_t i = backward ? 6 : 3; i < (backward ? 11 : 8); ++i) {
        auto* meta = context->GetInputShape(i);
        if (!meta || meta->GetStorageShape().GetDimNum() != 1 ||
            meta->GetStorageShape().GetDim(0) != q->GetStorageShape().GetDim(0)) return ge::GRAPH_FAILED;
    }
    auto* scale = attrs->GetAttrPointer<float>(0);
    if (!scale || !std::isfinite(*scale) || *scale <= 0) return ge::GRAPH_FAILED;
    auto* t = context->GetTilingData<SharedPrefixAttentionTilingData>();
    *t = {};
    t->total_tokens = q->GetStorageShape().GetDim(0);
    t->q_heads = q->GetStorageShape().GetDim(1);
    t->kv_heads = k->GetStorageShape().GetDim(1);
    t->head_dim = q->GetStorageShape().GetDim(2);
    t->padded_dim = Align(t->head_dim);
    t->scale = *scale;
    uint64_t size;
    if (!Multiply(t->total_tokens, t->q_heads, size) ||
        !Multiply(size, t->padded_dim, size) || !Multiply(size, sizeof(float), size))
        return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(info);
    uint64_t ub = 0, l1 = 0, l0a = 0, l0b = 0, l0c = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    // Budget all live queues/state, with headroom for Matmul/TPipe, not a model-specific table.
    uint32_t tile = 16;
    auto fits = [&](uint64_t b) {
        return 36 * b * b + 256 * b + 16 * 1024 <= ub &&
            4 * b * b <= l1 && 2 * b * b <= l0a && 2 * b * b <= l0b && 4 * b * b <= l0c;
    };
    if (!fits(tile) || !platform.GetCoreNumAic() || !platform.GetCoreNumAiv()) return ge::GRAPH_FAILED;
    while (fits(static_cast<uint64_t>(tile) * 2)) tile *= 2;
    while (tile > 16 && tile / 2 >= t->total_tokens) tile /= 2;
    for (;;) {
        if (Matmul(platform, tile, false, true, t->score_mm) &&
            Matmul(platform, tile, false, false, t->value_mm) &&
            Matmul(platform, tile, true, false, t->transpose_mm)) {
            const uint64_t apiUb = static_cast<uint64_t>(std::max(0, t->score_mm.shareUbSize)) +
                std::max(0, t->value_mm.shareUbSize) + std::max(0, t->transpose_mm.shareUbSize);
            if (34ULL * tile * tile + 200ULL * tile + apiUb + 4096 <= ub) break;
        }
        if (tile == 16) return ge::GRAPH_FAILED;
        tile /= 2;
    }
    t->tile = tile;
    uint64_t heads = t->q_heads;
    if (backward) {
        if (heads > static_cast<uint64_t>(INT64_MAX) - t->kv_heads) return ge::GRAPH_FAILED;
        heads += t->kv_heads;
    }
    if (!Multiply((t->total_tokens + tile - 1) / tile, heads, t->task_count)) return ge::GRAPH_FAILED;
    const uint32_t cubes = std::min<uint64_t>((t->task_count + 1) / 2,
        std::min(platform.GetCoreNumAic(), platform.GetCoreNumAiv() / 2));
    if (!cubes) return ge::GRAPH_FAILED;
    t->vector_cores = cubes * 2; // CANN MIX_AIC_1_2 launch contract.
    // Two BF16 A/B + FP32 C slots, and two BF16 probability/gradient blocks.
    t->core_workspace_bytes = 20ULL * tile * tile;
    uint64_t workspace;
    if (!Multiply(t->core_workspace_bytes, t->vector_cores, workspace) ||
        workspace > static_cast<uint64_t>(INT64_MAX) - platform.GetLibApiWorkSpaceSize())
        return ge::GRAPH_FAILED;
    context->SetBlockDim(cubes);
    context->GetWorkspaceSizes(1)[0] = workspace + platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}
#endif
