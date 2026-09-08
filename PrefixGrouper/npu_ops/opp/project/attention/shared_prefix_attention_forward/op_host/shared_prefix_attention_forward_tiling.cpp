#include "../../common/op_host/shape_utils.h"
#include "../op_kernel/shared_prefix_attention_forward_tiling_data.h"
#include "forward_schedule.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/softmax/softmax_tiling.h"
#include <algorithm>
#include <cmath>

namespace optiling {
namespace {
bool Matmul(platform_ascendc::PlatformAscendC& platform, uint32_t m, uint32_t n, uint32_t k,
            bool transpose, uint64_t l1, uint64_t l0c, uint64_t ub,
            AscendC::tiling::TCubeTiling& result)
{
    matmul_tiling::MatmulApiTiling mm(platform);
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BF16);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BF16, transpose);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT);
    mm.SetShape(m, n, k);
    mm.SetOrgShape(m, n, k);
    // Reduce/iterate D inside Matmul, rather than one service request per D tile.
    mm.SetFixSplit(m, std::min(n, 128U));
    mm.EnableBias(false);
    mm.SetBufferSpace(l1 / 2, l0c / 2, ub / 2);
    return mm.GetTiling(result) == 0;
}
}
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    using shared_prefix_host::Multiply;
    const auto* q = context->GetInputShape(0);
    const auto* k = context->GetInputShape(1);
    const auto* v = context->GetInputShape(2);
    const auto* attrs = context->GetAttrs();
    if (!q || !k || !v || !attrs || !context->GetPlatformInfo() ||
        !shared_prefix_host::Shape(q->GetStorageShape(), k->GetStorageShape()) ||
        v->GetStorageShape() != k->GetStorageShape()) return ge::GRAPH_FAILED;
    const auto* scale = attrs->GetAttrPointer<float>(0);
    if (!scale || !std::isfinite(*scale) || *scale <= 0) return ge::GRAPH_FAILED;
    auto* t = context->GetTilingData<SharedPrefixAttentionForwardTilingData>();
    *t = {};
    t->total_tokens = q->GetStorageShape().GetDim(0);
    t->q_heads = q->GetStorageShape().GetDim(1);
    t->kv_heads = k->GetStorageShape().GetDim(1);
    t->head_dim = q->GetStorageShape().GetDim(2);
    t->padded_dim = shared_prefix_host::Align(t->head_dim);
    t->scale = *scale;
    uint64_t qStride, kvStride, size;
    if (!Multiply(t->q_heads, t->head_dim, qStride) || qStride > INT32_MAX ||
        !Multiply(t->kv_heads, t->head_dim, kvStride) || kvStride > INT32_MAX ||
        t->padded_dim > INT32_MAX || !Multiply(t->total_tokens, t->q_heads, size) ||
        !Multiply(size, t->padded_dim * sizeof(float), size)) return ge::GRAPH_FAILED;
    for (uint32_t i = 3; i < 8; ++i) {
        const auto* meta = context->GetInputShape(i);
        if (!meta || meta->GetStorageShape().GetDimNum() != 1 ||
            meta->GetStorageShape().GetDim(0) != static_cast<int64_t>(t->total_tokens)) return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ub = 0, l1 = 0, l0c = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    // Independent query/KV dimensions, bounded by live Vector/API storage.
    uint32_t qm = 64, kn = 128;
    while (qm > 16 && qm / 2 >= t->total_tokens) qm /= 2;
    while (kn > 32 && kn / 2 >= t->total_tokens) kn /= 2;
    for (;;) {
        const ge::Shape shape({qm, kn});
        const uint32_t minTmp = std::max(AscendC::GetSoftMaxFlashV2MinTmpSize(shape, 4, 4, false),
                                     AscendC::GetSoftMaxFlashV2MinTmpSize(shape, 4, 4, true));
        // Score + BF16 P, three max/sum/alpha snapshots, mask and two output chunks.
        const uint64_t buffers = 6ULL * qm * kn + 288ULL * qm + 4ULL * std::max(kn, 64U) + 32 + 512ULL * qm;
        // A minimum-size scratch makes Softmax repeatedly process tiny chunks.
        // Reserve service storage first, then give its remaining UB to the API.
        const uint32_t tmp = buffers + 32 * 1024 < ub ? (ub - buffers - 32 * 1024) / 32 * 32 : 0;
        const uint64_t live = buffers + tmp;
        if (tmp >= minTmp && live + 16 * 1024 < ub &&
            Matmul(platform, qm, kn, t->head_dim, true, l1, l0c, ub - live - 4096, t->score_mm) &&
            Matmul(platform, qm, t->head_dim, kn, false, l1, l0c, ub - live - 4096, t->value_mm) &&
            live + std::max(0, t->score_mm.shareUbSize) + std::max(0, t->value_mm.shareUbSize) + 4096 <= ub) {
            t->softmax_tmp_bytes = tmp;
            break;
        }
        if (qm > 16) qm /= 2;
        else if (kn > 32) kn /= 2;
        else return ge::GRAPH_FAILED;
    }
    t->q_tile = qm;
    t->kv_tile = kn;
    const auto* prefixes = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const auto* suffixes = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    const auto* groups = attrs->GetAttrPointer<gert::ContinuousVector>(3);
    if (!prefixes || !suffixes || !groups || !shared_prefix_host::BuildForwardSchedule(
        static_cast<const int64_t*>(prefixes->GetData()), prefixes->GetSize(),
        static_cast<const int64_t*>(suffixes->GetData()), suffixes->GetSize(),
        static_cast<const int64_t*>(groups->GetData()), groups->GetSize(),
        std::min(platform.GetCoreNumAic(), platform.GetCoreNumAiv() / 2), *t)) return ge::GRAPH_FAILED;
    t->core_workspace_bytes = 3ULL * qm * (kn * 6ULL + t->padded_dim * 4ULL);
    uint64_t workspace;
    if (!Multiply(t->core_workspace_bytes, t->vector_cores, workspace) ||
        workspace > static_cast<uint64_t>(INT64_MAX) - platform.GetLibApiWorkSpaceSize()) return ge::GRAPH_FAILED;
    context->SetBlockDim(t->vector_cores / 2);
    context->GetWorkspaceSizes(1)[0] = workspace + platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(SharedPrefixAttentionForward).Tiling(TilingFunc);
}
