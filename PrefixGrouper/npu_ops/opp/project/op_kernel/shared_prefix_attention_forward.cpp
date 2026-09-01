#include "kernel_operator.h"
#include "shared_prefix_attention_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kBf16Bytes = kHeadDim * sizeof(bfloat16_t);
constexpr uint32_t kFp32Bytes = kHeadDim * sizeof(float);

class SharedPrefixAttentionForwardKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v,
        GM_ADDR prefix_start, GM_ADDR prefix_end, GM_ADDR sequence_start,
        GM_ADDR out, GM_ADDR lse,
        const SharedPrefixAttentionTilingData& tiling)
    {
        totalTokens_ = tiling.total_tokens;
        qHeads_ = tiling.q_heads;
        kvHeads_ = tiling.kv_heads;
        scale_ = tiling.scale;
        groupRatio_ = qHeads_ / kvHeads_;

        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
        prefixStartGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_start));
        prefixEndGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_end));
        sequenceStartGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequence_start));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));
        lseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));

        pipe_.InitBuffer(qBfBuf_, kBf16Bytes);
        pipe_.InitBuffer(kBfBuf_, kBf16Bytes);
        pipe_.InitBuffer(vBfBuf_, kBf16Bytes);
        pipe_.InitBuffer(outBfBuf_, kBf16Bytes);
        pipe_.InitBuffer(qFpBuf_, kFp32Bytes);
        pipe_.InitBuffer(kFpBuf_, kFp32Bytes);
        pipe_.InitBuffer(vFpBuf_, kFp32Bytes);
        pipe_.InitBuffer(accFpBuf_, kFp32Bytes);
        pipe_.InitBuffer(tmpFpBuf_, kFp32Bytes);
        pipe_.InitBuffer(workFpBuf_, kFp32Bytes);
        pipe_.InitBuffer(scalarInBuf_, 32);
        pipe_.InitBuffer(scalarOutBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        const uint32_t taskCount = totalTokens_ * qHeads_;
        for (uint32_t task = GetBlockIdx(); task < taskCount; task += GetBlockNum()) {
            const uint32_t queryToken = task / qHeads_;
            const uint32_t queryHead = task % qHeads_;
            ComputeOne(queryToken, queryHead);
        }
    }

private:
    __aicore__ inline void LoadBf16Row(
        GlobalTensor<bfloat16_t>& gm, uint64_t offset,
        LocalTensor<bfloat16_t>& bf, LocalTensor<float>& fp)
    {
        DataCopy(bf, gm[offset], kHeadDim);
        PipeBarrier<PIPE_ALL>();
        Cast(fp, bf, RoundMode::CAST_NONE, kHeadDim);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float Dot(LocalTensor<float>& a, LocalTensor<float>& b)
    {
        LocalTensor<float> tmp = tmpFpBuf_.Get<float>();
        LocalTensor<float> work = workFpBuf_.Get<float>();
        Mul(tmp, a, b, kHeadDim);
        ReduceSum(work, tmp, work, kHeadDim);
        event_t event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event);
        WaitFlag<HardEvent::V_S>(event);
        return work.GetValue(0);
    }

    __aicore__ inline float ExpScalar(float value)
    {
        LocalTensor<float> src = scalarInBuf_.Get<float>();
        LocalTensor<float> dst = scalarOutBuf_.Get<float>();
        Duplicate(src, value, 8);
        Exp(dst, src, 8);
        event_t event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event);
        WaitFlag<HardEvent::V_S>(event);
        return dst.GetValue(0);
    }

    __aicore__ inline float LogScalar(float value)
    {
        LocalTensor<float> src = scalarInBuf_.Get<float>();
        LocalTensor<float> dst = scalarOutBuf_.Get<float>();
        Duplicate(src, value, 8);
        Ln(dst, src, 8);
        event_t event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event);
        WaitFlag<HardEvent::V_S>(event);
        return dst.GetValue(0);
    }

    __aicore__ inline void ConsumeKey(
        uint32_t keyToken, uint32_t kvHead,
        LocalTensor<float>& qFp, LocalTensor<float>& accFp,
        float& runningMax, float& runningSum)
    {
        LocalTensor<bfloat16_t> kBf = kBfBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> vBf = vBfBuf_.Get<bfloat16_t>();
        LocalTensor<float> kFp = kFpBuf_.Get<float>();
        LocalTensor<float> vFp = vFpBuf_.Get<float>();
        LocalTensor<float> tmp = tmpFpBuf_.Get<float>();
        const uint64_t kvOffset = (static_cast<uint64_t>(keyToken) * kvHeads_ + kvHead) * kHeadDim;

        LoadBf16Row(kGm_, kvOffset, kBf, kFp);
        const float score = Dot(qFp, kFp) * scale_;
        const float nextMax = score > runningMax ? score : runningMax;
        const float oldWeight = runningSum == 0.0f ? 0.0f : ExpScalar(runningMax - nextMax);
        const float newWeight = ExpScalar(score - nextMax);

        Muls(accFp, accFp, oldWeight, kHeadDim);
        LoadBf16Row(vGm_, kvOffset, vBf, vFp);
        Muls(tmp, vFp, newWeight, kHeadDim);
        Add(accFp, accFp, tmp, kHeadDim);
        PipeBarrier<PIPE_ALL>();

        runningSum = runningSum * oldWeight + newWeight;
        runningMax = nextMax;
    }

    __aicore__ inline void ComputeOne(uint32_t queryToken, uint32_t queryHead)
    {
        LocalTensor<bfloat16_t> qBf = qBfBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> outBf = outBfBuf_.Get<bfloat16_t>();
        LocalTensor<float> qFp = qFpBuf_.Get<float>();
        LocalTensor<float> accFp = accFpBuf_.Get<float>();

        const uint64_t qOffset = (static_cast<uint64_t>(queryToken) * qHeads_ + queryHead) * kHeadDim;
        LoadBf16Row(qGm_, qOffset, qBf, qFp);
        Duplicate(accFp, 0.0f, kHeadDim);
        PipeBarrier<PIPE_ALL>();

        const uint32_t kvHead = queryHead / groupRatio_;
        const int32_t prefixStart = prefixStartGm_.GetValue(queryToken);
        const int32_t prefixEnd = prefixEndGm_.GetValue(queryToken);
        const int32_t sequenceStart = sequenceStartGm_.GetValue(queryToken);
        float runningMax = -3.402823466e+38F;
        float runningSum = 0.0f;

        if (sequenceStart == prefixStart) {
            for (int32_t key = prefixStart; key <= static_cast<int32_t>(queryToken); ++key) {
                ConsumeKey(static_cast<uint32_t>(key), kvHead, qFp, accFp, runningMax, runningSum);
            }
        } else {
            for (int32_t key = prefixStart; key < prefixEnd; ++key) {
                ConsumeKey(static_cast<uint32_t>(key), kvHead, qFp, accFp, runningMax, runningSum);
            }
            for (int32_t key = sequenceStart; key <= static_cast<int32_t>(queryToken); ++key) {
                ConsumeKey(static_cast<uint32_t>(key), kvHead, qFp, accFp, runningMax, runningSum);
            }
        }

        Muls(accFp, accFp, 1.0f / runningSum, kHeadDim);
        Cast(outBf, accFp, RoundMode::CAST_RINT, kHeadDim);
        PipeBarrier<PIPE_ALL>();
        DataCopy(outGm_[qOffset], outBf, kHeadDim);
        lseGm_.SetValue(static_cast<uint64_t>(queryToken) * qHeads_ + queryHead,
                        runningMax + LogScalar(runningSum));
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> qBfBuf_, kBfBuf_, vBfBuf_, outBfBuf_;
    TBuf<QuePosition::VECCALC> qFpBuf_, kFpBuf_, vFpBuf_, accFpBuf_, tmpFpBuf_, workFpBuf_;
    TBuf<QuePosition::VECCALC> scalarInBuf_, scalarOutBuf_;
    GlobalTensor<bfloat16_t> qGm_, kGm_, vGm_, outGm_;
    GlobalTensor<int32_t> prefixStartGm_, prefixEndGm_, sequenceStartGm_;
    GlobalTensor<float> lseGm_;
    uint32_t totalTokens_ = 0;
    uint32_t qHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t groupRatio_ = 0;
    float scale_ = 0.0f;
};
}

extern "C" __global__ __aicore__ void shared_prefix_attention_forward(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR prefix_start, GM_ADDR prefix_end, GM_ADDR sequence_start,
    GM_ADDR out, GM_ADDR lse, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SharedPrefixAttentionTilingData);
    GET_TILING_DATA(tilingData, tiling);
    SharedPrefixAttentionForwardKernel kernel;
    kernel.Init(q, k, v, prefix_start, prefix_end, sequence_start, out, lse, tilingData);
    kernel.Process();
}
