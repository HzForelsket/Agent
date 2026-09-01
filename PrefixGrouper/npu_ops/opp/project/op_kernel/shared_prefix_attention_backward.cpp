#include "kernel_operator.h"
#include "shared_prefix_attention_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kBf16Bytes = kHeadDim * sizeof(bfloat16_t);
constexpr uint32_t kFp32Bytes = kHeadDim * sizeof(float);

class SharedPrefixAttentionBackwardKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR grad_out, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse,
        GM_ADDR prefix_start, GM_ADDR prefix_end, GM_ADDR sequence_start,
        GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
        const SharedPrefixAttentionTilingData& tiling)
    {
        totalTokens_ = tiling.total_tokens;
        qHeads_ = tiling.q_heads;
        kvHeads_ = tiling.kv_heads;
        scale_ = tiling.scale;
        groupRatio_ = qHeads_ / kvHeads_;

        gradOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(grad_out));
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));
        lseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
        prefixStartGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_start));
        prefixEndGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(prefix_end));
        sequenceStartGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequence_start));
        dqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dq));
        dkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dk));
        dvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(dv));

        pipe_.InitBuffer(loadBfBuf_, kBf16Bytes);
        pipe_.InitBuffer(storeBfBuf_, kBf16Bytes);
        pipe_.InitBuffer(fp0Buf_, kFp32Bytes);
        pipe_.InitBuffer(fp1Buf_, kFp32Bytes);
        pipe_.InitBuffer(fp2Buf_, kFp32Bytes);
        pipe_.InitBuffer(fp3Buf_, kFp32Bytes);
        pipe_.InitBuffer(fp4Buf_, kFp32Bytes);
        pipe_.InitBuffer(acc0Buf_, kFp32Bytes);
        pipe_.InitBuffer(acc1Buf_, kFp32Bytes);
        pipe_.InitBuffer(tmpBuf_, kFp32Bytes);
        pipe_.InitBuffer(workBuf_, kFp32Bytes);
        pipe_.InitBuffer(scalarInBuf_, 32);
        pipe_.InitBuffer(scalarOutBuf_, 32);
    }

    __aicore__ inline void Process()
    {
        const uint32_t dqTasks = totalTokens_ * qHeads_;
        const uint32_t dkvTasks = totalTokens_ * kvHeads_;
        const uint32_t allTasks = dqTasks + dkvTasks;
        for (uint32_t task = GetBlockIdx(); task < allTasks; task += GetBlockNum()) {
            if (task < dqTasks) {
                ComputeDq(task / qHeads_, task % qHeads_);
            } else {
                const uint32_t dkvTask = task - dqTasks;
                ComputeDkv(dkvTask / kvHeads_, dkvTask % kvHeads_);
            }
        }
    }

private:
    __aicore__ inline void Load(
        GlobalTensor<bfloat16_t>& gm, uint64_t offset, LocalTensor<float>& fp)
    {
        LocalTensor<bfloat16_t> bf = loadBfBuf_.Get<bfloat16_t>();
        DataCopy(bf, gm[offset], kHeadDim);
        PipeBarrier<PIPE_ALL>();
        Cast(fp, bf, RoundMode::CAST_NONE, kHeadDim);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Store(
        GlobalTensor<bfloat16_t>& gm, uint64_t offset, LocalTensor<float>& fp)
    {
        LocalTensor<bfloat16_t> bf = storeBfBuf_.Get<bfloat16_t>();
        Cast(bf, fp, RoundMode::CAST_RINT, kHeadDim);
        PipeBarrier<PIPE_ALL>();
        DataCopy(gm[offset], bf, kHeadDim);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float Dot(LocalTensor<float>& a, LocalTensor<float>& b)
    {
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> work = workBuf_.Get<float>();
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

    __aicore__ inline bool KeyAllowed(uint32_t queryToken, uint32_t keyToken)
    {
        const int32_t prefixStart = prefixStartGm_.GetValue(queryToken);
        const int32_t prefixEnd = prefixEndGm_.GetValue(queryToken);
        const int32_t sequenceStart = sequenceStartGm_.GetValue(queryToken);
        if (sequenceStart == prefixStart) {
            return keyToken >= static_cast<uint32_t>(prefixStart) && keyToken <= queryToken;
        }
        const bool inPrefix = keyToken >= static_cast<uint32_t>(prefixStart) &&
                              keyToken < static_cast<uint32_t>(prefixEnd);
        const bool inSuffix = keyToken >= static_cast<uint32_t>(sequenceStart) && keyToken <= queryToken;
        return inPrefix || inSuffix;
    }

    __aicore__ inline void AccumulateDqKey(
        uint32_t queryToken, uint32_t queryHead, uint32_t keyToken,
        LocalTensor<float>& qFp, LocalTensor<float>& gradFp,
        LocalTensor<float>& accFp, float delta, float lse)
    {
        LocalTensor<float> kFp = fp2Buf_.Get<float>();
        LocalTensor<float> vFp = fp3Buf_.Get<float>();
        const uint32_t kvHead = queryHead / groupRatio_;
        const uint64_t kvOffset = (static_cast<uint64_t>(keyToken) * kvHeads_ + kvHead) * kHeadDim;
        Load(kGm_, kvOffset, kFp);
        const float probability = ExpScalar(Dot(qFp, kFp) * scale_ - lse);
        Load(vGm_, kvOffset, vFp);
        const float ds = probability * (Dot(gradFp, vFp) - delta) * scale_;
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        Muls(tmp, kFp, ds, kHeadDim);
        Add(accFp, accFp, tmp, kHeadDim);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeDq(uint32_t queryToken, uint32_t queryHead)
    {
        LocalTensor<float> qFp = fp0Buf_.Get<float>();
        LocalTensor<float> gradFp = fp1Buf_.Get<float>();
        LocalTensor<float> outFp = fp2Buf_.Get<float>();
        LocalTensor<float> accFp = acc0Buf_.Get<float>();
        const uint64_t qOffset = (static_cast<uint64_t>(queryToken) * qHeads_ + queryHead) * kHeadDim;
        Load(qGm_, qOffset, qFp);
        Load(gradOutGm_, qOffset, gradFp);
        Load(outGm_, qOffset, outFp);
        const float delta = Dot(gradFp, outFp);
        const float lse = lseGm_.GetValue(static_cast<uint64_t>(queryToken) * qHeads_ + queryHead);
        Duplicate(accFp, 0.0f, kHeadDim);
        PipeBarrier<PIPE_ALL>();

        const int32_t prefixStart = prefixStartGm_.GetValue(queryToken);
        const int32_t prefixEnd = prefixEndGm_.GetValue(queryToken);
        const int32_t sequenceStart = sequenceStartGm_.GetValue(queryToken);
        if (sequenceStart == prefixStart) {
            for (int32_t key = prefixStart; key <= static_cast<int32_t>(queryToken); ++key) {
                AccumulateDqKey(queryToken, queryHead, static_cast<uint32_t>(key), qFp, gradFp, accFp, delta, lse);
            }
        } else {
            for (int32_t key = prefixStart; key < prefixEnd; ++key) {
                AccumulateDqKey(queryToken, queryHead, static_cast<uint32_t>(key), qFp, gradFp, accFp, delta, lse);
            }
            for (int32_t key = sequenceStart; key <= static_cast<int32_t>(queryToken); ++key) {
                AccumulateDqKey(queryToken, queryHead, static_cast<uint32_t>(key), qFp, gradFp, accFp, delta, lse);
            }
        }
        Store(dqGm_, qOffset, accFp);
    }

    __aicore__ inline void ComputeDkv(uint32_t keyToken, uint32_t kvHead)
    {
        LocalTensor<float> kFp = fp0Buf_.Get<float>();
        LocalTensor<float> vFp = fp1Buf_.Get<float>();
        LocalTensor<float> qFp = fp2Buf_.Get<float>();
        LocalTensor<float> gradFp = fp3Buf_.Get<float>();
        LocalTensor<float> outFp = fp4Buf_.Get<float>();
        LocalTensor<float> dkFp = acc0Buf_.Get<float>();
        LocalTensor<float> dvFp = acc1Buf_.Get<float>();
        const uint64_t kvOffset = (static_cast<uint64_t>(keyToken) * kvHeads_ + kvHead) * kHeadDim;
        Load(kGm_, kvOffset, kFp);
        Load(vGm_, kvOffset, vFp);
        Duplicate(dkFp, 0.0f, kHeadDim);
        Duplicate(dvFp, 0.0f, kHeadDim);
        PipeBarrier<PIPE_ALL>();

        const uint32_t firstQHead = kvHead * groupRatio_;
        const uint32_t lastQHead = firstQHead + groupRatio_;
        for (uint32_t queryToken = 0; queryToken < totalTokens_; ++queryToken) {
            if (!KeyAllowed(queryToken, keyToken)) {
                continue;
            }
            for (uint32_t queryHead = firstQHead; queryHead < lastQHead; ++queryHead) {
                const uint64_t qOffset = (static_cast<uint64_t>(queryToken) * qHeads_ + queryHead) * kHeadDim;
                Load(qGm_, qOffset, qFp);
                Load(gradOutGm_, qOffset, gradFp);
                Load(outGm_, qOffset, outFp);
                const float lse = lseGm_.GetValue(static_cast<uint64_t>(queryToken) * qHeads_ + queryHead);
                const float probability = ExpScalar(Dot(qFp, kFp) * scale_ - lse);
                const float delta = Dot(gradFp, outFp);
                const float ds = probability * (Dot(gradFp, vFp) - delta) * scale_;
                LocalTensor<float> tmp = tmpBuf_.Get<float>();
                Muls(tmp, qFp, ds, kHeadDim);
                Add(dkFp, dkFp, tmp, kHeadDim);
                Muls(tmp, gradFp, probability, kHeadDim);
                Add(dvFp, dvFp, tmp, kHeadDim);
                PipeBarrier<PIPE_ALL>();
            }
        }
        Store(dkGm_, kvOffset, dkFp);
        Store(dvGm_, kvOffset, dvFp);
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> loadBfBuf_, storeBfBuf_;
    TBuf<QuePosition::VECCALC> fp0Buf_, fp1Buf_, fp2Buf_, fp3Buf_, fp4Buf_;
    TBuf<QuePosition::VECCALC> acc0Buf_, acc1Buf_, tmpBuf_, workBuf_;
    TBuf<QuePosition::VECCALC> scalarInBuf_, scalarOutBuf_;
    GlobalTensor<bfloat16_t> gradOutGm_, qGm_, kGm_, vGm_, outGm_;
    GlobalTensor<bfloat16_t> dqGm_, dkGm_, dvGm_;
    GlobalTensor<int32_t> prefixStartGm_, prefixEndGm_, sequenceStartGm_;
    GlobalTensor<float> lseGm_;
    uint32_t totalTokens_ = 0;
    uint32_t qHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t groupRatio_ = 0;
    float scale_ = 0.0f;
};
}

extern "C" __global__ __aicore__ void shared_prefix_attention_backward(
    GM_ADDR grad_out, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse,
    GM_ADDR prefix_start, GM_ADDR prefix_end, GM_ADDR sequence_start,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SharedPrefixAttentionTilingData);
    GET_TILING_DATA(tilingData, tiling);
    SharedPrefixAttentionBackwardKernel kernel;
    kernel.Init(grad_out, q, k, v, out, lse, prefix_start, prefix_end, sequence_start,
                dq, dk, dv, tilingData);
    kernel.Process();
}
