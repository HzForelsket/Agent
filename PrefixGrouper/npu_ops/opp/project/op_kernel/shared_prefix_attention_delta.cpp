#include "kernel_operator.h"
#include "shared_prefix_attention_tiling.h"
using namespace AscendC;
namespace {
class Delta {
public:
    __aicore__ inline void Init(GM_ADDR out, GM_ADDR grad, GM_ADDR delta, const SharedPrefixVectorTilingData& data)
    {
        t = data;
        o.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));
        g.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(grad));
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(delta));
        pipe.InitBuffer(inQueue, 2, 2 * t.tile * sizeof(bfloat16_t));
        pipe.InitBuffer(fpBuf, 3 * t.tile * sizeof(float));
        pipe.InitBuffer(resultBuf, 16 * sizeof(float));
    }
    __aicore__ inline uint32_t Count(uint64_t d)
    {
        return t.head_dim - d < t.tile ? t.head_dim - d : t.tile;
    }
    __aicore__ inline void Prefetch(uint64_t row, uint64_t d)
    {
        auto input = inQueue.AllocTensor<bfloat16_t>();
        const uint32_t count = Count(d);
        DataCopyExtParams cp{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, static_cast<bfloat16_t>(0.0f)};
        DataCopyPad(input, o[row * t.head_dim + d], cp, pad);
        DataCopyPad(input[t.tile], g[row * t.head_dim + d], cp, pad);
        inQueue.EnQue(input);
    }
    __aicore__ inline void Process()
    {
        auto fp = fpBuf.Get<float>();
        auto other = fp[t.tile];
        auto work = fp[2 * t.tile];
        auto result = resultBuf.Get<float>();
        for (uint64_t row = GetBlockIdx(); row < t.rows; row += t.cores) {
            float sum = 0.0f;
            Prefetch(row, 0);
            for (uint64_t d = 0; d < t.head_dim; d += t.tile) {
                if (d + t.tile < t.head_dim) Prefetch(row, d + t.tile);
                auto input = inQueue.DeQue<bfloat16_t>();
                const uint32_t count = Count(d);
                Cast(fp, input, RoundMode::CAST_NONE, count);
                Cast(other, input[t.tile], RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                Mul(fp, fp, other, count);
                PipeBarrier<PIPE_V>();
                ReduceSum(work, fp, work, count);
                auto event = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(event); WaitFlag<HardEvent::V_S>(event);
                sum += work.GetValue(0);
                inQueue.FreeTensor(input);
            }
            Duplicate(result, 0.0f, 16);
            auto event = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event); WaitFlag<HardEvent::V_S>(event);
            result.SetValue(0, sum);
            event = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(event); WaitFlag<HardEvent::S_MTE3>(event);
            DataCopy(dst[row * 16], result, 16);
            event = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(event); WaitFlag<HardEvent::MTE3_V>(event);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    TBuf<QuePosition::VECCALC> fpBuf, resultBuf;
    GlobalTensor<bfloat16_t> o, g;
    GlobalTensor<float> dst;
    SharedPrefixVectorTilingData t;
};
}
extern "C" __global__ __aicore__ void shared_prefix_attention_delta(
    GM_ADDR out, GM_ADDR grad, GM_ADDR delta, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SharedPrefixVectorTilingData);
    GET_TILING_DATA(t, tiling);
    Delta kernel;
    kernel.Init(out, grad, delta, t);
    kernel.Process();
}
