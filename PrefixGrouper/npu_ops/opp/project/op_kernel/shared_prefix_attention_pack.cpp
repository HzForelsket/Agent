#include "kernel_operator.h"
#include "shared_prefix_attention_tiling.h"
using namespace AscendC;
namespace {
class Pack {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const SharedPrefixVectorTilingData& data)
    {
        t = data;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
        pipe.InitBuffer(inQueue, 2, (16 * t.tile + 32) * sizeof(float));
        pipe.InitBuffer(outQueue, 2, t.tile * sizeof(bfloat16_t));
        pipe.InitBuffer(indexBuf, t.tile * sizeof(uint32_t));
        pipe.InitBuffer(valueBuf, t.tile * sizeof(float));
    }
    __aicore__ inline uint64_t SourceIndex(uint64_t logical)
    {
        return logical / t.head_dim * t.padded_dim + logical % t.head_dim;
    }
    __aicore__ inline uint32_t Count(uint64_t task)
    {
        const uint64_t remaining = t.rows * t.head_dim - task * t.tile;
        return remaining < t.tile ? remaining : t.tile;
    }
    __aicore__ inline void Prefetch(uint64_t task)
    {
        const uint64_t first = SourceIndex(task * t.tile) / 16 * 16;
        const uint64_t last = (SourceIndex(task * t.tile + Count(task) - 1) / 16 + 1) * 16;
        auto local = inQueue.AllocTensor<float>();
        DataCopy(local, src[first], last - first);
        inQueue.EnQue(local);
    }
    __aicore__ inline void Process()
    {
        const uint64_t start = GetBlockIdx();
        if (start >= t.tasks) return;
        Prefetch(start);
        for (uint64_t task = start; task < t.tasks; task += t.cores) {
            if (task + t.cores < t.tasks) Prefetch(task + t.cores);
            auto input = inQueue.DeQue<float>();
            auto indices = indexBuf.Get<uint32_t>();
            auto values = valueBuf.Get<float>();
            const uint32_t count = Count(task);
            const uint64_t first = SourceIndex(task * t.tile) / 16 * 16;
            // Offsets are relative to a bounded UB span, even for arbitrarily large D.
            for (uint32_t i = 0; i < count; ++i)
                indices.SetValue(i, (SourceIndex(task * t.tile + i) - first) * sizeof(float));
            auto event = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(event); WaitFlag<HardEvent::S_V>(event);
            Gather(values, input, indices, 0, count);
            PipeBarrier<PIPE_V>();
            auto out = outQueue.AllocTensor<bfloat16_t>();
            Cast(out, values, RoundMode::CAST_RINT, count);
            inQueue.FreeTensor(input);
            outQueue.EnQue(out);
            out = outQueue.DeQue<bfloat16_t>();
            DataCopyExtParams cp{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(dst[task * t.tile], out, cp);
            outQueue.FreeTensor(out);
            event = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(event); WaitFlag<HardEvent::V_S>(event);
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<QuePosition::VECCALC> indexBuf, valueBuf;
    GlobalTensor<float> src;
    GlobalTensor<bfloat16_t> dst;
    SharedPrefixVectorTilingData t;
};
}
extern "C" __global__ __aicore__ void shared_prefix_attention_pack(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SharedPrefixVectorTilingData);
    GET_TILING_DATA(t, tiling);
    Pack kernel;
    kernel.Init(input, output, t);
    kernel.Process();
}
