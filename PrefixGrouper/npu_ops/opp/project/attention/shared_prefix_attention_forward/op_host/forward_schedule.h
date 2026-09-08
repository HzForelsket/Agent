#ifndef PREFIX_GROUPER_FORWARD_SCHEDULE_H
#define PREFIX_GROUPER_FORWARD_SCHEDULE_H
#include "../op_kernel/shared_prefix_attention_forward_tiling_data.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace shared_prefix_host {
// Host-only scheduling. A range walks sequence-local Q blocks, then heads;
// the kernel never scans tasks assigned to another worker.
inline bool BuildForwardSchedule(const int64_t* prefixes, size_t prefixCount,
    const int64_t* suffixes, size_t suffixCount, const int64_t* groups, size_t groupCount,
    uint32_t maxCubes, SharedPrefixAttentionForwardTilingData& t)
{
    if (!prefixCount || prefixCount != groupCount || !maxCubes ||
        maxCubes > kForwardMaxVectorCores / 2 || !t.q_heads || !t.q_tile || !t.kv_tile) return false;
    struct Block { uint64_t start; long double cost; };
    std::vector<Block> blocks;
    uint64_t token = 0;
    size_t suffix = 0;
    auto append = [&](uint64_t length, uint64_t prefixLength) {
        if (!length || token > t.total_tokens || length > t.total_tokens - token) return false;
        for (uint64_t offset = 0; offset < length; offset += t.q_tile) {
            const uint64_t rows = std::min<uint64_t>(t.q_tile, length - offset);
            const uint64_t causal = offset + rows;
            const uint64_t calls = (prefixLength + t.kv_tile - 1) / t.kv_tile +
                                   (causal + t.kv_tile - 1) / t.kv_tile;
            const uint64_t alignedKv = (prefixLength + 15) / 16 * 16 + (causal + 15) / 16 * 16;
            // Relative work units: a service cost per KV block plus padded matrix
            // area. D is common to every task. This is not a hardware latency model.
            const long double cost = static_cast<long double>(calls) * t.q_tile * t.kv_tile +
                                     static_cast<long double>((rows + 15) / 16 * 16) * alignedKv;
            blocks.push_back({token + offset, cost});
        }
        token += length;
        return true;
    };
    for (size_t g = 0; g < prefixCount; ++g) {
        if (prefixes[g] <= 0 || groups[g] <= 0 ||
            static_cast<uint64_t>(groups[g]) > suffixCount - suffix || !append(prefixes[g], 0)) return false;
        for (int64_t j = 0; j < groups[g]; ++j) {
            if (suffixes[suffix] <= 0 || !append(suffixes[suffix], prefixes[g])) return false;
            ++suffix;
        }
    }
    if (suffix != suffixCount || token != t.total_tokens || blocks.empty() ||
        blocks.size() > static_cast<uint64_t>(INT64_MAX) / t.q_heads) return false;
    t.task_count = blocks.size() * t.q_heads;
    const uint32_t cubes = std::min<uint64_t>((t.task_count + 1) / 2, maxCubes);
    t.vector_cores = 2 * cubes;
    const uint32_t active = std::min<uint64_t>(t.task_count, t.vector_cores);
    std::vector<long double> cumulative(1, 0);
    for (const auto& block : blocks) cumulative.push_back(cumulative.back() + block.cost * t.q_heads);
    auto workBefore = [&](uint64_t task) {
        if (task == t.task_count) return cumulative.back();
        return cumulative[task / t.q_heads] + (task % t.q_heads) * blocks[task / t.q_heads].cost;
    };
    struct Chunk { ForwardTaskRange range; long double cost; };
    std::vector<Chunk> chunks;
    uint64_t first = 0;
    for (uint32_t worker = 0; worker < active; ++worker) {
        uint64_t last = t.task_count;
        if (worker + 1 < active) {
            const long double target = cumulative.back() * (worker + 1) / active;
            const size_t b = std::lower_bound(cumulative.begin() + 1, cumulative.end(), target) - cumulative.begin() - 1;
            last = b * t.q_heads + static_cast<uint64_t>(std::round((target - cumulative[b]) / blocks[b].cost));
            last = std::max(first + 1, std::min(last, t.task_count - (active - worker - 1)));
        }
        chunks.push_back({{blocks[first / t.q_heads].start, first % t.q_heads, last - first},
                          workBefore(last) - workBefore(first)});
        first = last;
    }
    while (chunks.size() < t.vector_cores) chunks.push_back({{0, 0, 0}, 0});
    // Two AIV clients share one Cube. Pair a heavier range with a lighter one,
    // instead of placing adjacent heavy suffixes on the same physical core.
    std::stable_sort(chunks.begin(), chunks.end(), [](const Chunk& a, const Chunk& b) { return a.cost > b.cost; });
    for (uint32_t cube = 0; cube < cubes; ++cube) {
        t.ranges[2 * cube] = chunks[cube].range;
        t.ranges[2 * cube + 1] = chunks[t.vector_cores - 1 - cube].range;
    }
    return true;
}
}
#endif
