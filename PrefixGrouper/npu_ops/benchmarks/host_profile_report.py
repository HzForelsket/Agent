"""Read exported profiler tables; all arithmetic here is host wall-time accounting."""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


CUSTOM_CPP_STAGES = (
    "cpp_validate", "allocate", "attention_bridge",
)
CUSTOM_PYTHON_STAGES = ("python_validate", "python_scale", "dispatch")
FUSION_STAGES = ("gather_k", "gather_v", "attention_bridge")


def read_host_metrics(operator_csv, operator, mode, step):
    outer = f"pg_attention/{operator}/{mode}/step_{step:03d}"
    with Path(operator_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Name", "Host Self Duration(us)", "Host Total Duration(us)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing host duration columns in {operator_csv}")
        events = {}
        for row in reader:
            name = row["Name"]
            if not (name.startswith(("pg_host/", "pg_attention/", "prefix_grouper_npu::")) or
                    name == "npu::npu_fusion_attention"):
                continue
            self_us = float(row["Host Self Duration(us)"])
            total_us = float(row["Host Total Duration(us)"])
            if not all(math.isfinite(value) for value in (self_us, total_us)):
                raise ValueError(f"Nonfinite host duration for {name}")
            item = events.setdefault(name, {"count": 0, "self_us": 0.0, "total_us": 0.0})
            item["count"] += 1
            item["self_us"] += self_us
            item["total_us"] += total_us
    custom = operator == "shared_prefix_attention"
    expected = [outer, "pg_attention/device_synchronize"]
    if mode != "backward":
        expected += ([f"pg_host/custom/{name}" for name in CUSTOM_CPP_STAGES + CUSTOM_PYTHON_STAGES]
                     if custom else [f"pg_host/fusion/{name}" for name in FUSION_STAGES])
        expected.append("prefix_grouper_npu::shared_prefix_attention_forward" if custom
                        else "npu::npu_fusion_attention")
    missing = [name for name in expected if name not in events]
    derived = {}
    if outer in events and "pg_attention/device_synchronize" in events:
        derived["before_final_sync_us"] = (
            events[outer]["total_us"] - events["pg_attention/device_synchronize"]["total_us"]
        )
    cpp = "prefix_grouper_npu::shared_prefix_attention_forward"
    if custom and mode != "backward" and cpp in events and not missing:
        derived["cpp_outside_stages_us"] = events[cpp]["total_us"] - sum(
            events[f"pg_host/custom/{name}"]["total_us"] for name in CUSTOM_CPP_STAGES
        )
        derived["dispatch_outside_cpp_us"] = (
            events["pg_host/custom/dispatch"]["total_us"] - events[cpp]["total_us"]
        )
    return {"events": events, "missing_events": missing, "derived": derived}


def host_profile_markdown(result):
    info = result["profiling"]
    lines = [
        "# 前向 Host 分段统计与回传指标", "",
        f"状态：`{info['status']}`；探针版本：`{info['host_probe_version']}`。",
        "单位均为 μs；Self/Total 是主机事件墙钟时间，不是 CPU 有效计算时间。",
        "桥接探针覆盖调用线程；异步工作线程中的准备、执行需结合 CANN 时间线查看。",
        "父子 Total 不可相加；添加子探针后父事件 Self 会重新归属，不能与旧版 Self 直接判定加速。", "",
        "## 1. 运行与二进制身份", "",
        f"- Python：`{result['python']}`",
        f"- 包路径：`{result['package_path']}`",
        f"- CANN：`{result.get('ascend_home_path')}`；设备：`{result.get('device', '未获取')}`",
        f"- PyTorch：`{result['torch']}`；torch-npu：`{result['torch_npu']}`",
        f"- 任务队列配置：`{result.get('runtime_environment', {})}`（未设置不等于禁用，以运行时为准）",
        f"- 完整命令和输入见 benchmark.json；输入：`{result['input']}`", "",
        "| 已安装文件 | SHA256 |", "|---|---|",
    ]
    for name, digest in result.get("artifact_sha256", {}).items():
        lines.append(f"| {name} | `{digest}` |")
    lines += ["", f"采集选项：record_shapes={info.get('record_shapes')}，"
              f"profile_memory={info.get('profile_memory')}，with_stack={info.get('with_stack')}。",
              "不同采集选项的 Host 时间不可直接比较；提速仍以关闭 profiler 的计时为准。"]
    lines += ["", "## 2. 未开启 profiler 的前向速度", "",
              "| 路径 | Mean (μs) | Median (μs) | 次数 |", "|---|---:|---:|---:|"]
    for operator, record in result.get("timings", {}).get("forward", {}).items():
        if "mean_ms" in record:
            lines.append(f"| {operator} | {record['mean_ms'] * 1000:.3f} | "
                         f"{record['median_ms'] * 1000:.3f} | {len(record['samples_ms'])} |")
    lines += ["", "## 3. 每次前向采集的探针", "",
              "Total 包含子事件；每行 Count 是本次 capture 内调用次数。缺失不填 0。"]
    across = {}
    for capture in info["captures"]:
        if capture["mode"] != "forward":
            continue
        operator, step = capture["operator"], capture["step"]
        lines += ["", f"### {operator} / step_{step:03d}", "",
                  f"采集状态：`{capture['status']}`。"]
        metrics = capture.get("host_metrics")
        if not metrics:
            lines += ["尚无可用分段统计。"]
            continue
        if metrics["missing_events"]:
            lines += ["缺失事件：" + ", ".join(f"`{name}`" for name in metrics["missing_events"]) +
                      "。确认已安装新 wheel；本次不能完成分段归因。"]
        lines += ["", "| 事件 | Count | Self | Total |", "|---|---:|---:|---:|"]
        for name, values in metrics["events"].items():
            lines.append(f"| `{name}` | {values['count']} | {values['self_us']:.3f} | {values['total_us']:.3f} |")
            key = name.rsplit("/step_", 1)[0]
            across.setdefault((operator, key), []).append(values["total_us"])
        lines.append("")
        for name, value in metrics["derived"].items():
            lines += [f"- `{name}`：{value:.3f} μs。"]
        lines += ["", "CANN API 汇总：同层按 Time 降序查看；不能与上表相加。"]
        api_files = capture.get("api_statistics", [])
        if not api_files:
            lines += ["未找到 api_statistic.csv；内部 API/缓存命中状态未知。"]
        for api in api_files:
            lines += [f"- [API 原表](<{api['path']}>)"]
            fields = api["columns"]
            lines += ["", "| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
            for row in api["rows"]:
                lines.append("| " + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field in fields) + " |")
        for name, path in capture["artifacts"].items():
            lines += [f"- [{name}](<{path}>)"]
    lines += ["", "## 4. 同路径多次 capture 的 Total 分布", "",
              "每次 capture 单独启动 profiler；此处分布含 profiler 启动及记录的影响。",
              "| 路径 | 事件 | captures | Median | Min | Max |", "|---|---|---:|---:|---:|---:|"]
    for (operator, name), values in across.items():
        lines.append(f"| {operator} | `{name}` | {len(values)} | {statistics.median(values):.3f} | "
                     f"{min(values):.3f} | {max(values):.3f} |")
    lines += ["", "## 5. 需要回传的关键数值", "",
              "- 隔离机器无需上传文件、路径或环境细节；按第 2/3 节手工告知关键数值即可。",
              "- 优先提供两条路径的正常计时 median、外层 Total、最后同步，以及 custom validate/scale/dispatch/allocate/attention_bridge。",
              "- dispatch 包含缓存入口查询及已注册算子调用；不再存在手工 autograd_apply 或每次 load_extension 探针。",
              "- 前向 BF16 转换和紧凑 LSE 写回已合入 Attention，pack_bridge/lse_compact 不再是独立事件。",
              "- 若只回传数值：第 3/4 节表格，以及 Attention/gather 的设备任务名称、次数、耗时和间隙。",
              "- 从时间线补充：Attention/fusion 的 GetWorkspaceSize、Tiling、Launch 的名称、线程、次数和耗时；",
              "  未导出就填“未观测”，缓存命中未经直接证据确认就填“未知”。",
              "- 不把 API 汇总、父子事件或不同线程的耗时直接相加；Launch 接近不能证明整个桥接成本接近。", ""]
    return "\n".join(lines)
