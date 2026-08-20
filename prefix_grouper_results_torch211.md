# PrefixGrouper 性能对比

- GPU：NVIDIA A100-SXM4-80GB
- 软件栈：torch 2.11.0 / vLLM 0.22.1+cu129 / VERL 0.9.0
- 精度：bfloat16，权重：random

| 模型 | 模式 | Prompt | 共享组 | 添加前 ms | 添加后 ms | 加速比 | 添加前 tok/s | 添加后 tok/s | 峰值显存节省 | 正确性 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | forward | 256 | 4 | 25.481 | 42.734 | 0.60x | 20093 | 11981 | -0.0% | 通过 |
| Qwen2.5-0.5B-Instruct | forward-backward | 256 | 4 | 75.615 | 104.492 | 0.72x | 6771 | 4900 | 42.0% | 通过 |
| Qwen2.5-0.5B-Instruct | forward | 512 | 8 | 41.957 | 43.501 | 0.96x | 12203 | 11770 | 25.8% | 通过 |
| Qwen2.5-0.5B-Instruct | forward-backward | 512 | 8 | 130.967 | 106.856 | 1.23x | 3909 | 4791 | 63.0% | 通过 |
| SmolLM2-135M-Instruct | forward | 256 | 4 | 17.807 | 48.581 | 0.37x | 28752 | 10539 | 0.3% | 通过 |
| SmolLM2-135M-Instruct | forward-backward | 256 | 4 | 63.650 | 124.319 | 0.51x | 8044 | 4118 | 41.5% | 通过 |
| SmolLM2-135M-Instruct | forward | 512 | 8 | 19.997 | 49.198 | 0.41x | 25604 | 10407 | 27.4% | 通过 |
| SmolLM2-135M-Instruct | forward-backward | 512 | 8 | 65.354 | 119.455 | 0.55x | 7834 | 4286 | 62.1% | 通过 |
| TinyLlama-1.1B-Chat-v1.0 | forward | 256 | 4 | 44.660 | 36.945 | 1.21x | 11464 | 13858 | 0.3% | 通过 |
| TinyLlama-1.1B-Chat-v1.0 | forward-backward | 256 | 4 | 133.509 | 96.117 | 1.39x | 3835 | 5327 | 33.9% | 通过 |
| TinyLlama-1.1B-Chat-v1.0 | forward | 512 | 8 | 75.621 | 37.469 | 2.02x | 6771 | 13665 | 5.8% | 通过 |
| TinyLlama-1.1B-Chat-v1.0 | forward-backward | 512 | 8 | 232.274 | 97.041 | 2.39x | 2204 | 5276 | 54.0% | 通过 |
