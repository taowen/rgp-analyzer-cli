# Ch13: GGUF Runtime Asset Migration For OmniVoice Vulkan GGML

这一章从零重建，不再延续之前那版 `ch13` 的实验性实现。

目标不是先追求“已经完整跑通整图 Vulkan 推理”，而是先把架构迁移的地基打对：

1. 从官方 `OmniVoice` 导出**运行时真正需要**的资产
2. 把这些资产打包成单个 **GGUF** 文件
3. 用本地 `ggml/gguf` C++ 工具验证：
   - metadata 正确
   - tensor 能被找到
   - 运行时不再依赖“满地 `.bin` 文件”
4. 用 Vulkan backend 跑通**第一张真正从 GGUF 取权重/张量构图的 ggml graph**
5. 为下一步做“GGUF + 更整图 Vulkan ggml 推理”做准备

## 本章当前边界

这章当前是**架构迁移里程碑**，不是最终性能里程碑。

它当前提供：

- `export-runtime-assets.py`
  - 用官方 `OmniVoice` 生成参考音频与 token
  - 导出 backbone / iterative decode / audio decoder 运行时资产
- `ch13_pack_runtime_gguf`
  - 把分散导出资产打包成单个 `GGUF`
- `ch13_inspect_runtime_gguf`
  - 验证 `GGUF` 里的关键 metadata / tensors 是否齐全
- `ch13_runtime_gguf_smoke`
  - 从 `GGUF` 直接加载到 Vulkan backend，并做张量 checksum
- `ch13_runtime_gguf_first_graph`
  - 从 `GGUF` 直接取 runtime tensor，在 Vulkan 上跑第一张真实 ggml graph
- `ch13_runtime_gguf_layer0`
  - 从 `GGUF` 直接取 layer0 权重和 cond 输入，在 Vulkan 上跑第一层 pre-attention 子图，并与 PyTorch 参考值对比
- `ch13_runtime_gguf_backbone`
  - 从 `GGUF` 直接跑 cond path 的多层 backbone，并对比每层输出 / final hidden / logits
  - 现在支持 `gpu_fused_qk` 路径：用 chapter-local Vulkan fused Q/K shader 取代原来的 late-layer `q_proj+k_proj+q_norm+k_norm+rope` 组合
- `ch13_runtime_gguf_iterative_decode`
  - 用 `gpu_fused_qk` backbone 路径直接做 32-step iterative token generation
  - 输出 `generated_tokens_fused_i32.bin`
  - 并和 Python 参考 `generated_tokens_i32.bin` 统计 mismatch
- `extract-official-qwen3-hf.py`
  - 从 OmniVoice 的主权重里抽出标准 `Qwen3ForCausalLM` 子模型目录
- `convert-official-qwen3-gguf.sh`
  - 直接调用官方 `llama.cpp/convert_hf_to_gguf.py` 生成官方布局的 Qwen3 GGUF
- `run-official-qwen3-baseline.sh`
  - 用官方 `llama.cpp` 运行这份从 OmniVoice 抽出的 Qwen3 GGUF，作为 backbone 结构基线
- `compare-official-qwen3-backbone.py`
  - 对比官方 `llama.cpp` converter 产出的 Qwen3 GGUF 与 `ch13` 自己写出的 runtime GGUF 在 backbone 共享张量上的 shape/dtype/数值差异
- `compare-layer-probe-cpu-vs-vulkan.sh`
  - 对比同一层 probe 在 CPU backend 和 Vulkan backend 上的误差表现，用来判断问题是否更偏向 Vulkan 数值路径

当前 C++ 侧的 Vulkan 执行方式，优先参考 `ggml` / `llama.cpp` 一类工程的写法：

- `ggml_backend_load_all()`
- `GPU backend + CPU backend`
- `ggml_backend_sched_new(...)`
- `ggml_backend_sched_alloc_graph(...)`
- `ggml_backend_sched_graph_compute(...)`

也就是尽量避免把 `GGUF` runtime 重新写成一套“手工 Vulkan 调度器”，而是沿着 `ggml` 自己的 scheduler / backend 约定前进。

它当前**还没有**提供：

- 最终的 GGUF 直接驱动整图 Vulkan ggml 推理
- 最终的 KV cache / 常驻 graph / 完整 runtime scheduler 优化

## 目录结构

```text
ch13/
  README.md
  src/
    CMakeLists.txt
    build.sh
    setup-venv.sh
    export-runtime-assets.py
    pack-runtime-gguf.cpp
    inspect-runtime-gguf.cpp
    run.sh
```

## Build

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch13/src
bash ./build.sh
```

## Run

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch13/src
bash ./run.sh "hello world"
```

## Official Qwen3 baseline

先抽出 OmniVoice 里嵌入的标准 Qwen3 主干，再交给官方 `llama.cpp` converter：

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch13/src
bash ./convert-official-qwen3-gguf.sh
```

然后可直接用官方 `llama.cpp` runtime 跑这份 backbone 基线：

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch13/src
bash ./run-official-qwen3-baseline.sh "Hello"
```

如果要看官方 backbone GGUF 和 `ch13` runtime GGUF 在共享权重上的差异：

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch13/src
python3 ./compare-official-qwen3-backbone.py \
  --runtime-gguf ../output/ch13-runtime.gguf \
  --official-gguf ../output/qwen3-from-omnivoice-f16.gguf
```

如果要直接对比某一层（默认 layer23）在 CPU / Vulkan 上的误差：

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch13/src
bash ./compare-layer-probe-cpu-vs-vulkan.sh 23
```

截至当前验证，结论是：

- `layer0` 完整 forward 数值对齐良好
- `layer19~22` 虽有误差，但仍可控
- `layer23` 是主要断点
- `layer23` 的误差主放大链路在 MLP：
  - `silu(gate) * up`
  - `down_proj`
- CPU backend 也有误差，但明显小于 Vulkan backend
- 官方 `llama.cpp` converter 导出的 backbone GGUF 与 `ch13` runtime GGUF 的 backbone 主权重 shape / layout 已对齐

进一步拆分后的当前诊断结论：

- `q_proj / k_proj / v_proj` 单独线性层：CPU 与 Vulkan 差异很小
- `q_norm + rope`：CPU 与 Vulkan 几乎一致
- 直接喂 `q_rope_ref / k_rope_ref / v_proj_ref` 的 `flash_attn_ext` core：CPU 与 Vulkan 几乎一致
- 直接单测 `o_proj` / `down_proj`：CPU 与 Vulkan 有差异，但并没有达到完整 layer23 那种爆炸程度

因此，当前最合理的判断是：

> layer23 的问题不是某一个单独算子“完全错了”，而是前面的 attention-half 已经在 Vulkan 上产生了比 CPU 更大的偏移，而 layer23 的高动态范围 MLP 再把这个偏移继续放大。

也就是说：

- **首个明显偏移点在 attention-half**
- **主放大点在 MLP-half**

最新修复进展：

- 已新增一个 chapter-local Vulkan fused Q/K shader：
  - `q_proj`
  - `k_proj`
  - `q_norm`
  - `k_norm`
  - `rope`
- 这条 fused 路径已经接进：
  - `layer23` attention-half probe
  - `layer23` full-layer probe
  - `runtime_gguf_backbone gpu_fused_qk`

当前实测改善：

- `layer23` attention-half
  - 旧全 GPU：
    - `attn_out.max_abs_diff = 0.0848198`
    - `o_proj.max_abs_diff = 0.349976`
  - 新 fused Q/K：
    - `attn_out.max_abs_diff = 0.0374947`
    - `o_proj.max_abs_diff = 0.190338`

- `layer23` full layer
  - 旧：
    - `layer_out.max_abs_diff = 10.9385`
  - 新 fused Q/K：
    - `layer_out.max_abs_diff = 6.61084`

- full backbone
  - 旧 GPU：
    - `backbone.final_hidden.max_abs_diff = 0.382416`
    - `backbone.logits.mean_abs_diff = 0.0198881`
  - 新 `gpu_fused_qk`：
    - `backbone.final_hidden.max_abs_diff = 0.328278`
    - `backbone.logits.mean_abs_diff = 0.019109`

- iterative decode
  - 已有纯 Vulkan `gpu_fused_qk` 实现
  - 当前 `hello world` 参考下：
    - `iterative_decode.token_mismatch_count = 253`
  - 说明 iterative 路径已接通，但 full-stack 数值对齐还没有完全收敛

这也是为什么单独替换某一个点到 CPU，通常不会立刻让整层误差大幅下降。

## Current Output

运行后会得到：

- `output/generated.wav`
- `output/generated_tokens_i32.bin`
- `output/generated_tokens_meta.txt`
- `output/export/`
- `output/ch13-runtime.gguf`

## Why This Chapter Exists

之前的 `ch13` 更像：

- 导出一堆 `.bin`
- C++/ggml 在 host 端把它们重新拼起来 replay

这不符合你要的方向：

- **GGUF 格式**
- **更整图的 Vulkan ggml 推理**

所以这版 `ch13` 先解决第一个问题：

> 先把运行时资产收拢成一个正确的 `GGUF` 文件。

然后下一步才有意义继续做：

- GGUF loader
- 常驻权重
- 更整图 Vulkan runtime
- 性能优化
