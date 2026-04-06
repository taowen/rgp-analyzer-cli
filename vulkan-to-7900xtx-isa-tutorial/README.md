# Vulkan To 7900 XTX ISA Tutorial

这个子目录面向 `RX 7900 XTX + RADV` 的 Linux Vulkan compute 调优路径。

目标不是泛泛地讲 Vulkan API，而是把下面这条链真正做通：

1. 写一个可运行的 Vulkan compute shader
2. 把它编译成 SPIR-V
3. 让 AMD 驱动把它编译成实际执行的 AMDGPU ISA
4. 抓 `.rgp`
5. 用 `rgp-analyzer-cli` 看 code object、资源占用、runtime 指标和热点 PC

## Chapters

- `ch01`
  基本开发环境、最小 compute 实验、以及 `Vulkan -> SPIR-V -> AMDGPU ISA -> .rgp` 的对应关系。
- `ch02`
  同一个 Vulkan compute harness 的两个 shader 变体，对比资源、ISA 和 runtime 指标怎么变化。
- `ch03`
  用三个 shader 变体区分 global-memory、sync/LDS、普通算术三种不同瓶颈信号。
- `ch04`
  用 scalar 和 `VK_KHR_cooperative_matrix` 两种 matmul shader，看 7900 XTX 在 Vulkan + RADV 下实际暴露什么矩阵相关信号。
- `ch05`
  只用 cooperative matrix / WMMA kernel 变体，演示 tile 复用、K 维展开和寄存器占用之间的取舍。
- `ch06`
  用两个数学结果相同的 LDS kernel，演示 clean access 和 conflict access 对 `avg_stall / sync_wait / wave_lifetime` 的影响。
- `ch07`
  最小 attention baseline。把 `QK^T -> softmax -> V` 变成可抓 `.rgp`、可看 ISA、可做 source/ISA block 对照的最小 Vulkan compute 实验。
- `ch08`
  最小 exact FlashAttention forward。用 `tile + online softmax` 对照 naive attention，真正把 FlashAttention 的前向主循环跑起来。
- `ch09`
  最小 SageAttention forward。实现 `smooth_k + qk_int8 + pv_fp16`，再和 exact FlashAttention 做直接对比。
- `ch10`
  官方 `ggml` Vulkan backend。把 `ggml_mul_mat` 图节点、`ggml-vulkan` 官方 shader、`.rgp` 和 AMDGPU ISA 串起来。
- `ch11`
  官方 `whisper.cpp` + `ggml` Vulkan backend。先把 `tiny.en` ASR 跑通，再转到 encoder/attention 热点调优。
- `ch12`
  官方 `OmniVoice` + `PyTorch ROCm`。在 AMD GPU 上生成一段英语语音，再用 `Whisper tiny.en` 做 ASR 回验。
- `ch13`
  **Vulkan-first** 的 `ggml` OmniVoice iterative decode。用官方 OmniVoice 导出参考，再用 `ggml-vulkan` 在 `RX 7900 XTX` 上生成 `ggml` token，并把 token 解码成 `wav` 做 ASR 回验。
- `ch13-old`
  旧的实验性 bridge / CPU-fallback 版本，保留早期 `ggml` replay 与数值对齐探索过程。

## Chapter Layout

每一章都使用同样的结构：

```text
vulkan-to-7900xtx-isa-tutorial/
  chXX/
    README.md
    src/
      ... self-contained experiment code and scripts ...
```

每章都要求：

- `src/` 代码自包含，可直接构建和运行
- 能在 Linux + RADV 上抓 `.rgp`
- 能接上 `rgp-analyzer-cli`
