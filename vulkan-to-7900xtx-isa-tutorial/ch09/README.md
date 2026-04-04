# Ch09: Minimal SageAttention Forward

这一章不再发明 “sage-style” 名字。

它直接实现一个**教程尺度的 SageAttention forward**，核心路径是：

1. `smooth_k`
2. `Q/K` INT8 quantization
3. `PV` 走 FP16 value 路径
4. softmax 仍保持 exact online update

这对应公开 SageAttention API 里的 `qk_int8 + pv_fp16` 方向。

## 这一章具体做了什么

### `flash_attention.comp`

- 用 float `Q/K/V`
- 跑 exact FlashAttention forward
- 作为本章 baseline

### `sage_attention.comp`

- `K` 先做 per-dimension mean subtraction
- `Q/K` 做 per-block INT8 quantization
- `V` 先 pack 成 FP16
- 在 shader 里用量化后的 `Q/K` 计算 score
- 用 packed FP16 `V` 做输出累积

## 边界

这章实现的是**真实的 SageAttention 核心数值路径**，但仍然是教程尺度，不是官方 CUDA kernel 的工程复刻。

它没有去复刻：

- 官方 kernel 的 launch topology
- 特定 GPU 的 tensor-core / MMA 内核写法
- 更复杂的 kernel fusion

但它确实实现了：

- `smooth_k`
- `qk_int8`
- `pv_fp16`

## 目录结构

```text
ch09/
  README.md
  captures/
  src/
    main.cpp
    build.sh
    compile-shaders.sh
    run-flash-attention.sh
    run-sage-attention.sh
    capture-flash-attention.sh
    capture-sage-attention.sh
    analyze-flash-attention.sh
    analyze-sage-attention.sh
    compare.sh
    shaders/
      flash_attention.comp
      sage_attention.comp
```

## Build

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch09/src
bash ./compile-shaders.sh
bash ./build.sh
```

`ch09` 现在会优先使用本地编出来的：

```text
/tmp/glslang/build/StandAlone/glslang
```

原因是系统 `glslc` 不支持 `GL_EXT_integer_dot_product`，而本章的优化版 Sage kernel 需要
`dotPacked4x8EXT` 才能真正走 packed int8 dot-product 路径。

## Run

```bash
bash ./run-flash-attention.sh 32
bash ./run-sage-attention.sh 32
```

一组实际输出：

```text
flash_attention:
  gpu_total_ms: 3.14
  gpu_avg_dispatch_us: 49.00
  checksum: 427.87

sage_attention:
  gpu_total_ms: 1.68
  gpu_avg_dispatch_us: 26.23
  checksum: 427.87
```

也就是：

- `gpu_total_ms: 3.14 -> 1.68`，约 `-46.5%`
- `gpu_avg_dispatch_us: 49.00 -> 26.23`，约 `-46.5%`

## Capture

```bash
bash ./capture-flash-attention.sh 8
bash ./capture-sage-attention.sh 8
```

当前 capture：

```text
captures/flash_attention.rgp size_bytes=31811400
captures/sage_attention.rgp size_bytes=6503064
```

## Analyze

```bash
bash ./analyze-flash-attention.sh
bash ./analyze-sage-attention.sh
bash ./compare.sh
```

## 本章关注的指标

- `instructions`
- `avg_stall_per_inst`
- `sync_wait_share`
- `immed_stall_per_inst`
- `VGPR / LDS`
- `avg_wave_lifetime`

## 真实结果

### `flash_attention -> sage_attention`

```text
gpu_total_ms: 3.14 -> 1.68 (delta=-1.46, ratio=-46.5%)
gpu_avg_dispatch_us: 49.00 -> 26.23 (delta=-22.77, ratio=-46.5%)
max_abs_error: 0.00 -> 0.00
checksum: 427.87 -> 427.87
```

## 读法

这章要回答的问题是：

- 当 exact FlashAttention forward 改成 `smooth_k + qk_int8 + pv_fp16` 后，
  低层代价是怎么变化的？

也就是：

- 哪些开销被量化路径压下去了
- 哪些同步或等待反而抬起来了

在这轮真实结果里，最重要的变化不是微调 `barrier()` 或 `exp()`，而是：

- 把 `Q/K` 这条路径改成真正的 packed int8 dot-product
- host 显式打开 `shaderIntegerDotProduct`
- shader 编译改走支持 `GL_EXT_integer_dot_product` 的新前端

也就是说，本章当前版本已经不再是“手工 unpack int8 再做标量乘加”，而是真的在吃
Vulkan integer dot-product 能力。

所以这章真正教的是：

- `smooth_k + qk_int8 + pv_fp16` 是真实 SageAttention 路径
- 想让它真正快起来，关键不是继续磨边角同步，而是把 `quantized_qk` 变成 packed dot-product
- 这条路径在 `RX 7900 XTX + RADV` 上已经能把本章的 Sage kernel 拉到明显快于 Flash baseline
