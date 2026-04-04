# Ch08: Minimal Exact FlashAttention Forward

这一章实现的是一个**真正的、精确的 FlashAttention forward 教学版**。

它不是官方 CUDA kernel 的逐行复刻，但算法上已经包含 FlashAttention 前向最关键的 3 个部分：

1. 不 materialize 全部 `QK^T`。
2. 按 `K/V` tile 迭代。
3. 用 online softmax 维护 `m_i / l_i / O_i`。

## 本章实现

### `attention_naive.comp`

- 先写完整行 `scores[seq_len]`
- 再由 lane 0 做整行 `max/sum`
- 最后再读完整 `V`

### `flash_attention.comp`

- 每次只加载一个 `K/V` tile
- 对 tile 内 score 做 online softmax 更新
- 在寄存器里维护输出向量 `acc`
- 最后一次性写回 `O_i`

## 目录结构

```text
ch08/
  README.md
  captures/
  src/
    main.cpp
    build.sh
    compile-shaders.sh
    run-naive.sh
    run-flash-attention.sh
    capture-naive.sh
    capture-flash-attention.sh
    analyze-naive.sh
    analyze-flash-attention.sh
    compare.sh
    shaders/
      attention_naive.comp
      flash_attention.comp
```

## Build

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch08/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
bash ./run-naive.sh 32
bash ./run-flash-attention.sh 32
```

一组实际输出：

```text
naive:
  gpu_total_ms: 0.47
  gpu_avg_dispatch_us: 58.24
  checksum: 427.87

flash_attention:
  gpu_total_ms: 0.48
  gpu_avg_dispatch_us: 59.46
  checksum: 427.87
```

## Capture

```bash
bash ./capture-naive.sh 8
bash ./capture-flash-attention.sh 8
```

当前 capture：

```text
captures/attention_naive.rgp size_bytes=2490368
captures/flash_attention.rgp size_bytes=4292608
```

## Analyze

```bash
bash ./analyze-naive.sh
bash ./analyze-flash-attention.sh
bash ./compare.sh
```

## 本章关注的指标

- `avg_stall_per_inst`
- `sync_wait_share`
- `immed_stall_per_inst`
- `VGPR / LDS`
- `avg_wave_lifetime`

## 真实结果

### `attention_naive -> flash_attention`

```text
vgpr_count: 48 -> 192
lds_size: 512 -> 8704
instructions: 150432 -> 198024
avg_stall_per_inst: 6.92 -> 3.57 (delta=-3.35, ratio=-48.4%)
sync_wait_share: 0.57 -> 0.40 (delta=-0.16, ratio=-28.5%)
sync_wait_cycles_per_inst: 7.04 -> 3.65 (delta=-3.39, ratio=-48.1%)
immed_stall_per_inst: 50.95 -> 24.49 (delta=-26.46, ratio=-51.9%)
avg_wave_lifetime: 623759.00 -> 595839.67 (delta=-27919.33, ratio=-4.5%)
```

## 读法

这章要回答的问题只有一个：

- 当 attention forward 从“整行 materialize”改成“tile + online softmax”时，
  `.rgp` 里的 ISA、资源和 runtime 指标会怎么变。

如果这章跑通，后面 `ch09` 才有资格去讲真正带量化和 smoothing 的 SageAttention 路线。

在这份真实结果里，最关键的观察是：

- exact FlashAttention forward 明显降低了
  - `avg_stall_per_inst`
  - `sync_wait_share`
  - `immed_stall_per_inst`
- 但它也明显抬高了
  - `VGPR`
  - `LDS`

这正是 FlashAttention 教程里最值得看的一组取舍。
