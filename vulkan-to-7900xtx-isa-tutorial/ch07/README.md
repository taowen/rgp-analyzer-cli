# Ch07: Attention Baseline

这一章从最朴素的 attention kernel 开始。

目标不是先追求最快，而是把 attention 的基本结构变成一份可以抓 `.rgp`、可以看 ISA、可以和后面 `flash attention`/`sage-style` 变体做对照的最小实验。

## 本章目标

- 用一个最小 Vulkan compute shader 跑通 `Q * K^T -> softmax -> V`
- 让同一份 `.rgp` 能关联到：
  - focused shader
  - 静态 ISA
  - 源码块
- 建立后两章的 baseline：
  - `ch08`: flash-style online softmax
  - `ch09`: sage-style subgroup reduction

## 目录结构

```text
ch07/
  README.md
  captures/
  src/
    main.cpp
    build.sh
    compile-shaders.sh
    run.sh
    capture.sh
    analyze.sh
    shaders/
      attention_naive.comp
```

## Kernel 结构

这个 baseline shader 的特点很明确：

- 一个 workgroup 负责一行 query
- 先把整行 `QK^T` score 写到 `shared float scores[64]`
- 再由 lane 0 做 softmax 的 `max/sum`
- 最后所有 lane 再读 `V` 做输出累加

也就是说，它有：

- 完整的 score materialization
- 显式 `shared` 数组
- 多个 `barrier()`
- 很重的 global memory 访问

这正是后面 flash-style 改写要解决的对象。

## Build

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch07/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch07/src
bash ./run.sh 64
```

程序会：

- 生成固定的 `Q / K / V`
- 在 CPU 上算 reference
- 在 GPU 上跑 attention
- 校验 `max_abs_error`
- 打印 GPU 时间和 checksum

一组实际输出：

```text
device: AMD Radeon RX 7900 XTX (RADV NAVI31)
gpu_total_ms: 0.46
gpu_avg_dispatch_us: 57.62
max_abs_error: 0.00
checksum: 427.87
dispatch_ok shader=.../attention_naive.spv seq_len=64 head_dim=64 dispatches=8 repeats=8 checksum=427.87
```

## Capture

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch07/src
bash ./capture.sh 8
```

本章当前教学版 capture：

```text
captures/attention_naive.rgp size_bytes=3220536
```

## Analyze

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch07/src
bash ./analyze.sh
```

这会调用 `rgp-analyzer-cli` 的：

- `shader-focus`
- `code-object-isa`

## 真实结果

`shader-focus` 当前输出：

```text
trace_quality: level=dispatch_isa sqtt_bytes=21341088 dispatch_spans=48 mapped_dispatch=64/64
resource: vgpr=48 sgpr=128 lds=512 scratch=0 wavefront=64
runtime(global): instructions=1053024 avg_stall=5.96 stalled_inst_share=0.16 avg_wave_lifetime=640726.00 stall_share=0.83 occ_avg=388.79 occ_max=512
runtime_proxies: global_mem_duration_share=0.01 lds_duration_share=0.02 sync_wait_share=0.45 sync_wait_per_inst=5.72 immed_stall_per_inst=41.24 lds_stall_per_inst=0.55
```

`code-object-isa` 当前能把这几个源码块和 ISA 段对应起来：

```text
source_isa_blocks:
  - bounds_check
    source line=41 match=return;
  - shared_exchange
    source line=30 match=shared float scores[64];
    source line=31 match=shared float denom;
    source line=54 match=barrier();
    source line=69 match=barrier();
    isa pc=0x78 s_waitcnt lgkmcnt(0)
```

## 这一章应该看什么

你应该重点盯这些量：

- `global_mem_duration_share`
- `sync_wait_share`
- `immed_stall_per_inst`
- `avg_stall_per_inst`
- `avg_wave_lifetime`
- `source_isa_blocks`

如果这份 baseline capture 已经能稳定跑出 `dispatch_isa`，那后面两章就能在同一个问题上做结构性改写，而不是换一个完全不同的 kernel。

这章当前已经说明了两件事：

- naive attention 的 `sync_wait` 很重，`immed_stall_per_inst=41.24`
- score materialization + barrier 结构已经在 `source_isa_blocks` 里显式可见
