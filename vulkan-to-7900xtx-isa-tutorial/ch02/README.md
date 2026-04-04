# Ch02: One Shader, Two Variants

这一章把 `ch01` 往前推一步：

- 不是只看“有一份 ISA”
- 而是写两个功能相近但写法不同的 compute shader
- 然后用 `rgp-analyzer-cli` 看它们的资源、ISA 和 runtime 指标怎么变

这章的重点是：

- baseline shader
- register-pressure variant
- `.rgp` A/B compare

## 本章目标

- 运行同一个 Vulkan compute harness 的两个 shader 变体
- 为两个变体各抓一份 `.rgp`
- 用 `rgp-analyzer-cli` 比较：
  - `resource-summary`
  - `shader-focus`
  - `code-object-isa`
  - `compare-shader-focus`

## 这一章要观察什么

我们关心的是同一个 kernel 在两种写法下的变化：

- `VGPR / SGPR / LDS / scratch`
- `avg_stall_per_inst`
- `occupancy_average_active`
- `runtime_top_hotspot`
- `code-object-isa`

这一章故意选“寄存器压力”作为第一类变化，因为它最容易稳定讲清：

- 源码块变复杂了什么
- 为什么资源占用会变
- ISA 里增加了哪些算术和临时值相关指令

## 目录结构

```text
vulkan-to-7900xtx-isa-tutorial/ch02/
  README.md
  src/
    main.cpp
    shaders/
      baseline.comp
      reg_pressure.comp
    compile-shaders.sh
    build.sh
    run-baseline.sh
    run-reg-pressure.sh
    capture-baseline.sh
    capture-reg-pressure.sh
    analyze-baseline.sh
    analyze-reg-pressure.sh
    compare.sh
```

## Build

```bash
cd ~/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch02/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
bash ./run-baseline.sh 128
bash ./run-reg-pressure.sh 128
```

两条命令都会：

- 选一张 Vulkan compute 设备
- 运行同一个 host harness
- 校验 buffer 输出
- 打印 checksum

## Capture

```bash
bash ./capture-baseline.sh 128
bash ./capture-reg-pressure.sh 128
```

捕获会分别保存到：

```text
vulkan-to-7900xtx-isa-tutorial/ch02/captures/baseline.rgp
vulkan-to-7900xtx-isa-tutorial/ch02/captures/reg_pressure.rgp
```

## Analyze

单独分析：

```bash
bash ./analyze-baseline.sh
bash ./analyze-reg-pressure.sh
```

对比分析：

```bash
bash ./compare.sh
```

这一章的建议阅读顺序是：

1. 先看 `resource-summary`
2. 再看 `shader-focus`
3. 再看 `code-object-isa`
4. 最后看 `compare-shader-focus`

## 你应该看到什么

这一章不是追求端到端性能，而是追求“一个局部写法改变了什么”。

当前这章在 `RX 7900 XTX (RADV NAVI31)` 上的一次真实 A/B 结果是：

- `vgpr_count: 12 -> 24`
- `instructions: 39 -> 72`
- `avg_stall_per_inst: 1.62 -> 0.19`
- `occupancy_average_active: 0.99 -> 1.15`

也就是说，这一章已经能稳定展示：

- `reg_pressure.comp` 的源码块更多
- 它的 ISA 比 baseline 多更多中间算术
- `VGPR` 明显上升
- runtime 指标也随之变化

单独分析时，你应该能看到类似这样的差别：

```text
baseline:
  vgpr=12
  source_isa_blocks:
    - label=value_compute
      isa pc=0x20 v_mul_lo_u32 ...
      isa pc=0x30 v_add_nc_u32_e32 ...

reg_pressure:
  vgpr=24
  source_isa_blocks:
    - label=value_compute
      isa pc=0x20 v_add_nc_u32_e32 ...
      isa pc=0x3c v_mul_lo_u32 ...
      isa pc=0x5c v_xad_u32 ...
      isa pc=0xb0 v_xor3_b32 ...
```

如果这条链跑通，后面的章节就可以继续讲：

- LDS / barrier
- subgroup / reduction
- 更真实的 shader 调优
