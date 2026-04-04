# Ch06: LDS Bank Conflicts

这一章回答一个很容易被忽略的问题：

- 同样都用了 LDS/shared memory
- 同样都有 barrier
- 访问模式不同，为什么运行时表现会差这么多

这一章用两个最小 shader 对照：

1. `lds_clean.comp`
   每个 lane 读写相邻的 shared slot。
2. `lds_conflict.comp`
   每个 lane 读写 `lid * 32` 这样的 shared slot，故意制造更坏的 bank 映射。

它们的数学结果完全一样，区别只在 LDS 索引步长。

## 本章目标

- 构造一个最小的 LDS bank-conflict 对照实验
- 抓两份 `.rgp`
- 用 `rgp-analyzer-cli` 看：
  - `lds_duration_share`
  - `lds_stall_per_inst`
  - `sync_wait_share`
  - `avg_stall_per_inst`
  - `avg_wave_lifetime`

## 目录结构

```text
vulkan-to-7900xtx-isa-tutorial/ch06/
  README.md
  src/
    main.cpp
    shaders/
      lds_clean.comp
      lds_conflict.comp
    compile-shaders.sh
    build.sh
    run-clean.sh
    run-conflict.sh
    capture-clean.sh
    capture-conflict.sh
    analyze-clean.sh
    analyze-conflict.sh
    compare.sh
```

## Build

```bash
cd ~/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch06/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
bash ./run-clean.sh 128
bash ./run-conflict.sh 128
```

这两条我已经实际跑过。真实输出是：

```text
dispatch_ok variant=lds_clean element_count=256 workgroups=4 repeats=64 checksum=1668864
dispatch_ok variant=lds_conflict element_count=256 workgroups=4 repeats=64 checksum=1668864
```

## Capture

```bash
bash ./capture-clean.sh 128
bash ./capture-conflict.sh 128
```

真实抓到的 `.rgp` 大小：

```text
captures/clean.rgp size_bytes=442768
captures/conflict.rgp size_bytes=458440
```

## Analyze

```bash
bash ./analyze-clean.sh
bash ./analyze-conflict.sh
bash ./compare.sh
```

## 真实结果

### `lds_clean`

这份 capture 的作用是给一个“LDS 使用方式相对干净”的基线。

```text
trace_quality: level=dispatch_isa sqtt_bytes=245984 dispatch_spans=768 mapped_dispatch=42/128
resource: entry_point=_amdgpu_cs_main vgpr=12 sgpr=128 lds=512 scratch=0 wavefront=64
runtime(global): instructions=81 avg_stall=5.52 stalled_inst_share=0.19 avg_wave_lifetime=592.00 stall_share=0.33 occ_avg=1.76 occ_max=10
runtime_proxies: global_mem_duration_share=0.02 lds_duration_share=0.03 sync_wait_share=0.26 sync_wait_per_inst=5.70 immed_stall_per_inst=29.80 lds_stall_per_inst=0.00
```

这里最关键的是：

- `lds=512`
- `avg_stall_per_inst=5.52`
- `sync_wait_share=0.26`
- `avg_wave_lifetime=592.00`

这说明 LDS 和 barrier 确实已经进入运行时，但还没有恶化到很重。

### `lds_conflict`

这个版本和 `lds_clean` 的数学结果相同，但把 shared slot 从 `lid` 改成了 `lid * 32`。

```text
trace_quality: level=dispatch_isa sqtt_bytes=247840 dispatch_spans=768 mapped_dispatch=28/128
resource: entry_point=_amdgpu_cs_main vgpr=12 sgpr=128 lds=8192 scratch=0 wavefront=64
runtime(global): instructions=54 avg_stall=14.59 stalled_inst_share=0.19 avg_wave_lifetime=822.50 stall_share=0.57 occ_avg=2.33 occ_max=10
runtime_proxies: global_mem_duration_share=0.01 lds_duration_share=0.02 sync_wait_share=0.49 sync_wait_per_inst=14.78 immed_stall_per_inst=78.80 lds_stall_per_inst=0.00
```

这里最关键的是：

- `lds: 512 -> 8192`
- `avg_stall_per_inst: 5.52 -> 14.59`
- `sync_wait_share: 0.26 -> 0.49`
- `sync_wait_cycles_per_inst: 5.70 -> 14.78`
- `immed_stall_per_inst: 29.80 -> 78.80`
- `avg_wave_lifetime: 592.00 -> 822.50`

也就是说，虽然总指令数反而下降了：

- `instructions: 81 -> 54`

但单条指令上的等待、wave 生命周期和同步等待都明显变重了。

### `clean -> conflict` compare

真实 compare 输出摘录：

```text
lds_size: 512 -> 8192
avg_stall_per_inst: 5.52 -> 14.59 (delta=+9.07, ratio=+164.4%)
hotspot_avg_duration_per_hit: 16.89 -> 25.43 (delta=+8.54, ratio=+50.5%)
sync_wait_share: 0.26 -> 0.49 (delta=+0.22, ratio=+86.5%)
sync_wait_cycles_per_inst: 5.70 -> 14.78 (delta=+9.07, ratio=+159.1%)
immed_stall_per_inst: 29.80 -> 78.80 (delta=+49.00, ratio=+164.4%)
avg_wave_lifetime: 592.00 -> 822.50 (delta=+230.50, ratio=+38.9%)
```

这一组数字说明：

- 同样都在用 LDS
- 同样都有 barrier
- 只是把 shared 索引步长改坏，就会让等待显著增加

这正是学习者应该从这一章得到的最核心印象。

## 这一章应该怎么看

这章不要只盯 `lds_duration_share`。  
真正更有判别力的是这组组合信号：

- `lds_size`
- `avg_stall_per_inst`
- `sync_wait_share`
- `sync_wait_cycles_per_inst`
- `immed_stall_per_inst`
- `avg_wave_lifetime`

在这次真实实验里，`lds_conflict` 最明显的不是“LDS 指令更多”，而是：

- 同样的 LDS 路径让 wave 等更久
- `WAIT / IMMED` 更重
- 单条指令的平均等待显著上升

## 这一章和前面几章的关系

- `ch03` 讲的是：怎么区分 global memory、sync/LDS、普通算术三种信号
- `ch06` 讲的是：都已经确认是 LDS/sync 路径之后，访问模式本身还能继续把性能拉坏

所以这章是前面“读信号”之后的下一步：

- 不只是知道“问题在 LDS”
- 而是知道“LDS 访问模式也会决定好坏”
