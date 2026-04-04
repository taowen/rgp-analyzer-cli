# Ch05: Tuning Cooperative Matrix Kernels

这一章只讲一件事：

- 同样都走 `VK_KHR_cooperative_matrix`
- 不同的 WMMA kernel 写法，仍然会有不同的 `VGPR / occupancy / avg_stall / wave_lifetime`

这更接近真实库里的调优问题。到了这一章，不再比较 `scalar vs WMMA`，而是比较：

1. `wmma_tile16`
   一个 workgroup 计算一个 `16x16` 输出 tile。
2. `wmma_row2`
   一个 workgroup 计算同一行上的两个 `16x16` 输出 tile，复用同一个 `A` tile。
3. `wmma_k2`
   一个 workgroup 仍然只算一个 `16x16` 输出 tile，但每次循环处理两个 `K` tile，减少 loop 次数。

## 本章目标

这一章要回答的问题是：

- 都是 cooperative matrix，为什么还需要多个 kernel 变体
- `row2` 这种“多产出 tile”写法，换来了什么，失去了什么
- `k2` 这种“减少循环次数”写法，是真的更好，还是只是看起来更少指令

## 目录结构

```text
vulkan-to-7900xtx-isa-tutorial/ch05/
  README.md
  src/
    main.cpp
    shaders/
      wmma_tile16.comp
      wmma_row2.comp
      wmma_k2.comp
    compile-shaders.sh
    build.sh
    run-wmma-tile16.sh
    run-wmma-row2.sh
    run-wmma-k2.sh
    capture-wmma-tile16.sh
    capture-wmma-row2.sh
    capture-wmma-k2.sh
    analyze-wmma-tile16.sh
    analyze-wmma-row2.sh
    analyze-wmma-k2.sh
    compare-row2.sh
    compare-k2.sh
```

## Build

```bash
cd ~/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch05/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
bash ./run-wmma-tile16.sh 64
bash ./run-wmma-row2.sh 64
bash ./run-wmma-k2.sh 64
```

这 3 条我已经实际跑过。真实输出摘录：

```text
variant=wmma_tile16 device=AMD Radeon RX 7900 XTX (RADV NAVI31) vendor=0x1002 device=0x744c queue_family=0
cooperative_matrix_extension=1 cooperative_matrix_feature=1 cooperative_matrix_proc=1 cooperative_matrix_property_count=14
dispatch_ok variant=wmma_tile16 matrix_size=32 repeats=64 checksum=2.05973e+06

variant=wmma_row2 device=AMD Radeon RX 7900 XTX (RADV NAVI31) vendor=0x1002 device=0x744c queue_family=0
cooperative_matrix_extension=1 cooperative_matrix_feature=1 cooperative_matrix_proc=1 cooperative_matrix_property_count=14
dispatch_ok variant=wmma_row2 matrix_size=32 repeats=64 checksum=2.05973e+06

variant=wmma_k2 device=AMD Radeon RX 7900 XTX (RADV NAVI31) vendor=0x1002 device=0x744c queue_family=0
cooperative_matrix_extension=1 cooperative_matrix_feature=1 cooperative_matrix_proc=1 cooperative_matrix_property_count=14
dispatch_ok variant=wmma_k2 matrix_size=32 repeats=64 checksum=2.05973e+06
```

## Capture

```bash
bash ./capture-wmma-tile16.sh 64
bash ./capture-wmma-row2.sh 64
bash ./capture-wmma-k2.sh 64
```

真实抓到的 `.rgp` 大小：

```text
captures/wmma_tile16.rgp size_bytes=332868
captures/wmma_row2.rgp size_bytes=351732
captures/wmma_k2.rgp size_bytes=345772
```

## Analyze

```bash
bash ./analyze-wmma-tile16.sh
bash ./analyze-wmma-row2.sh
bash ./analyze-wmma-k2.sh
bash ./compare-row2.sh
bash ./compare-k2.sh
```

## 真实结果

### `wmma_tile16`

这是这一章的 WMMA 基线。

```text
trace_quality: level=dispatch_isa sqtt_bytes=137088 dispatch_spans=384 mapped_dispatch=32/64
resource: entry_point=_amdgpu_cs_main vgpr=24 sgpr=128 lds=0 scratch=0 wavefront=64
runtime(global): instructions=210 avg_stall=2.10 stalled_inst_share=0.08 avg_wave_lifetime=977.50 stall_share=0.32 occ_avg=3.26 occ_max=12
runtime_proxies: global_mem_duration_share=0.05 sync_wait_share=0.24 sync_wait_per_inst=2.21 immed_stall_per_inst=20.09
```

这个基线说明：

- cooperative matrix 路径已经把总指令数压到 `210`
- `avg_stall_per_inst` 和 `sync_wait` 都已经比较低
- `VGPR=24`
- `occ_avg=3.26`

### `wmma_row2`

这个版本每个 workgroup 一次算两个横向输出 tile，重点是复用同一个 `A` tile。

```text
trace_quality: level=dispatch_isa sqtt_bytes=138112 dispatch_spans=384 mapped_dispatch=17/64
resource: entry_point=_amdgpu_cs_main vgpr=36 sgpr=128 lds=0 scratch=0 wavefront=64
runtime(global): instructions=147 avg_stall=1.88 stalled_inst_share=0.09 avg_wave_lifetime=1172.00 stall_share=0.33 occ_avg=1.92 occ_max=6
runtime_proxies: global_mem_duration_share=0.07 sync_wait_share=0.25 sync_wait_per_inst=2.01 immed_stall_per_inst=13.75
```

相对 `wmma_tile16`，它的变化是：

- `VGPR: 24 -> 36`
- `instructions: 210 -> 147`
- `avg_stall_per_inst: 2.10 -> 1.88`
- `sync_wait_cycles_per_inst: 2.21 -> 2.01`
- `immed_stall_per_inst: 20.09 -> 13.75`
- `occupancy_average_active: 3.26 -> 1.92`
- `avg_wave_lifetime: 977.50 -> 1172.00`

真实 compare 摘录：

```text
avg_stall_per_inst: 2.10 -> 1.88 (delta=-0.23, ratio=-10.8%)
hotspot_avg_duration_per_hit: 6.63 -> 5.78 (delta=-0.86, ratio=-12.9%)
occupancy_average_active: 3.26 -> 1.92 (delta=-1.35, ratio=-41.2%)
immed_stall_per_inst: 20.09 -> 13.75 (delta=-6.34, ratio=-31.6%)
VALU: 98 -> 68 (delta=-30, ratio=-30.6%)
```

这就是很典型的 WMMA 调优权衡：

- 指令和等待更少
- 但寄存器更多，occupancy 更低

### `wmma_k2`

这个版本不改输出 tile 形状，只改 K 维循环结构：每次处理两个 `K` tile。

```text
trace_quality: level=dispatch_isa sqtt_bytes=138432 dispatch_spans=384 mapped_dispatch=32/64
resource: entry_point=_amdgpu_cs_main vgpr=24 sgpr=128 lds=0 scratch=0 wavefront=64
runtime(global): instructions=188 avg_stall=2.41 stalled_inst_share=0.09 avg_wave_lifetime=963.50 stall_share=0.33 occ_avg=3.31 occ_max=12
runtime_proxies: global_mem_duration_share=0.05 sync_wait_share=0.25 sync_wait_per_inst=2.52 immed_stall_per_inst=22.70
```

相对 `wmma_tile16`，它的变化是：

- `VGPR: 24 -> 24`
- `instructions: 210 -> 188`
- `avg_stall_per_inst: 2.10 -> 2.41`
- `sync_wait_cycles_per_inst: 2.21 -> 2.52`
- `immed_stall_per_inst: 20.09 -> 22.70`
- `occupancy_average_active: 3.26 -> 3.31`

真实 compare 摘录：

```text
avg_stall_per_inst: 2.10 -> 2.41 (delta=+0.31, ratio=+14.7%)
hotspot_avg_duration_per_hit: 6.63 -> 7.22 (delta=+0.58, ratio=+8.8%)
immed_stall_per_inst: 20.09 -> 22.70 (delta=+2.61, ratio=+13.0%)
sync_wait_cycles: 464.00 -> 474.00 (delta=+10.00, ratio=+2.2%)
```

这说明“循环更短”不等于“更快”。  
`wmma_k2` 的总指令确实更少，但单条指令上的等待反而更高。

## 这一章真正想说明什么

这一章是纯 WMMA 的调优例子，所以重点不是“有没有走到 cooperative matrix”，而是：

1. 都已经走到 cooperative matrix 之后，还会有多个 kernel 变体。
2. `wmma_row2` 说明：
   - 复用更多 tile 数据，能继续减少指令和等待
   - 但会推高 `VGPR`，压低 occupancy
3. `wmma_k2` 说明：
   - 仅仅减少 loop 次数，不一定带来正收益
   - 还可能把 `avg_stall_per_inst` 和 `immed_stall_per_inst` 拉高

这就是学习者在 WMMA 调优里最该建立的判断：

- 先看 `VGPR`
- 再看 `instructions`
- 再看 `avg_stall_per_inst / immed_stall_per_inst / sync_wait_cycles_per_inst`
- 最后再判断这个变体是不是值得留下

## 和 `ch04` 的关系

- `ch04` 解决的是：
  - 7900 XTX + Vulkan + RADV 到底能不能跑 cooperative matrix
  - 最小 cooperative matrix 例子长什么样
- `ch05` 解决的是：
  - 都已经进了 cooperative matrix 路径之后，怎么继续做 kernel 写法调优

如果你想先确认最小 cooperative matrix 路径和 `VK_KHR_cooperative_matrix` 属性，先看 `ch04`。  
如果你想看纯 WMMA kernel 的写法取舍，直接看这一章。
