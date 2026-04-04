# Ch04: Can 7900 XTX Do WMMA From Vulkan?

这一章用 `VK_KHR_cooperative_matrix` 做一个真正的最小实验。

目标不是先做最快的 GEMM，而是先确认三件事：

- 这台 `RX 7900 XTX + RADV` 是否真的暴露了 cooperative matrix 能力
- Vulkan shader 能不能真正编译并运行 `GL_KHR_cooperative_matrix`
- `.rgp` 和 ISA 里会不会出现更像矩阵路径的指令

## 本章目标

- 运行两个矩阵乘 shader：
  - `scalar_matmul.comp`
  - `cooperative_matmul.comp`
- 抓两份 `.rgp`
- 用 `rgp-analyzer-cli` 看：
  - `VGPR / SGPR / LDS`
  - `WAIT / IMMED / sync_wait`
  - `source_isa_blocks`
  - `code-object-isa`
  - cooperative matrix property 枚举

## 两个变体

- `scalar_matmul.comp`
  朴素版本，输入是 `f16`，每个 output 元素独立做一条 dot product，累加是 `f32`

- `cooperative_matmul.comp`
  启用 `GL_KHR_cooperative_matrix`，做一个 subgroup-scope 的 `16x16 x 16x16 -> 16x16` cooperative matrix multiply-add。
  在这台 `7900 XTX + RADV` 上，稳定可运行的最小配置是 `local_size_x = 64`。

这章最关键的观察不是“谁更快”，而是：

- host 端是不是列出了 `VK_KHR_cooperative_matrix`
- cooperative matrix property 里有没有 `16x16x16 f16->f32 subgroup`
- ISA 里是不是出现了 `wmma` / `mfma` 这类更像矩阵路径的指令

## 目录结构

```text
vulkan-to-7900xtx-isa-tutorial/ch04/
  README.md
  src/
    main.cpp
    shaders/
      scalar_matmul.comp
      cooperative_matmul.comp
    compile-shaders.sh
    build.sh
    run-scalar.sh
    run-cooperative.sh
    capture-scalar.sh
    capture-cooperative.sh
    analyze-scalar.sh
    analyze-cooperative.sh
    compare.sh
```

## Build

```bash
cd ~/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch04/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
bash ./run-scalar.sh 64
bash ./run-cooperative.sh 64
```

## Capture

```bash
bash ./capture-scalar.sh 64
bash ./capture-cooperative.sh 64
```

## Analyze

```bash
bash ./analyze-scalar.sh
bash ./analyze-cooperative.sh
bash ./compare.sh
```

## 实跑结果

下面这些输出都已经在这台机器上实际跑过：

- 设备：`AMD Radeon RX 7900 XTX (RADV NAVI31)`
- cooperative matrix feature:
  - `cooperative_matrix_extension=1`
  - `cooperative_matrix_feature=1`
  - `cooperative_matrix_proc=1`
  - `cooperative_matrix_property_count=14`

其中一组最关键的 property 是：

```text
coopmat_property[2] M=16 N=16 K=16 AType=7 BType=7 CType=9 ResultType=9 scope=3
```

也就是这一章用到的：

- `16x16x16`
- `f16 x f16 -> f32`
- `subgroup scope`

### Run

```text
$ bash ./run-scalar.sh 8
dispatch_ok variant=scalar matrix_size=16 repeats=8 checksum=97325

$ bash ./run-cooperative.sh 8
dispatch_ok variant=cooperative matrix_size=16 repeats=8 checksum=97325
```

两条路径都通过了 CPU 侧矩阵乘校验。

### Scalar analysis

```text
trace_quality: level=dispatch_isa sqtt_bytes=238880 dispatch_spans=768 mapped_dispatch=61/128
resource: vgpr=12 sgpr=128 lds=0 scratch=0 wavefront=64
runtime(global): instructions=843 avg_stall=6.77 stall_share=0.72 occ_avg=8.95 occ_max=27
runtime_proxies: sync_wait_share=0.56 sync_wait_per_inst=6.91 immed_stall_per_inst=50.06
```

`code_object_isa` 里能看到普通 ALU / buffer load 路径，例如：

```text
pc=0x58 v_mul_lo_u32 ...
pc=0x70 buffer_load_d16_b16 ...
pc=0x94 v_fma_mix_f32 ...
```

### Cooperative analysis

```text
trace_quality: level=dispatch_isa sqtt_bytes=204960 dispatch_spans=768 mapped_dispatch=17/128
resource: vgpr=36 sgpr=128 lds=0 scratch=0 wavefront=64
runtime(global): instructions=32 avg_stall=4.06 stall_share=0.28 occ_avg=0.61 occ_max=3
runtime_proxies: global_memory_duration_share=0.06 sync_wait_share=0.23 immed_stall_per_inst=18.57
```

这章最关键的证据在 `code_object_isa`：

```text
source_isa_blocks:
  - label=cooperative_matrix
    source line=27 match=coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> mat_a_tile;
    source line=28 match=coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> mat_b_tile;
    source line=29 match=coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> mat_c_tile =
    isa pc=0xa0 v_wmma_f32_16x16x16_f16 ...
```

也就是说，在这台卡和这条驱动路径上：

- `VK_KHR_cooperative_matrix` 不是只停留在 capability 枚举
- ISA 里真的出现了 `v_wmma_f32_16x16x16_f16`

### Scalar -> Cooperative compare

```text
resource_deltas:
  vgpr_count: 12 -> 36

runtime_deltas:
  instructions: 843 -> 32
  avg_stall_per_inst: 6.77 -> 4.06
  stall_share_of_duration: 0.72 -> 0.28
  occupancy_average_active: 8.95 -> 0.61

runtime_proxy_deltas:
  sync_wait_share: 0.56 -> 0.23
  sync_wait_cycles_per_inst: 6.91 -> 4.28
  immed_stall_per_inst: 50.06 -> 18.57

hotspot_profile_deltas:
  avg_duration_per_hit: 9.37 -> 14.53
  avg_stall_per_hit: 6.77 -> 4.06
```

这说明 cooperative path 的信号和 scalar path 很不一样：

- 指令条数显著减少
- `VGPR` 上升
- `sync_wait` 和 `IMMED` 压力下降
- 但 `occupancy_average_active` 也明显降低

所以这章的重点不是“cooperative 一定更快”，而是：

- 你已经能从 Vulkan shader 一路走到真正的 `WMMA` 风格 ISA
- 同时也能看到它带来的资源和 runtime 形态变化

## 这一章应该看什么

### scalar -> cooperative

重点看：

- host 输出里的
  - `cooperative_matrix_extension`
  - `cooperative_matrix_feature`
  - `coopmat_property[*]`
- `resource_summary`
- `code_object_isa`
- `source_isa_blocks`

如果 cooperative 版本真的走到了矩阵路径，你更可能看到：

- host 端列出 `16x16x16 f16->f32 subgroup`
- ISA 出现 `wmma` 或 `mfma` 风格的指令名
- `source_isa_blocks` 里出现 `cooperative_matrix` 相关块

这章已经实证到了这一步：

- host 端列出了 `16x16x16 f16->f32 subgroup`
- ISA 里出现了 `v_wmma_f32_16x16x16_f16`
- `source_isa_blocks` 已经把这段 source 和 ISA 对上了

这章的意义，就是让学习者先知道：

- `VK_KHR_cooperative_matrix` 在这台卡上不是头文件假象，而是实际设备能力
- 但 capability、shader source、ISA、runtime 仍然要分层看
