# Ch01: Minimal Vulkan Compute To ISA

这一章做一件很具体的事：

写一个最小 Vulkan compute 程序，把它跑在 `RX 7900 XTX` 上，抓一份 `.rgp`，然后用 `rgp-analyzer-cli` 看这条 shader 最终落成了什么资源和 ISA 证据。

## 本章目标

- 搭起 Linux + RADV 的最小 Vulkan compute 开发环境
- 理解 `Vulkan -> SPIR-V -> AMDGPU ISA -> .rgp` 的对应关系
- 运行一个自包含的 compute shader 实验
- 抓一份 `.rgp`
- 用 `rgp-analyzer-cli` 看：
  - `resource-summary`
  - `shader-triage`
  - `shader-focus`
  - `code-object-isa`

## Vulkan 和 ISA 的对应关系

这一章里，代码路径是：

1. 你写的是 GLSL compute shader
2. `glslc` 把 GLSL 编成 `SPIR-V`
3. Vulkan 程序在运行时创建 compute pipeline，并把 `SPIR-V` 交给驱动
4. RADV 把 `SPIR-V` 编译成 AMDGPU code object
5. `.rgp` 里会记录：
   - code object
   - resource metadata
   - runtime SQTT / dispatch 证据
6. `rgp-analyzer-cli` 再把这些信息拆出来，给你看：
   - `VGPR / SGPR / LDS / scratch`
   - dispatch / hotspot
   - top PCs / static ISA 文本

也就是说，这一章不是只让程序“跑起来”，而是把“shader 最终在 7900 XTX 上变成什么”这件事也串起来。

这里的 `code-object-isa` 不是脱离 capture 的孤立反汇编。
它会固定在同一份 `.rgp` 选出来的 focused `code_object` 上，同时显示：

- 当前 capture 的 focused shader
- 当前 capture 的 `top_pcs`，如果有
- 同一个 focused shader 的静态 ISA 反汇编
- 同一个源码文件的 `source_hints`
- 一组稳定的 `source_isa_blocks`

这不是调试信息级逐行精确映射，但至少保证“源码、ISA、capture 关注对象”是同一条线。

为了让这一章更适合教学，shader 故意写成了几个清楚的步骤：

- `idx = gl_GlobalInvocationID.x`
- `if (idx >= pc.element_count) return`
- `scaled = idx * pc.multiplier`
- `value = scaled + pc.bias`
- `out_buf.data[idx] = value`

这样 `code-object-isa` 就可以把这些源码块和最相关的 ISA 片段并排展示。

## 目录结构

```text
vulkan-to-7900xtx-isa-tutorial/ch01/
  README.md
  src/
    main.cpp
    shaders/
      fill_buffer.comp
    compile-shaders.sh
    build.sh
    run.sh
    capture-rgp.sh
    analyze-latest.sh
```

## 依赖

- Linux
- AMD GPU
- RADV
- `glslc`
- `c++`
- Vulkan loader 和头文件
- 邻居仓库里的 `rgp-analyzer-cli`

常见包名：

- Ubuntu/Debian:
  - `build-essential`
  - `glslang-tools`
  - `libvulkan-dev`
  - `vulkan-tools`

## Build

```bash
cd ~/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch01/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
bash ./run.sh 128
```

如果一切正常，你会看到：

- 选择到一张 Vulkan 物理设备
- `dispatch_ok`
- 一个稳定的 checksum

这一章当前在 `RX 7900 XTX (RADV NAVI31)` 上的一次真实输出是：

```text
device=AMD Radeon RX 7900 XTX (RADV NAVI31) vendor=0x1002 device=0x744c queue_family=0
dispatch_ok element_count=256 workgroups=4 repeats=128 checksum=99712
```

默认会重复 dispatch 多次，这样 `.rgp` 里的 runtime 证据更容易稳定出现。

## Capture `.rgp`

```bash
bash ./capture-rgp.sh 128
```

脚本会：

- 用 `MESA_VK_TRACE=rgp`
- 用 `MESA_VK_TRACE_PER_SUBMIT=1`
- 从 `/tmp` 里挑出本轮新生成的 `.rgp`
- 复制到：

```text
rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch01/captures/latest.rgp
```

## Analyze With `rgp-analyzer-cli`

```bash
bash ./analyze-latest.sh
```

这个脚本会自动尝试定位邻居仓库里的 `rgp-analyzer-cli`，然后跑：

- `resource-summary`
- `shader-triage`
- `shader-focus`
- `code-object-isa`

当前这章在同一份 `.rgp` 上的一次真实分析结果里，可以直接看到：

```text
shader_triage:
  resource: entry_point=_amdgpu_cs_main vgpr=12 sgpr=128 lds=0 scratch=0 wavefront=64
  runtime: instructions=26 ... avg_stall=1.62 ... occ_avg=1.34 occ_max=6
  trace_quality: level=dispatch_isa ... dispatch_spans=768 mapped_dispatch=36/128

shader_focus:
  focus_code_object: 0
  resource: entry_point=_amdgpu_cs_main vgpr=12 sgpr=128 lds=0 scratch=0 wavefront=64
  runtime(global): instructions=26 ... avg_stall=1.62 ... occ_avg=1.34 occ_max=6
  source_hints: ... available=True match_count=0

code_object_isa:
  focus_code_object: 0
  entry_point: _amdgpu_cs_main
  source_isa_blocks:
    - label=invocation_index
      source line=16 match=uint idx = gl_GlobalInvocationID.x;
      isa pc=0x0 v_lshl_add_u32 ...
    - label=bounds_check
      source line=17 match=if (idx >= pc.element_count) {
      source line=18 match=return;
      isa pc=0xc v_cmpx_gt_u32_e32 ...
      isa pc=0x10 s_cbranch_execz ...
    - label=value_compute
      source line=21 match=uint scaled = idx * pc.multiplier;
      source line=22 match=uint value = scaled + pc.bias;
      isa pc=0x20 v_mul_lo_u32 ...
      isa pc=0x30 v_add_nc_u32_e32 ...
    - label=buffer_store
      source line=23 match=out_buf.data[idx] = value;
      isa pc=0x34 s_waitcnt lgkmcnt(0) ...
      isa pc=0x38 buffer_store_b32 ...
  isa:
    - pc=0x0 v_lshl_add_u32 ...
    - pc=0x34 s_waitcnt lgkmcnt(0) ...
    - pc=0x38 buffer_store_b32 ...
```

如果你想手工跑：

```bash
cd ~/projects/rgp-analyzer-cli
PYTHONPATH=src python3 -m rgp_analyzer_cli resource-summary \
  ~/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch01/captures/latest.rgp
```

## 这一章实验的意义

这不是性能 benchmark。

它的意义是：

- 证明你能在本机上稳定跑 Vulkan compute
- 证明你能抓 `.rgp`
- 证明你能把一份最小 shader 从 Vulkan 一路追到 AMDGPU ISA 证据

## 这一章当前“源码 -> ISA”能做到什么

这一章现在已经能做到：

- `fill_buffer.comp` 通过 Vulkan 进入同一份 `.rgp`
- `code-object-isa` 从这份 `.rgp` 选出 focused `code_object`
- 输出这个 focused shader 的静态 ISA
- 把 `invocation_index / bounds_check / value_compute / buffer_store` 这些源码块和对应 ISA 段放在一起看

这一章现在**还不能**做到：

- 把 GLSL 源码逐行精确映射到每条 ISA
- 对每条 ISA 都给出唯一源码行
- 保证源码块和 ISA 是一一对应关系

所以如果你看到：

```text
source_isa_blocks:
  - label=bounds_check
    source line=17 match=if (idx >= pc.element_count) {
    isa pc=0xc v_cmpx_gt_u32_e32 ...
    isa pc=0x10 s_cbranch_execz ...
```

这表示：

- capture 和 focused shader 关联是通的
- 静态 ISA 也已经拿到了
- 工具已经能把稳定的源码块和一组 ISA 片段放到一起看
- 但这仍然不是调试信息级的逐行精确映射

这一步通了，后面才适合继续做更复杂的 shader/ISA 调优章节。
