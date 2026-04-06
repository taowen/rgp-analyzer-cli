# Ch10: ggml Graph To ISA

这一章把前面“自己写 Vulkan shader”的路线，切到**官方 `ggml` Vulkan backend**。

目标是把这条链真正走通：

1. 用官方 `ggml` API 构造一个最小 `ggml_mul_mat` 图
2. 明确让它跑到官方 Vulkan backend
3. 抓 `.rgp`
4. 用 `rgp-analyzer-cli` 看到：
   - `ggml` 图节点
   - Vulkan code object
   - 官方 shader 源码
   - 对应的 AMDGPU ISA

这章不是自己写一个“像 ggml 的 matmul”，而是直接用官方仓库：

- official repo: [third_party/ggml](/home/taowen/projects/rgp-analyzer-cli/third_party/ggml)
- current HEAD: `49f84a9`

## What This Chapter Runs

`src/main.cpp` 基于官方 [`examples/simple/simple-backend.cpp`](/home/taowen/projects/rgp-analyzer-cli/third_party/ggml/examples/simple/simple-backend.cpp) 改成了：

- 显式使用 `ggml_backend_vk_init(0)`
- 保留 CPU backend 作为 scheduler 辅助 backend
- 把矩阵尺寸调大到不会走 `MUL_MAT_VEC`
- 输出 graph/op/backend/device 信息

这一章故意选：

- `M = 64`
- `N = 16`
- `K = 64`

因为在官方 Vulkan backend 里，`GGML_OP_MUL_MAT` 当 `n <= 8` 时会退到 `MUL_MAT_VEC` 路径；这里把 `n` 设到 `16`，既能避开 `MUL_MAT_VEC`，又能把 `.rgp` capture 的开销控制在教程可接受范围。

## Relevant Official ggml Sources

- graph submit path:
  [`ggml_backend_vk_graph_compute`](/home/taowen/projects/rgp-analyzer-cli/third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp)
- submit batching knobs:
  [`nodes_per_submit = 100`](/home/taowen/projects/rgp-analyzer-cli/third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L14370)
- Vulkan backend name:
  [`GGML_VK_NAME "Vulkan"`](/home/taowen/projects/rgp-analyzer-cli/third_party/ggml/include/ggml-vulkan.h)
- expected shader for this chapter:
  [`mul_mm.comp`](/home/taowen/projects/rgp-analyzer-cli/third_party/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp)

## Directory Layout

```text
ch10/
  README.md
  captures/
  src/
    CMakeLists.txt
    main.cpp
    build.sh
    run.sh
    capture-rgp.sh
    analyze-latest.sh
```

## Build

```bash
cd /home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch10/src
bash ./build.sh
```

这个脚本会：

1. 用 CMake 打开 `third_party/ggml`
2. 开启 `GGML_VULKAN=ON`
3. 关闭 ggml tests/examples
4. 构建本章的 `ch10_ggml_isa`

## Run

```bash
bash ./run.sh 16
```

一组实际输出：

```text
ggml_backend=Vulkan0
ggml_device_desc[0]=AMD Radeon RX 7900 XTX (RADV NAVI31)
graph_nodes=1
node0=MUL_MAT name=mul_mat_result result_ne=[64,16,1,1]
repeat_count=2
checksum=-5.25061
sample_out=1.58722,-2.49728,6.69595,-7.35703
```

## Capture

```bash
bash ./capture-rgp.sh 8
```

这一步会额外打开：

- `GGML_VK_DISABLE_FUSION=1`
- `GGML_VK_PROFILE_NODES_PER_SUBMIT=1`
- `MESA_VK_TRACE=rgp`
- `MESA_VK_TRACE_PER_SUBMIT=1`

本章只有一个主节点，但这组开关能把 `ggml graph -> Vulkan submit -> .rgp` 的链路压得更清楚。

capture 会保存到：

- [captures/latest.rgp](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch10/captures/latest.rgp)

这一章的 `ggml` workload 在 `.rgp` 模式下收尾比较慢，所以 `capture-rgp.sh` 不是等程序自然退出，而是：

1. 后台启动 `ch10_ggml_isa`
2. 轮询 `/tmp` 里的新 `.rgp`
3. 一旦拿到新 capture 就复制并结束进程

这样更适合 `ggml` 这类长生命周期 backend workload。

## Analyze

```bash
bash ./analyze-latest.sh
```

它会依次跑：

1. `resource-summary`
2. `shader-triage`
3. `shader-focus`
4. `code-object-isa`

这里 `code-object-isa` 会显式把源码文件指向官方：

- [`mul_mm.comp`](/home/taowen/projects/rgp-analyzer-cli/third_party/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp)

所以这一章的重点不是“猜这个 ISA 来自哪”，而是：

- `ggml_mul_mat`
- 官方 Vulkan backend
- 官方 `mul_mm.comp`
- 同一份 `.rgp` 里的 focused code object / ISA

## Actual Results

### resource-summary

```text
resource_summary:
  code_object[0] entry_point=_amdgpu_cs_main vgpr=84 sgpr=128 lds=6144 scratch=0
```

### shader-triage

```text
shader_triage:
  resource: entry_point=_amdgpu_cs_main vgpr=84 sgpr=128 lds=6144 scratch=0 wavefront=64
  trace_quality: level=resource_only sqtt_bytes=9280 queue_events=0 instructions=0 waves=0 dispatch_spans=0 mapped_dispatch=0/0
```

这一章当前拿到的是：

- 资源级信息是完整的
- code object / ISA 是完整的
- runtime thread-trace 仍然是 `resource_only`

这说明对这种非常小、只有一个主图节点的官方 `ggml` workload，当前 `.rgp` 里更容易稳定拿到的是：

- `ggml op -> code object`
- `code object -> 官方 shader`
- `shader -> ISA`

而不是 dense 的 runtime dispatch 证据。

### code-object-isa

这一章最重要的不是 runtime summary，而是 `code-object-isa` 已经能把官方 `mul_mm.comp` 的块和 ISA 串起来。例如：

```text
source_isa_blocks:
  - label=value_accumulate
    source line=255 match=coopmat<ACC_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[cms_per_row * cms_per_col];
    isa pc=0x88 v_mul_f32_e32 ...
  - label=shared_exchange
    source line=128 match=shared FLOAT_TYPE_VEC2 buf_a[BM * SHMEM_STRIDE];
    source line=224 match=barrier();
    isa pc=0x5c s_waitcnt lgkmcnt(0) ...
  - label=output_store
    source line=404 match=coopMatStore(cm_dtype, data_d, offsets + (dc + cm_col * TN) * p.stride_d + dr + cm_row * TM, p.stride_d, gl_CooperativeMatrixLayoutColumnMajor);
    isa pc=0x5c s_waitcnt lgkmcnt(0) ...
```

## What To Look For

这一章建议重点看这几层。

### 1. Host / ggml layer

从 `run.sh` 的输出看：

- backend 是否真的是 `Vulkan0`
- device description 是否是 `RX 7900 XTX`
- graph 是否真的是 `MUL_MAT`

### 2. resource-summary

看 focused code object 的：

- `vgpr`
- `sgpr`
- `lds`

### 3. shader-triage / shader-focus

看：

- `trace_quality`
- `dispatch_spans`
- `avg_stall`
- `occ_avg`

### 4. code-object-isa

看 `mul_mm.comp` 的这些块和 ISA 是不是能对应起来：

- shared tile load
- matmul core
- output store

## Why This Chapter Matters

前几章主要是：

- 你写的 Vulkan shader
- 然后看它的 ISA

这一章变成：

- 框架生成图
- backend 选择 shader
- 驱动生成 ISA

也就是开始从“自己写 kernel”过渡到“读框架 kernel”。这正是后面看 `ggml-vulkan`、`flash_attn`、`sage attention` 这类真实 workload 的前提。
