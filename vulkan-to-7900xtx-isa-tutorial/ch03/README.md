# Ch03: Memory, Sync, or Compute?

这一章回答一个更像真实调优的问题：

同样是一个 Vulkan compute kernel，当前瓶颈到底更像：

- global memory
- synchronization / LDS
- 还是普通算术

这一章不追求“更快”，而是追求“会读信号”。

## 本章目标

- 用同一个 Vulkan compute harness 跑 3 个 shader 变体
- 抓 3 份 `.rgp`
- 用 `rgp-analyzer-cli` 区分：
  - baseline
  - memory-heavy
  - sync-heavy

## 三个变体

- `baseline.comp`
  一个简单的 load + compute + store kernel

- `memory_heavy.comp`
  故意做更多 global buffer gather/load，让 `buffer_load*` 和 global-memory 相关信号更明显

- `sync_heavy.comp`
  故意引入 `shared` 和 `barrier()`，让 `WAIT / IMMED / LDS` 相关信号更明显

## 目录结构

```text
vulkan-to-7900xtx-isa-tutorial/ch03/
  README.md
  src/
    main.cpp
    shaders/
      baseline.comp
      memory_heavy.comp
      sync_heavy.comp
    compile-shaders.sh
    build.sh
    run-baseline.sh
    run-memory-heavy.sh
    run-sync-heavy.sh
    capture-baseline.sh
    capture-memory-heavy.sh
    capture-sync-heavy.sh
    analyze-baseline.sh
    analyze-memory-heavy.sh
    analyze-sync-heavy.sh
    compare-memory.sh
    compare-sync.sh
```

## Build

```bash
cd ~/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch03/src
bash ./compile-shaders.sh
bash ./build.sh
```

## Run

```bash
bash ./run-baseline.sh 128
bash ./run-memory-heavy.sh 128
bash ./run-sync-heavy.sh 128
```

## Capture

```bash
bash ./capture-baseline.sh 128
bash ./capture-memory-heavy.sh 128
bash ./capture-sync-heavy.sh 128
```

捕获保存到：

```text
vulkan-to-7900xtx-isa-tutorial/ch03/captures/baseline.rgp
vulkan-to-7900xtx-isa-tutorial/ch03/captures/memory_heavy.rgp
vulkan-to-7900xtx-isa-tutorial/ch03/captures/sync_heavy.rgp
```

## Analyze

单独分析：

```bash
bash ./analyze-baseline.sh
bash ./analyze-memory-heavy.sh
bash ./analyze-sync-heavy.sh
```

对比分析：

```bash
bash ./compare-memory.sh
bash ./compare-sync.sh
```

## 真实输出示例

这章已经在 `RX 7900 XTX (RADV NAVI31)` 上实际跑过。

三条 `run` 的一次真实输出分别是：

```text
dispatch_ok variant=baseline element_count=256 workgroups=4 repeats=64 checksum=1668736
dispatch_ok variant=memory_heavy element_count=256 workgroups=4 repeats=64 checksum=13284352
dispatch_ok variant=sync_heavy element_count=256 workgroups=4 repeats=64 checksum=1668864
```

两组 `compare` 的真实信号是：

### baseline -> memory_heavy

```text
vgpr_count: 12 -> 24
VMEM count: 2 -> 5
global_memory_duration_share: 0.018 -> 0.024
vector_duration_share: 0.057 -> 0.291
```

### baseline -> sync_heavy

```text
lds_size: 0 -> 512
LDS count: 0 -> 4
lds_duration_share: 0.000 -> 0.027
sync_wait_share: 0.213 -> 0.262
WAIT share also increases
```

单独分析时，`sync_heavy` 现在会直接显示：

```text
resource: ... vgpr=12 sgpr=128 lds=512 ...
runtime_proxies: ... lds_duration_share=0.03 ... sync_wait_share=0.26 ...
source_hints:
  - line=30 match=barrier();
  - line=33 match=barrier();
  - line=37 match=barrier();
```

## 这一章要观察什么

### memory-heavy

重点看：

- `global_memory_duration_share`
- `memory_instruction_share`
- `buffer_load*` 相关 ISA
- `source_isa_blocks` 里的 `global_load`

### sync-heavy

重点看：

- `sync_wait_share`
- `sync_wait_cycles`
- `immed_stall_per_inst`
- `lds_duration_share`
- `WAIT`
- `source_isa_blocks` 里的 `shared_exchange`

## 这一章应该教会你的东西

如果你看到：

- `VMEM/SMEM` 和 global-memory proxy 抬高  
  先怀疑访存模式

- `WAIT / IMMED / LDS` 抬高  
  先怀疑共享内存和同步结构

- `lds_size` 从 `0` 变成非零，而且 `source_hints` 直接落到 `barrier()`  
  更像显式同步和 shared-memory 路径在起作用

- 资源和事件结构都差不多，但 `VALU` 和算术 ISA 暴涨  
  更像算术写法本身变复杂了

这一步通了，后面才适合把这些判别方法用到更复杂的 shader 上。
