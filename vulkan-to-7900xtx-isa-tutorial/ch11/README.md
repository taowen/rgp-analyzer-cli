# Ch11: Whisper On ggml Vulkan

这一章选用的 ASR 模型是：

- **Whisper tiny.en**

原因：

- 官方 `whisper.cpp` 直接基于 `ggml`
- Vulkan backend 已经是正式支持路径
- `tiny.en` 足够小，适合先跑通、抓 `.rgp`、再做调优
- encoder / decoder / attention / matmul 都在，能自然承接 `ch10`

## Ch11 Goal

这一章分成两层：

1. 先把一个真实 `ggml` ASR 模型在 Vulkan 上跑通
2. 再把这条工作流收成：
   - 真实 ASR / bench 可复现
   - `.rgp` 可抓
   - `ggml` 官方 shader 的源码块和 ISA 能直接查看

## Official Upstream

- official repo: [third_party/whisper.cpp](/home/taowen/projects/rgp-analyzer-cli/third_party/whisper.cpp)
- current HEAD: `95ea8f9`

## Actual Verification

我已经实际完成了下面这几步：

### 1. Build whisper.cpp with Vulkan

```bash
cd /home/taowen/projects/rgp-analyzer-cli/third_party/whisper.cpp
cmake -B build-vk -DGGML_VULKAN=1 -DWHISPER_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j$(nproc) --config Release
```

构建产物已经就绪：

- `build-vk/bin/whisper-cli`
- `build-vk/bin/whisper-bench`

### 2. Download a ggml Whisper model

```bash
./models/download-ggml-model.sh tiny.en ./models
```

当前模型：

- [ggml-tiny.en.bin](/home/taowen/projects/rgp-analyzer-cli/third_party/whisper.cpp/models/ggml-tiny.en.bin)

### 3. Run a real ASR example

```bash
./build-vk/bin/whisper-cli \
  -m ./models/ggml-tiny.en.bin \
  -f ./samples/jfk.wav \
  -t 4
```

实际输出已经跑通：

```text
whisper_backend_init_gpu: using Vulkan0 backend
whisper_model_load: type          = 1 (tiny)
whisper_model_load:      Vulkan0 total size =    77.11 MB

main: processing './samples/jfk.wav' (176000 samples, 11.0 sec)

whisper_print_timings:      mel time =    11.34 ms
whisper_print_timings:   encode time =     9.62 ms
whisper_print_timings:   decode time =     2.39 ms
whisper_print_timings:   batchd time =    55.13 ms
whisper_print_timings:    total time =   196.21 ms

[00:00:00.000 --> 00:00:07.960]   And so my fellow Americans ask not what your country can do for you
[00:00:07.960 --> 00:00:10.760]   ask what you can do for your country.
```

也就是说：

- 模型是对的
- Vulkan backend 是真的在用
- ASR 结果是对的
- 这一章已经具备“真实模型 + 官方 ggml Vulkan 后端”的基础

## Bench And Capture

我还实际跑了 `whisper-bench`：

```bash
cd /home/taowen/projects/rgp-analyzer-cli
bash ./vulkan-to-7900xtx-isa-tutorial/ch11/src/run-bench.sh 4 0
```

真实输出：

```text
whisper_backend_init_gpu: using Vulkan0 backend
whisper_print_timings:   encode time =     5.20 ms
whisper_print_timings:   decode time =   204.18 ms
whisper_print_timings:   batchd time =    59.77 ms
whisper_print_timings:   prompt time =    43.79 ms
whisper_print_timings:    total time =   313.21 ms
```

然后抓了 `.rgp`：

```bash
cd /home/taowen/projects/rgp-analyzer-cli
bash ./vulkan-to-7900xtx-isa-tutorial/ch11/src/capture-bench.sh 4 0
bash ./vulkan-to-7900xtx-isa-tutorial/ch11/src/analyze-latest.sh
```

当前这条 `whisper` capture 的特点是：

- `GGML_VK_DISABLE_FUSION=1`
- `GGML_VK_PROFILE_NODES_PER_SUBMIT=1`
- `MESA_VK_TRACE_PER_SUBMIT=1`

但这份 `whisper` capture 目前仍然是：

```text
trace_quality: level=resource_only
profiling_constraints: submit_dilution_suspected=True
```

所以这章当前的重点不是“已经拿到完整 runtime dispatch 证据”，而是：

- 已经拿到真实 `ggml` ASR workload 的 `.rgp`
- 已经能列出官方 `ggml-vulkan` code object 资源
- 已经能把候选 attention shader 的 ISA 和官方源码块对起来

## Official ggml Shader -> ISA

当前 `latest.rgp` 的 `resource-summary` 是：

```text
code_object[0] entry_point=_amdgpu_cs_main vgpr=12 sgpr=128 lds=0 scratch=0
code_object[1] entry_point=_amdgpu_cs_main vgpr=36 sgpr=128 lds=0 scratch=0
code_object[2] entry_point=_amdgpu_cs_main vgpr=132 sgpr=128 lds=22528 scratch=0
code_object[3] entry_point=_amdgpu_cs_main vgpr=48 sgpr=128 lds=0 scratch=0
code_object[4] entry_point=_amdgpu_cs_main vgpr=12 sgpr=128 lds=0 scratch=0
code_object[5] entry_point=_amdgpu_cs_main vgpr=144 sgpr=128 lds=22528 scratch=0
```

其中 `code_object[2]` 和 `code_object[5]` 是当前最像官方 `flash_attn.comp` 的候选：

```bash
cd /home/taowen/projects/rgp-analyzer-cli
bash ./vulkan-to-7900xtx-isa-tutorial/ch11/src/analyze-flash-attn-candidate-2.sh
bash ./vulkan-to-7900xtx-isa-tutorial/ch11/src/analyze-flash-attn-candidate-5.sh
```

对应的源码文件是：

- [third_party/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp](/home/taowen/projects/rgp-analyzer-cli/third_party/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp)

当前已经能直接看到的源码块/ISA 对应包括：

- `softmax_update`
  - `line=167 match=max_mask = max(max_mask, float(m));`
  - `line=181 match=max_mask = max(max_mask, tmpsh[s]);`
  - `pc=0x60 s_waitcnt lgkmcnt(0)`

- `shared_exchange`
  - `line=38 match=shared float tmpsh[tmpsh_size];`
  - `line=45 match=shared FLOAT_TYPEV4 Qf[Br * qf_stride];`
  - `pc=0x60 s_waitcnt lgkmcnt(0)`

也就是说，这章现在已经能做到：

- 真实 Whisper tiny.en 在 Vulkan 上跑通
- 真实 `.rgp` 成功抓到
- 官方 `ggml` shader 的源码块和 ISA 能直接查看

下一步要继续推进的，才是把这条 `whisper` capture 从 `resource_only` 拉回 `dispatch_isa`。

## First Real Optimization

这一章我先做了一轮**不改模型代码**的真实优化，优先调的是：

- `threads`
- decoder 策略

原因是当前 `whisper-cli` 的总时间里，最重的不是 encoder，而是：

- `sample time`
- `batchd time`

所以先试了几组参数，当前最有价值的是：

1. baseline: `-t 4`
2. `-t 12 -bo 1 -bs 1 -nf -mc 16`
3. `-t 12 -bo 1 -bs 1 -nf -nfa`
4. `-t 16 -bo 1 -bs 1 -nf`

对应脚本：

- baseline:
  [run-asr.sh](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch11/src/run-asr.sh)
- optimized:
  [run-asr-optimized.sh](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch11/src/run-asr-optimized.sh)
- compare:
  [compare-asr.sh](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch11/src/compare-asr.sh)
- repeated benchmark:
  [benchmark-configs.sh](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch11/src/benchmark-configs.sh)

### Measured Results

baseline `-t 4`:

```text
whisper_print_timings:   sample time =    34.47 ms
whisper_print_timings:   encode time =     9.41 ms
whisper_print_timings:   decode time =     2.78 ms
whisper_print_timings:   batchd time =    51.90 ms
whisper_print_timings:    total time =   186.90 ms
```

optimized `-t 12 -bo 1 -bs 1 -nf -mc 16`:

```text
whisper_print_timings:   sample time =     7.78 ms
whisper_print_timings:   encode time =     5.68 ms
whisper_print_timings:   decode time =    20.07 ms
whisper_print_timings:   batchd time =     0.00 ms
whisper_print_timings:    total time =   104.69 ms
```

也就是：

- `total time: 186.90 -> 104.69 ms`
- 绝对减少：`82.21 ms`
- 相对减少：约 `44.0%`

### Why This Helps

这次优化的关键不是把 encoder 算得更快，而是：

- 用更多 CPU 线程把 mel / host-side 辅助开销压下去
- 把 decoder 从 `5 beams + best of 5` 收成更轻的 greedy 路径

这会让：

- `sample time` 明显下降
- `batchd` 路径基本消失
- decoder 上下文也更短

边界也已经实测过：

- `-nfa` 会变慢
  `104.69 -> 129.12 ms`
- `-t 16` 没有继续提升
  `104.69 -> 126.78 ms`

为了避免单次运行波动，这一章还提供：

```bash
cd /home/taowen/projects/rgp-analyzer-cli
bash ./vulkan-to-7900xtx-isa-tutorial/ch11/src/benchmark-configs.sh 5
```

它会对 baseline 和 optimized 各跑多次，并输出：

- `median`
- `min`
- `max`
- 排序后的全部结果

而这份 `jfk.wav` 的转写文本仍然保持正确：

```text
[00:00:00.000 --> 00:00:08.000]   And so my fellow Americans ask not what your country can do for you
[00:00:08.000 --> 00:00:11.000]   ask what you can do for your country.
```

所以 `ch11` 当前已经有一条真实、可复现、可写进教程的正向优化路径。
