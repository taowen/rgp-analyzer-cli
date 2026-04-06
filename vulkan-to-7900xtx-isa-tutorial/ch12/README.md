# Ch12: OmniVoice On PyTorch ROCm

这一章选用的 TTS 模型是：

- **OmniVoice**

目标是两件事：

1. 在 `RX 7900 XTX + ROCm + PyTorch` 上把官方 OmniVoice 跑起来
2. 生成一段英语语音，再用 `Whisper tiny.en` 做 ASR 回验

## Ch12 Goal

这一章不走 `ggml`，而是走官方 `PyTorch` 路线。

我们要得到的是一条真实可复现工作流：

1. 建独立 `venv`
2. 安装 AMD 官方推荐的 `PyTorch ROCm` wheel
3. 安装官方 `OmniVoice`
4. 在 AMD GPU 上生成一段英文语音
5. 用 `whisper.cpp` 把生成音频转回文本
6. 比较目标文本和 ASR 文本

## Official Upstream

- official repo: [third_party/OmniVoice](/home/taowen/projects/rgp-analyzer-cli/third_party/OmniVoice)
- official model: <https://huggingface.co/k2-fsa/OmniVoice>
- official paper: <https://arxiv.org/abs/2604.00688>

## Files

- setup ROCm venv:
  [setup-venv.sh](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/src/setup-venv.sh)
- generate audio:
  [run-omnivoice.sh](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/src/run-omnivoice.sh)
- verify with ASR:
  [verify-asr.sh](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/src/verify-asr.sh)
- one-shot pipeline:
  [run-and-verify.sh](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/src/run-and-verify.sh)
- inference driver:
  [infer_omnivoice.py](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/src/infer_omnivoice.py)
- ASR comparer:
  [compare_asr.py](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/src/compare_asr.py)

## Setup

```bash
cd /home/taowen/projects/rgp-analyzer-cli
bash ./vulkan-to-7900xtx-isa-tutorial/ch12/src/setup-venv.sh
```

这会做：

- 创建 `ch12/.venv`
- 安装 AMD 官方 ROCm 7.2.1 的 `torch/torchvision/torchaudio/triton`
- 安装 `OmniVoice`

## Run

```bash
cd /home/taowen/projects/rgp-analyzer-cli
bash ./vulkan-to-7900xtx-isa-tutorial/ch12/src/run-omnivoice.sh
```

默认文本：

```text
This is a short speech synthesis test running on an AMD Radeon graphics card.
```

输出文件：

- [generated.wav](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/output/generated.wav)
- [inference.json](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/output/inference.json)

## Verify With ASR

```bash
cd /home/taowen/projects/rgp-analyzer-cli
bash ./vulkan-to-7900xtx-isa-tutorial/ch12/src/verify-asr.sh
```

这个脚本会：

- 调用 `whisper.cpp` 的 `whisper-cli`
- 用 `tiny.en` 转写 `generated.wav`
- 把目标文本和 ASR 文本都归一化
- 输出一个简单的 token-level 对比

输出文件：

- [asr.txt](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/output/asr.txt)
- [verification.json](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/output/verification.json)

## One-Shot

```bash
cd /home/taowen/projects/rgp-analyzer-cli
bash ./vulkan-to-7900xtx-isa-tutorial/ch12/src/run-and-verify.sh
```

## Actual Verification

我已经实际跑过：

```bash
cd /home/taowen/projects/rgp-analyzer-cli
bash ./vulkan-to-7900xtx-isa-tutorial/ch12/src/setup-venv.sh
bash ./vulkan-to-7900xtx-isa-tutorial/ch12/src/run-omnivoice.sh
bash ./vulkan-to-7900xtx-isa-tutorial/ch12/src/verify-asr.sh
```

### 1. ROCm PyTorch on AMD

实际输出：

```text
torch 2.9.1+rocm7.2.1.gitff65f5bc
cuda_available True
device_name AMD Radeon RX 7900 XTX
hip_version 7.2.53211-e1a6bc5663
```

也就是说：

- 官方 ROCm wheel 安装成功
- `torch.cuda.is_available()` 为真
- 设备正确识别到 `RX 7900 XTX`

### 2. OmniVoice generation

目标文本：

```text
This is a short speech synthesis test running on an AMD Radeon graphics card.
```

实际推理结果：

```json
{
  "device": "cuda:0",
  "torch_version": "2.9.1+rocm7.2.1.gitff65f5bc",
  "device_name": "AMD Radeon RX 7900 XTX",
  "sampling_rate": 24000,
  "load_ms": 4658.33,
  "generate_ms": 2720.39,
  "total_ms": 7378.72
}
```

生成音频：

- [generated.wav](/home/taowen/projects/rgp-analyzer-cli/vulkan-to-7900xtx-isa-tutorial/ch12/output/generated.wav) (`188K`)

### 3. ASR verification

我用 `whisper.cpp` 的 `tiny.en` 对生成音频做了回验。实际输出：

```text
This is a short speech synthesis test running on an AMD Radeon graphics card.
```

验证 JSON：

```json
{
  "target_text": "This is a short speech synthesis test running on an AMD Radeon graphics card.",
  "asr_text": "This is a short speech synthesis test running on an AMD Radeon graphics card.",
  "target_token_count": 14,
  "asr_token_count": 14,
  "matched_prefix_tokens": 14,
  "exact_match": true
}
```

也就是说，这条最小路径已经真实成立：

1. `OmniVoice`
2. `PyTorch ROCm`
3. `RX 7900 XTX`
4. 生成英语语音
5. 用 `Whisper tiny.en` 回验
6. 文本完全一致
