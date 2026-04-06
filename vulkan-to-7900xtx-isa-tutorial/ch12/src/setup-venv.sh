#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chapter_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
venv_dir="${chapter_dir}/.venv"
wheel_dir="${chapter_dir}/wheels"

mkdir -p "${wheel_dir}"

python3 -m venv "${venv_dir}"
source "${venv_dir}/bin/activate"

if python - <<'PY' >/dev/null 2>&1
import importlib.util
import torch
assert torch.cuda.is_available()
assert importlib.util.find_spec("omnivoice") is not None
PY
then
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_name", torch.cuda.get_device_name(0))
print("hip_version", getattr(torch.version, "hip", None))
print("omnivoice_ready", True)
PY
  exit 0
fi

python -m pip install --upgrade pip wheel setuptools
python -m pip install numpy==1.26.4

base_url="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
torch_whl="torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
torchvision_whl="torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
torchaudio_whl="torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
triton_whl="triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

download_wheel() {
  local whl="$1"
  if [[ ! -f "${wheel_dir}/${whl}" ]]; then
    wget -O "${wheel_dir}/${whl}" "${base_url}/${whl}"
  fi
}

download_wheel "${torch_whl}"
download_wheel "${torchvision_whl}"
download_wheel "${torchaudio_whl}"
download_wheel "${triton_whl}"

python -m pip install \
  "${wheel_dir}/${torch_whl}" \
  "${wheel_dir}/${torchvision_whl}" \
  "${wheel_dir}/${torchaudio_whl}" \
  "${wheel_dir}/${triton_whl}"

python -m pip install \
  "transformers==5.3.0" \
  accelerate \
  pydub \
  gradio \
  tensorboardX \
  webdataset \
  soundfile

python -m pip install -e "${repo_root}/third_party/OmniVoice" --no-deps

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
    print("hip_version", getattr(torch.version, "hip", None))
PY
