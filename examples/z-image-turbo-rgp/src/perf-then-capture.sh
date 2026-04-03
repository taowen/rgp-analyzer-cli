#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prompt="${*:-a red robot reading a book in a library}"

echo "[1/2] diffusion-only ggml Vulkan perf"
bash "${script_dir}/run-phase-perf.sh" diffusion "${prompt}"
echo
echo "[2/2] diffusion-only RGP profile capture"
bash "${script_dir}/capture-diffusion-profile.sh" "${prompt}"
