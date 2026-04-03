#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bash "${script_dir}/sync-weights.sh"
cd "${app_dir}"
exec make -j"${JOBS:-4}" LLAMA_PORTABLE=1 LLAMA_VULKAN=1 NO_VULKAN_EXTENSIONS=1 direct-txt2img
