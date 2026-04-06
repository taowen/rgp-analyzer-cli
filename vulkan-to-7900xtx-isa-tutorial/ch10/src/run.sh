#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${script_dir}/build/ch10_ggml_isa"
repeats="${1:-16}"

bash "${script_dir}/build.sh" >/dev/null
env GGML_VK_DISABLE_FUSION=1 GGML_VK_PROFILE_NODES_PER_SUBMIT=1 "${binary}" "${repeats}"
