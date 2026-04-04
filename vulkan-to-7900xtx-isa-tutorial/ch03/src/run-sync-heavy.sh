#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${script_dir}/build/ch03_vulkan_bottlenecks"
shader="${script_dir}/build/shaders/sync_heavy.comp.spv"
repeats="${1:-128}"
bash "${script_dir}/build.sh"
"${binary}" "${shader}" sync_heavy "${repeats}"
