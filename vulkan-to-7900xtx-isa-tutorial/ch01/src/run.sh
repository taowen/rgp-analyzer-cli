#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${script_dir}/build/ch01_vulkan_compute"
shader="${script_dir}/build/shaders/fill_buffer.comp.spv"
repeats="${1:-128}"

bash "${script_dir}/build.sh"
"${binary}" "${shader}" "${repeats}"
