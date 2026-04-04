#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${script_dir}/build/ch02_vulkan_compare"
shader="${script_dir}/build/shaders/reg_pressure.comp.spv"
repeats="${1:-128}"

bash "${script_dir}/build.sh"
"${binary}" "${shader}" reg_pressure "${repeats}"
