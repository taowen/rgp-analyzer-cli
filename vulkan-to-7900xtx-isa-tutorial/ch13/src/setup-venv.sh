#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

bash "${repo_root}/vulkan-to-7900xtx-isa-tutorial/ch12/src/setup-venv.sh"
