#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

bash "${script_dir}/build.sh"

prompt="${*:-a red robot reading a book in a library}"
mapfile -t generation_args < <(build_generation_args)
mapfile -t load_args < <(build_load_args)

mkdir -p "${output_dir}"
rm -rf "${phase_dir}"
mkdir -p "${phase_dir}"
cd "${example_root}"
apply_zimage_runtime_env
exec "${app_binary}" "${load_args[@]}" "${generation_args[@]}" "${prompt}"
