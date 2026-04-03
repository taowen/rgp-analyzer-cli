#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${weights_dir}"

if have_required_weights "${weights_dir}"; then
    echo "weights already available at ${weights_dir}"
    ls -lh "${weights_dir}"
    exit 0
fi

games_root="$(bash "${script_dir}/mount-windows-games.sh")"

candidate_dirs=(
  "${games_root}/amd-vulkan-tutorial/shared/models/zimage"
  "${games_root}/koboldcpp/models/zimage"
)

source_dir=""
for candidate in "${candidate_dirs[@]}"; do
    if have_required_weights "${candidate}"; then
        source_dir="${candidate}"
        break
    fi
done

if [[ -z "${source_dir}" ]]; then
    echo "failed to locate z-image weights under ${games_root}" >&2
    exit 1
fi

cp -afv "${source_dir}/." "${weights_dir}/"
ls -lh "${weights_dir}"
