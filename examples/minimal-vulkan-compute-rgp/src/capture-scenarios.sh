#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
capture_dir="$(cd "${script_dir}/.." && pwd)/captures"

mkdir -p "${capture_dir}"

run_capture() {
    local scenario="$1"
    shift
    bash "${script_dir}/capture-rgp.sh" "$@"
    cp "${capture_dir}/latest.rgp" "${capture_dir}/${scenario}.rgp"
    echo "saved scenario capture: ${capture_dir}/${scenario}.rgp"
}

bash "${script_dir}/compile-shaders.sh" >/dev/null
bash "${script_dir}/build.sh" >/dev/null

run_capture \
    single-baseline \
    baseline \
    --mode single \
    --dispatches 8 \
    --warmup 1 \
    --iterations 8

run_capture \
    multi-pipeline \
    baseline \
    --mode multi-pipeline \
    --shader-secondary "${script_dir}/build/shaders/lds_mix.spv" \
    --dispatches 8 \
    --warmup 1 \
    --iterations 8

run_capture \
    multi-cmdbuf \
    baseline \
    --mode multi-cmdbuf \
    --command-buffers 3 \
    --dispatches 9 \
    --warmup 1 \
    --iterations 8

run_capture \
    barrier-mix \
    lds_mix \
    --mode barrier-mix \
    --barrier-every 1 \
    --dispatches 6 \
    --warmup 1 \
    --iterations 8
