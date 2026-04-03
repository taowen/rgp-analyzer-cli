#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

prompt="${*:-a red robot reading a book in a library}"
repro_dir="${capture_dir}/repro"
mkdir -p "${repro_dir}"

echo "[1/4] default diffusion-only capture"
env \
    ZIMAGE_PROFILE_MODE=default \
    ZIMAGE_STOP_AFTER_PHASE=diffusion \
    ZIMAGE_STEPS="${ZIMAGE_STEPS:-2}" \
    ZIMAGE_WIDTH="${ZIMAGE_WIDTH:-256}" \
    ZIMAGE_HEIGHT="${ZIMAGE_HEIGHT:-256}" \
    ZIMAGE_REPEAT=1 \
    bash "${script_dir}/capture-rgp.sh" "${prompt}"
cp -p "${capture_dir}/latest.rgp" "${repro_dir}/default-diffusion.rgp"
cp -p "${capture_dir}/last-capture-manifest.txt" "${repro_dir}/default-diffusion.manifest.txt"
echo

echo "[2/4] minimal profiling capture"
bash "${script_dir}/capture-diffusion-profile.sh" "${prompt}"
cp -p "${capture_dir}/latest.rgp" "${repro_dir}/profile-diffusion.rgp"
cp -p "${capture_dir}/last-capture-manifest.txt" "${repro_dir}/profile-diffusion.manifest.txt"
echo

echo "[3/4] trace quality comparison"
python3 "${script_dir}/compare-trace-quality.py" \
    "${repro_dir}/default-diffusion.rgp" \
    "${repro_dir}/profile-diffusion.rgp"
echo

echo "[4/4] fixed workload evidence"
tmp_json="$(mktemp -t zimage-repro-triage-XXXXXX.json)"
trap 'rm -f "${tmp_json}"' EXIT
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli shader-triage \
    "${repro_dir}/profile-diffusion.rgp" \
    --build-helper \
    --json > "${tmp_json}"
python3 "${script_dir}/summarize-workload-evidence.py" \
    "${tmp_json}" \
    "${repro_dir}/profile-diffusion.manifest.txt"
