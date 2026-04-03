#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

capture="${capture_dir}/latest.rgp"
latest_perf_log="$(ls -1t "${example_root}"/perf-logs/*.log 2>/dev/null | head -n 1 || true)"
capture_manifest="${capture_dir}/last-capture-manifest.txt"
triage_json="$(mktemp -t z-image-turbo-triage-XXXXXX.json)"
perf_json=""

cleanup() {
    rm -f "${triage_json}"
    if [[ -n "${perf_json}" ]]; then
        rm -f "${perf_json}"
    fi
}
trap cleanup EXIT

if [[ ! -f "${capture}" ]]; then
    echo "missing capture: ${capture}" >&2
    exit 1
fi

PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli shader-triage "${capture}" --build-helper --json > "${triage_json}"

if [[ -n "${latest_perf_log}" && -f "${latest_perf_log}" ]]; then
    echo "latest_perf_log: ${latest_perf_log}"
    python3 "${script_dir}/summarize-vk-timings.py" "${latest_perf_log}" || true
    perf_json="$(mktemp -t z-image-turbo-perf-XXXXXX.json)"
    python3 "${script_dir}/summarize-vk-timings.py" "${latest_perf_log}" --json > "${perf_json}" || true
    echo
fi

if [[ -f "${capture_manifest}" ]]; then
    echo "last_capture_manifest: ${capture_manifest}"
    sed -n '1,40p' "${capture_manifest}"
    echo
    python3 "${script_dir}/correlate-captures-to-phases.py" "${capture_dir}" || true
    echo
    python3 "${script_dir}/rank-captures.py" "${capture_manifest}" || true
    echo
fi

if [[ -n "${perf_json}" && -f "${perf_json}" ]]; then
    python3 "${script_dir}/summarize-workload-evidence.py" "${triage_json}" "${perf_json}" "${capture_manifest}"
else
    if [[ -f "${capture_manifest}" ]]; then
        python3 "${script_dir}/summarize-workload-evidence.py" "${triage_json}" "${capture_manifest}"
    else
        python3 "${script_dir}/summarize-workload-evidence.py" "${triage_json}"
    fi
fi
echo

PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli inspect "${capture}"
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli resource-summary "${capture}"
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli stitch-report "${capture}"
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli decode-sqtt "${capture}" --build-helper
echo
PYTHONPATH="${repo_root}/src" python3 -m rgp_analyzer_cli shader-triage "${capture}" --build-helper
