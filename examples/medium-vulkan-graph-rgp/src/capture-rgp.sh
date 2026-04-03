#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
capture_dir="$(cd "${script_dir}/.." && pwd)/captures"
mkdir -p "${capture_dir}"

before_file="$(mktemp)"
after_file="$(mktemp)"
before_state_file="$(mktemp)"
watch_dir="$(mktemp -d /tmp/medium-rgp-watch.XXXXXX)"
watch_state="$(mktemp /tmp/medium-rgp-watch-state.XXXXXX)"
watcher_pid=""
trap 'rm -f "${before_file}" "${after_file}" "${before_state_file}" "${watch_state}"; rm -rf "${watch_dir}"' EXIT

trace_buffer_bytes="${RADV_THREAD_TRACE_BUFFER_SIZE:-67108864}"

snapshot_rgp_captures() {
    local seq=0
    while [[ -f "${watch_state}" ]]; do
        for source_path in /tmp/medium_graph_compute_*.rgp; do
            [[ -f "${source_path}" ]] || continue
            local source_name
            source_name="$(basename "${source_path}")"
            local size_bytes
            size_bytes="$(stat -c %s "${source_path}" 2>/dev/null || echo 0)"
            local mtime_ns
            mtime_ns="$(python3 - <<'PY' "${source_path}" 2>/dev/null
import os, sys
print(os.stat(sys.argv[1]).st_mtime_ns)
PY
)"
            local key="${source_name}|${size_bytes}|${mtime_ns}"
            if grep -Fqx "${key}" "${watch_state}" 2>/dev/null; then
                continue
            fi
            printf '%s\n' "${key}" >> "${watch_state}"
            local snapshot_name="${source_name%.rgp}__snap$(printf '%04d' "${seq}").rgp"
            cp -p "${source_path}" "${watch_dir}/${snapshot_name}" 2>/dev/null || true
            seq=$((seq + 1))
        done
        sleep 0.05
    done
}

find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${before_file}"
cp "${before_file}" "${before_state_file}"
touch "${watch_state}"
snapshot_rgp_captures &
watcher_pid=$!

MANGOHUD=0 MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 RADV_THREAD_TRACE_BUFFER_SIZE="${trace_buffer_bytes}" \
    bash "${script_dir}/run.sh" "$@"

rm -f "${watch_state}"
if [[ -n "${watcher_pid}" ]]; then
    wait "${watcher_pid}" 2>/dev/null || true
fi

find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort > "${after_file}"
mapfile -t new_files < <(comm -13 "${before_file}" "${after_file}")
mapfile -t watched_snapshots < <(find "${watch_dir}" -maxdepth 1 -type f -name '*.rgp' -printf '%f\n' | sort)

largest_snapshot=""
largest_snapshot_size=0
for snapshot_name in "${watched_snapshots[@]}"; do
    base_name="${snapshot_name%%__snap*}.rgp"
    if grep -Fqx "${base_name}" "${before_state_file}" 2>/dev/null; then
        continue
    fi
    snapshot_path="${watch_dir}/${snapshot_name}"
    [[ -f "${snapshot_path}" ]] || continue
    size_bytes="$(stat -c %s "${snapshot_path}")"
    cp -p "${snapshot_path}" "${capture_dir}/"
    if (( size_bytes > largest_snapshot_size )); then
        largest_snapshot_size="${size_bytes}"
        largest_snapshot="${snapshot_path}"
    fi
done

for name in "${new_files[@]}"; do
    source_path="/tmp/${name}"
    [[ -f "${source_path}" ]] || continue
    size_bytes="$(stat -c %s "${source_path}")"
    cp -p "${source_path}" "${capture_dir}/"
    if (( size_bytes > largest_snapshot_size )); then
        largest_snapshot_size="${size_bytes}"
        largest_snapshot="${source_path}"
    fi
done

if [[ -z "${largest_snapshot}" || ! -f "${largest_snapshot}" ]]; then
    echo "no new medium_graph_compute .rgp capture was produced" >&2
    exit 1
fi

cp -p "${largest_snapshot}" "${capture_dir}/latest.rgp"
echo "captured ${#new_files[@]} trace file(s)"
echo "selected largest capture: ${largest_snapshot} (${largest_snapshot_size} bytes)"
echo "copied to ${capture_dir}/latest.rgp"
echo "trace buffer bytes: ${trace_buffer_bytes}"
