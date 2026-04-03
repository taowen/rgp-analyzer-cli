#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
example_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
app_dir="${example_root}/app"
app_binary="${app_dir}/direct-txt2img"
weights_dir="${example_root}/models/zimage"
capture_dir="${example_root}/captures"
output_dir="${example_root}/output"
phase_dir="${example_root}/phase-markers"
windows_mount_root="${WINDOWS_MOUNT_ROOT:-/mnt/win-c}"
windows_games_root="${WINDOWS_GAMES_ROOT:-${windows_mount_root}/games}"

zimage_steps="${ZIMAGE_STEPS:-8}"
zimage_width="${ZIMAGE_WIDTH:-1024}"
zimage_height="${ZIMAGE_HEIGHT:-1024}"
zimage_repeat="${ZIMAGE_REPEAT:-1}"
zimage_seed="${ZIMAGE_SEED:-123456}"
zimage_stop_after_phase="${ZIMAGE_STOP_AFTER_PHASE:-}"
zimage_flash_attn="${ZIMAGE_FLASH_ATTN:-0}"
zimage_vae_cpu="${ZIMAGE_VAE_CPU:-0}"
zimage_clip_cpu="${ZIMAGE_CLIP_CPU:-1}"
zimage_diffusion_conv_direct="${ZIMAGE_DIFFUSION_CONV_DIRECT:-0}"
zimage_vae_conv_direct="${ZIMAGE_VAE_CONV_DIRECT:-0}"
zimage_trace_buffer_bytes="${RADV_THREAD_TRACE_BUFFER_SIZE:-268435456}"
zimage_counter_buffer_bytes="${RADV_CACHE_COUNTERS_BUFFER_SIZE:-67108864}"
zimage_profile_nodes_per_submit="${GGML_VK_PROFILE_NODES_PER_SUBMIT:-8}"
zimage_profile_matmul_bytes_per_submit="${GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT:-8388608}"
zimage_force_split_k="${GGML_VK_FORCE_SPLIT_K:-}"
zimage_profile_mode="${ZIMAGE_PROFILE_MODE:-default}"

build_generation_args() {
    printf '%s\n' \
        --steps "${zimage_steps}" \
        --width "${zimage_width}" \
        --height "${zimage_height}" \
        --repeat "${zimage_repeat}" \
        --seed "${zimage_seed}"
    if [[ -n "${zimage_stop_after_phase}" ]]; then
        printf '%s\n' --stop-after-phase "${zimage_stop_after_phase}"
    fi
}

build_load_args() {
    if [[ "${zimage_flash_attn}" == "1" ]]; then
        printf '%s\n' --flash-attn
    fi
    if [[ "${zimage_vae_cpu}" == "1" ]]; then
        printf '%s\n' --vae-cpu
    else
        printf '%s\n' --no-vae-cpu
    fi
    if [[ "${zimage_clip_cpu}" == "1" ]]; then
        printf '%s\n' --clip-cpu
    else
        printf '%s\n' --no-clip-cpu
    fi
    if [[ "${zimage_diffusion_conv_direct}" == "1" ]]; then
        printf '%s\n' --diffusion-conv-direct
    fi
    if [[ "${zimage_vae_conv_direct}" == "1" ]]; then
        printf '%s\n' --vae-conv-direct
    fi
}

normalize_phase_name() {
    local phase="${1:-}"
    case "${phase}" in
        ""|full|all|none)
            printf '%s\n' ""
            ;;
        condition|diffusion)
            printf '%s\n' "${phase}"
            ;;
        *)
            echo "unsupported phase: ${phase}" >&2
            return 1
            ;;
    esac
}

apply_zimage_runtime_env() {
    export GGML_VK_DEBUG_MARKERS="${GGML_VK_DEBUG_MARKERS:-1}"
    export ZIMAGE_PHASE_MARKER_DIR="${phase_dir}"
    export RADV_THREAD_TRACE_BUFFER_SIZE="${RADV_THREAD_TRACE_BUFFER_SIZE:-${zimage_trace_buffer_bytes}}"
    export RADV_CACHE_COUNTERS_BUFFER_SIZE="${RADV_CACHE_COUNTERS_BUFFER_SIZE:-${zimage_counter_buffer_bytes}}"

    if [[ "${zimage_profile_mode}" == "rgp_profile" ]]; then
        export GGML_VK_DISABLE_FUSION="${GGML_VK_DISABLE_FUSION:-1}"
        export GGML_VK_PROFILE_NODES_PER_SUBMIT="${GGML_VK_PROFILE_NODES_PER_SUBMIT:-1}"
        export GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT="${GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT:-${zimage_profile_matmul_bytes_per_submit}}"
    else
        export GGML_VK_PROFILE_NODES_PER_SUBMIT="${GGML_VK_PROFILE_NODES_PER_SUBMIT:-${zimage_profile_nodes_per_submit}}"
        export GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT="${GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT:-${zimage_profile_matmul_bytes_per_submit}}"
    fi
    if [[ -n "${zimage_force_split_k}" ]]; then
        export GGML_VK_FORCE_SPLIT_K="${GGML_VK_FORCE_SPLIT_K:-${zimage_force_split_k}}"
    fi
}

print_zimage_runtime_env_summary() {
    cat <<EOF
profile_mode=${zimage_profile_mode}
trace_buffer_bytes=${RADV_THREAD_TRACE_BUFFER_SIZE:-${zimage_trace_buffer_bytes}}
counter_buffer_bytes=${RADV_CACHE_COUNTERS_BUFFER_SIZE:-${zimage_counter_buffer_bytes}}
debug_markers=${GGML_VK_DEBUG_MARKERS:-1}
flash_attn=${zimage_flash_attn}
vae_cpu=${zimage_vae_cpu}
clip_cpu=${zimage_clip_cpu}
diffusion_conv_direct=${zimage_diffusion_conv_direct}
vae_conv_direct=${zimage_vae_conv_direct}
nodes_per_submit=${GGML_VK_PROFILE_NODES_PER_SUBMIT:-${zimage_profile_nodes_per_submit}}
matmul_bytes_per_submit=${GGML_VK_PROFILE_MATMUL_BYTES_PER_SUBMIT:-${zimage_profile_matmul_bytes_per_submit}}
force_split_k=${GGML_VK_FORCE_SPLIT_K:-}
disable_fusion=${GGML_VK_DISABLE_FUSION:-0}
disable_graph_optimize=${GGML_VK_DISABLE_GRAPH_OPTIMIZE:-0}
EOF
}

required_weight_files=(
  "z-image-turbo-Q4_K_S.gguf"
  "Qwen3-4B-Q4_K_S.gguf"
  "zimage-vae.safetensors"
)

have_required_weights() {
    local dir="${1}"
    local file
    for file in "${required_weight_files[@]}"; do
        [[ -f "${dir}/${file}" ]] || return 1
    done
}
