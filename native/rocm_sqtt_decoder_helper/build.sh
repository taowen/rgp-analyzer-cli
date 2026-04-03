#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

CXX="${CXX:-g++}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"
INCLUDE_DIR="${ROCM_ROOT}/include"
LIB_DIR="${ROCM_ROOT}/lib"
OUT="${BUILD_DIR}/rocm-sqtt-decoder-helper"

"${CXX}" \
  -std=c++17 \
  -O2 \
  -Wall \
  -Wextra \
  -D__HIP_PLATFORM_AMD__=1 \
  -I"${INCLUDE_DIR}" \
  "${SCRIPT_DIR}/main.cpp" \
  -L"${LIB_DIR}" \
  -Wl,-rpath,"${LIB_DIR}" \
  -lrocprofiler-sdk \
  -ldl \
  -pthread \
  -o "${OUT}"

echo "${OUT}"
