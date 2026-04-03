#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -d "${windows_games_root}" ]]; then
    echo "${windows_games_root}"
    exit 0
fi

sudo mkdir -p "${windows_mount_root}"

partition="${WINDOWS_PARTITION:-$(lsblk -lnpo NAME,FSTYPE,SIZE | awk '$2=="ntfs"{print $1, $3}' | sort -k2 -h | tail -n1 | awk '{print $1}')}"
if [[ -z "${partition}" ]]; then
    echo "failed to locate an NTFS partition to mount" >&2
    exit 1
fi

if ! findmnt -rno TARGET "${windows_mount_root}" >/dev/null 2>&1; then
    sudo mount -t ntfs "${partition}" "${windows_mount_root}" >/dev/null
fi

if [[ ! -d "${windows_games_root}" ]]; then
    echo "missing Windows games root: ${windows_games_root}" >&2
    exit 1
fi

echo "${windows_games_root}"
