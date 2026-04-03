#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


LINE_RE = re.compile(
    r"^(?P<name>.+?):\s+(?P<count>\d+)\s+x\s+(?P<avg_us>[0-9.]+)\s+us\s+=\s+(?P<total_us>[0-9.]+)\s+us(?:\s+\((?P<gflops>[0-9.]+)\s+GFLOPS/s\))?$"
)
TOTAL_RE = re.compile(r"^Total time:\s+(?P<total_us>[0-9.]+)\s+us\.?$")


def parse_timings(text: str) -> tuple[list[dict[str, object]], float | None]:
    in_block = False
    rows_by_name: dict[str, dict[str, object]] = {}
    total_time_sum = 0.0
    saw_total = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "Vulkan Timings:":
            in_block = True
            continue
        total_match = TOTAL_RE.match(line)
        if in_block and total_match:
            total_time_sum += float(total_match.group("total_us"))
            saw_total = True
            in_block = False
            continue
        if not in_block:
            continue
        match = LINE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        count = int(match.group("count"))
        avg_us = float(match.group("avg_us"))
        total_us = float(match.group("total_us"))
        gflops = float(match.group("gflops")) if match.group("gflops") else None
        item = rows_by_name.setdefault(
            name,
            {
                "name": name,
                "count": 0,
                "avg_us": 0.0,
                "total_us": 0.0,
                "gflops": None,
            },
        )
        item["count"] = int(item["count"]) + count
        item["total_us"] = float(item["total_us"]) + total_us
        item["avg_us"] = float(item["total_us"]) / max(int(item["count"]), 1)
        if gflops is not None:
            item["gflops"] = max(float(item["gflops"] or 0.0), gflops)

    rows = list(rows_by_name.values())
    rows.sort(key=lambda row: float(row["total_us"]), reverse=True)
    return rows, (total_time_sum if saw_total else None)


def canonical_family(name: str) -> str:
    family = name
    if family.startswith("SILU, MUL_MAT "):
        family = family.replace("SILU, ", "", 1)
        family = f"SILU+{family}"
    elif family.startswith("ADD, MUL_MAT "):
        family = family.replace("ADD, ", "", 1)
        family = f"ADD+{family}"

    if family.startswith("MUL_MAT "):
        quant_match = re.match(r"^MUL_MAT\s+([A-Za-z0-9_]+)\s+", family)
        if quant_match:
            return f"MUL_MAT {quant_match.group(1)}"
    return family.split(":", 1)[0]


def summarize_families(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        family = canonical_family(str(row["name"]))
        item = summary.setdefault(
            family,
            {
                "family": family,
                "total_us": 0.0,
                "count": 0,
                "variants": 0,
                "max_avg_us": 0.0,
            },
        )
        item["total_us"] = float(item["total_us"]) + float(row["total_us"])
        item["count"] = int(item["count"]) + int(row["count"])
        item["variants"] = int(item["variants"]) + 1
        item["max_avg_us"] = max(float(item["max_avg_us"]), float(row["avg_us"]))
    result = list(summary.values())
    result.sort(key=lambda row: float(row["total_us"]), reverse=True)
    return result


def format_rows(rows: list[dict[str, object]], total_time_us: float | None) -> str:
    if not rows:
        return "No Vulkan timing rows found."

    top_rows = rows[:10]
    lines = [""]
    if total_time_us is not None:
        lines.append(f"Vulkan Total Time: {total_time_us:.1f} us")
        lines.append("")
    lines.append("Top Vulkan Ops:")
    header = f"{'total_us':>12}  {'avg_us':>12}  {'count':>7}  {'gflops':>10}  name"
    lines.append(header)
    lines.append("-" * len(header))

    for row in top_rows:
        gflops = "-" if row["gflops"] is None else f"{float(row['gflops']):.2f}"
        lines.append(
            f"{float(row['total_us']):12.1f}  {float(row['avg_us']):12.1f}  {int(row['count']):7d}  {gflops:>10}  {row['name']}"
        )

    return "\n".join(lines)


def format_family_rows(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No Vulkan timing families found."

    top_rows = rows[:10]
    lines = ["", "Top Vulkan Op Families:"]
    header = f"{'total_us':>12}  {'count':>12}  {'variants':>8}  {'max_avg_us':>12}  family"
    lines.append(header)
    lines.append("-" * len(header))

    for row in top_rows:
        lines.append(
            f"{float(row['total_us']):12.1f}  {int(row['count']):12d}  {int(row['variants']):8d}  {float(row['max_avg_us']):12.1f}  {row['family']}"
        )

    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print(f"usage: {Path(sys.argv[0]).name} <log-file> [--json]", file=sys.stderr)
        return 2

    log_path = Path(sys.argv[1])
    if not log_path.is_file():
        print(f"log file not found: {log_path}", file=sys.stderr)
        return 2

    rows, total_time_us = parse_timings(log_path.read_text(encoding="utf-8", errors="replace"))
    family_rows = summarize_families(rows)
    if len(sys.argv) == 3:
        if sys.argv[2] != "--json":
            print(f"unknown option: {sys.argv[2]}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"total_time_us": total_time_us, "rows": rows, "families": family_rows},
                ensure_ascii=True,
                indent=2,
            )
        )
    else:
        print(format_rows(rows, total_time_us))
        print(format_family_rows(family_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
