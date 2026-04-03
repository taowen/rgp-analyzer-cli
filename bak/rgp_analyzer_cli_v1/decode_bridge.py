from __future__ import annotations

from typing import Any

from .rgp_records import (
    collect_code_object_records,
    collect_loader_load_records,
    collect_pso_records,
    find_code_object_record,
)
from .stitch_model import build_stitch_model


def decoder_bridge(report: dict[str, Any]) -> dict[str, Any]:
    model = build_stitch_model(report)
    return {
        "bridge_kind": "co_col_pso",
        "code_object_record_count": model["code_object_record_count"],
        "loader_load_record_count": model["loader_load_record_count"],
        "pso_record_count": model["pso_record_count"],
        "bridged_count": len(model["entries"]),
        "resolved_entry_count": model["resolved_entry_count"],
        "warnings": model["warnings"],
        "entries": model["entries"],
        "streams": model["streams"],
    }


__all__ = [
    "build_stitch_model",
    "collect_code_object_records",
    "collect_loader_load_records",
    "collect_pso_records",
    "decoder_bridge",
    "find_code_object_record",
]
