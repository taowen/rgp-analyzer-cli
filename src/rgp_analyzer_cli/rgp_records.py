from __future__ import annotations

from typing import Any


def hash_pair(values: list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if not values or len(values) != 2:
        return None
    return int(values[0]), int(values[1])


def hash_text(values: list[int] | tuple[int, int] | None) -> str | None:
    pair = hash_pair(values)
    if pair is None:
        return None
    return f"{pair[0]:016x}:{pair[1]:016x}"


def collect_code_object_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for db in report["code_object_databases"]:
        records.extend(db["records"])
    return records


def collect_loader_load_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chunk in report["loader_events"]:
        for record in chunk["records"]:
            if record["loader_event_type_name"] == "LOAD_TO_GPU_MEMORY":
                records.append(record)
    return records


def collect_pso_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chunk in report["pso_correlations"]:
        records.extend(chunk["records"])
    return records


def find_code_object_record(report: dict[str, Any], code_object_index: int) -> dict[str, Any] | None:
    for record in collect_code_object_records(report):
        if record["index"] == code_object_index:
            return record
    return None
