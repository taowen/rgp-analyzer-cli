from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .analyze import (
    compare_reports,
    correlation_summary,
    flattened_events,
    generate_advice,
    summarize_isa_text,
)
from .decode_bridge import build_stitch_model, decoder_bridge
from .json_utils import json_ready
from .native_decode import default_decoder_lib_dir, repo_root as decode_repo_root, resolve_helper_path, run_decode_helper
from .parser import materialize_code_object_payload, parse_rgp
from .resource_metadata import extract_resource_metadata
from .shader_triage import shader_triage
from .sqtt_stitch import render_stitched_hotspots
from .marker_scan import filter_markers, render_general_api_markers, render_markers, scan_general_api_markers, scan_markers
from .tinygrad_support.decoder import analyze_tinygrad_sqtt, render_tinygrad_sqtt
from .tinygrad_support.isa_map import (
    map_dispatch_spans_to_isa,
    map_tinygrad_packets_to_isa,
    render_dispatch_isa_map,
    render_tinygrad_isa_map,
)
from .render import (
    render_advice,
    render_chunks,
    render_compare,
    render_correlate,
    render_code_objects,
    render_decoder_bridge,
    render_events,
    render_inspect,
    render_loader_events,
    render_stitch_report,
    render_pso,
    render_scan_sqtt,
    render_sqtt,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rgp-analyzer", description="Inspect and analyze Radeon GPU Profiler .rgp files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("rgp_file", type=Path)
        subparser.add_argument("--json", action="store_true", dest="as_json")

    inspect_parser = subparsers.add_parser("inspect", help="One-shot capture inspection")
    add_common(inspect_parser)
    inspect_parser.add_argument("--limit", type=int, default=10)

    chunks_parser = subparsers.add_parser("chunks", help="List chunk inventory")
    add_common(chunks_parser)

    events_parser = subparsers.add_parser("events", help="List queue events")
    add_common(events_parser)
    events_parser.add_argument("--limit", type=int, default=20)
    events_parser.add_argument(
        "--sort",
        choices=["gpu_duration", "cpu_timestamp", "frame_index", "submit_sub_index"],
        default="gpu_duration",
    )

    code_parser = subparsers.add_parser("code-objects", help="Summarize code object records")
    add_common(code_parser)
    code_parser.add_argument("--limit", type=int, default=10)
    code_parser.add_argument("--show-strings", action="store_true")

    pso_parser = subparsers.add_parser("pso", help="List PSO correlation records")
    add_common(pso_parser)
    pso_parser.add_argument("--limit", type=int, default=20)

    loader_parser = subparsers.add_parser("loader-events", help="List code object load/unload records")
    add_common(loader_parser)
    loader_parser.add_argument("--limit", type=int, default=20)

    sqtt_parser = subparsers.add_parser("sqtt", help="List SQTT metadata chunks")
    add_common(sqtt_parser)
    sqtt_parser.add_argument("--limit", type=int, default=20)

    scan_sqtt_parser = subparsers.add_parser("scan-sqtt", help="Preview raw SQTT stream DWORD patterns")
    add_common(scan_sqtt_parser)
    scan_sqtt_parser.add_argument("--stream-limit", type=int, default=2)
    scan_sqtt_parser.add_argument("--dword-limit", type=int, default=32)

    marker_parser = subparsers.add_parser("general-api-markers", help="Scan SQTT streams for raw RGP General API markers")
    add_common(marker_parser)
    marker_parser.add_argument("--stream-limit", type=int, default=1)
    marker_parser.add_argument("--limit", type=int, default=20)

    tinygrad_parser = subparsers.add_parser(
        "tinygrad-sqtt", help="Experimental: use tinygrad's raw SQTT decoder on extracted SQTT streams"
    )
    add_common(tinygrad_parser)
    tinygrad_parser.add_argument("--tinygrad-path", type=Path, default=None)
    tinygrad_parser.add_argument("--stream-limit", type=int, default=1)
    tinygrad_parser.add_argument("--packet-limit", type=int, default=20000)
    tinygrad_parser.add_argument("--preview-limit", type=int, default=20)
    tinygrad_parser.add_argument("--limit", type=int, default=20)

    tinygrad_map_parser = subparsers.add_parser(
        "tinygrad-isa-map", help="Experimental: map tinygrad-decoded SQTT packets onto llvm-objdump ISA"
    )
    add_common(tinygrad_map_parser)
    tinygrad_map_parser.add_argument("--tinygrad-path", type=Path, default=None)
    tinygrad_map_parser.add_argument("--tool", default=None)
    tinygrad_map_parser.add_argument("--stream-index", type=int, default=0)
    tinygrad_map_parser.add_argument("--packet-limit", type=int, default=20000)
    tinygrad_map_parser.add_argument("--mapped-limit", type=int, default=64)
    tinygrad_map_parser.add_argument("--limit", type=int, default=32)

    dispatch_map_parser = subparsers.add_parser(
        "dispatch-isa-map", help="Experimental: map tinygrad-decoded SQTT packets per dispatch span onto llvm-objdump ISA"
    )
    add_common(dispatch_map_parser)
    dispatch_map_parser.add_argument("--tinygrad-path", type=Path, default=None)
    dispatch_map_parser.add_argument("--tool", default=None)
    dispatch_map_parser.add_argument("--stream-index", type=int, default=0)
    dispatch_map_parser.add_argument("--dispatch-limit", type=int, default=8)
    dispatch_map_parser.add_argument("--context-packets", type=int, default=64)
    dispatch_map_parser.add_argument("--tail-packets", type=int, default=512)
    dispatch_map_parser.add_argument("--packet-limit", type=int, default=20000)
    dispatch_map_parser.add_argument("--mapped-limit", type=int, default=32)
    dispatch_map_parser.add_argument("--limit", type=int, default=16)

    markers_parser = subparsers.add_parser("markers", help="Experimental: scan SQTT streams for raw RGP marker records")
    add_common(markers_parser)
    markers_parser.add_argument("--stream-limit", type=int, default=1)
    markers_parser.add_argument("--limit", type=int, default=20)
    markers_parser.add_argument("--confidence", choices=["high", "medium", "low"], default=None)

    correlate_parser = subparsers.add_parser("correlate", help="Correlate PSO hashes with loader-event hashes")
    add_common(correlate_parser)
    correlate_parser.add_argument("--limit", type=int, default=20)

    bridge_parser = subparsers.add_parser("decoder-bridge", help="Show the code-object to loader-event bridge used for SQTT decode")
    add_common(bridge_parser)
    bridge_parser.add_argument("--limit", type=int, default=20)

    stitch_parser = subparsers.add_parser(
        "stitch-report", help="Show the authoritative CO + COL + PSO stitch model derived from the .rgp capture"
    )
    add_common(stitch_parser)
    stitch_parser.add_argument("--limit", type=int, default=20)

    compare_parser = subparsers.add_parser("compare", help="Compare two .rgp captures")
    compare_parser.add_argument("baseline_rgp", type=Path)
    compare_parser.add_argument("candidate_rgp", type=Path)
    compare_parser.add_argument("--json", action="store_true", dest="as_json")

    extract_code_parser = subparsers.add_parser("extract-code-objects", help="Extract embedded code object ELF payloads")
    add_common(extract_code_parser)
    extract_code_parser.add_argument("--out-dir", type=Path, required=True)

    disasm_parser = subparsers.add_parser("disassemble-code-objects", help="Disassemble embedded AMDGPU code object ELF payloads")
    add_common(disasm_parser)
    disasm_parser.add_argument("--tool", default="llvm-objdump-18")
    disasm_parser.add_argument("--limit", type=int, default=10)
    disasm_parser.add_argument("--symbol", default=None)

    isa_summary_parser = subparsers.add_parser("isa-summary", help="Summarize ISA mix from embedded AMDGPU code objects")
    add_common(isa_summary_parser)
    isa_summary_parser.add_argument("--tool", default="llvm-objdump-18")
    isa_summary_parser.add_argument("--limit", type=int, default=10)
    isa_summary_parser.add_argument("--symbol", default=None)

    extract_sqtt_parser = subparsers.add_parser("extract-sqtt", help="Extract raw SQTT data streams")
    add_common(extract_sqtt_parser)
    extract_sqtt_parser.add_argument("--out-dir", type=Path, required=True)

    resource_parser = subparsers.add_parser("resource-summary", help="Extract VGPR/SGPR/LDS/scratch metadata from embedded code objects")
    add_common(resource_parser)
    resource_parser.add_argument("--tool", default=None)
    resource_parser.add_argument("--limit", type=int, default=10)

    triage_parser = subparsers.add_parser("shader-triage", help="Combine runtime SQTT, ISA, and resource metadata into shader tuning findings")
    add_common(triage_parser)
    triage_parser.add_argument("--build-helper", action="store_true")
    triage_parser.add_argument("--helper", type=Path, default=None)
    triage_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    triage_parser.add_argument("--isa-tool", default=None)
    triage_parser.add_argument("--readelf-tool", default=None)
    triage_parser.add_argument("--limit", type=int, default=10)
    triage_parser.add_argument("--hotspot-limit", type=int, default=8)

    decode_sqtt_parser = subparsers.add_parser("decode-sqtt", help="Decode SQTT streams through ROCm thread-trace decoder")
    add_common(decode_sqtt_parser)
    decode_sqtt_parser.add_argument("--stream-limit", type=int, default=0)
    decode_sqtt_parser.add_argument("--build-helper", action="store_true")
    decode_sqtt_parser.add_argument("--helper", type=Path, default=None)
    decode_sqtt_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    decode_sqtt_parser.add_argument("--keep-temp", action="store_true")
    decode_sqtt_parser.add_argument("--strict", action="store_true")
    decode_sqtt_parser.add_argument("--hotspot-limit", type=int, default=8)

    advice_parser = subparsers.add_parser("advice", help="Show next-step suggestions")
    add_common(advice_parser)

    return parser


def _json_output(data: Any) -> int:
    print(json.dumps(json_ready(data), ensure_ascii=True, indent=2))
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    report = parse_rgp(args.rgp_file)

    if args.command == "inspect":
        if args.as_json:
            payload = dict(report)
            payload["advice"] = generate_advice(report)
            return _json_output(payload)
        print(render_inspect(report, limit=args.limit))
        return 0

    if args.command == "chunks":
        if args.as_json:
            return _json_output(report["chunks"])
        print(render_chunks(report))
        return 0

    if args.command == "events":
        events = sorted(flattened_events(report), key=lambda item: item.get(args.sort, 0), reverse=True)
        if args.as_json:
            return _json_output(events[: args.limit])
        print(render_events(report, limit=args.limit, sort_key=args.sort))
        return 0

    if args.command == "code-objects":
        records = []
        for chunk in report["code_object_databases"]:
            records.extend(chunk["records"])
        if args.as_json:
            return _json_output(records[: args.limit])
        print(render_code_objects(report, show_strings=args.show_strings, limit=args.limit))
        return 0

    if args.command == "pso":
        records = []
        for chunk in report["pso_correlations"]:
            records.extend(chunk["records"])
        if args.as_json:
            return _json_output(records[: args.limit])
        print(render_pso(report, limit=args.limit))
        return 0

    if args.command == "loader-events":
        records = []
        for chunk in report["loader_events"]:
            records.extend(chunk["records"])
        if args.as_json:
            return _json_output(records[: args.limit])
        print(render_loader_events(report, limit=args.limit))
        return 0

    if args.command == "advice":
        advice = generate_advice(report)
        if args.as_json:
            return _json_output(advice)
        print(render_advice(report))
        return 0

    if args.command == "correlate":
        corr = correlation_summary(report)
        if args.as_json:
            return _json_output(corr)
        print(render_correlate(report, limit=args.limit))
        return 0

    if args.command == "decoder-bridge":
        bridge = decoder_bridge(report)
        if args.as_json:
            return _json_output(bridge)
        print(render_decoder_bridge(report, limit=args.limit))
        return 0

    if args.command == "stitch-report":
        if args.as_json:
            return _json_output(build_stitch_model(report))
        print(render_stitch_report(report, limit=args.limit))
        return 0

    if args.command == "sqtt":
        payload = {
            "sqtt_descs": report["sqtt_descs"][: args.limit],
            "sqtt_data_chunks": report["sqtt_data_chunks"][: args.limit],
        }
        if args.as_json:
            return _json_output(payload)
        print(render_sqtt(report, limit=args.limit))
        return 0

    if args.command == "scan-sqtt":
        payload = []
        from .analyze import scan_sqtt_payload

        for index in range(min(args.stream_limit, len(report["sqtt_data_chunks"]))):
            payload.append(scan_sqtt_payload(report, index=index, dword_limit=args.dword_limit))
        if args.as_json:
            return _json_output(payload)
        print(render_scan_sqtt(report, stream_limit=args.stream_limit, dword_limit=args.dword_limit))
        return 0

    if args.command == "general-api-markers":
        result = scan_general_api_markers(report, stream_limit=args.stream_limit)
        if args.as_json:
            return _json_output(result)
        print(render_general_api_markers(result, limit=args.limit))
        return 0

    if args.command == "tinygrad-sqtt":
        result = analyze_tinygrad_sqtt(
            report,
            tinygrad_path=args.tinygrad_path,
            stream_limit=args.stream_limit,
            packet_limit=args.packet_limit,
            preview_limit=args.preview_limit,
        )
        if args.as_json:
            return _json_output(result)
        print(render_tinygrad_sqtt(result, limit=args.limit, preview_limit=args.preview_limit))
        return 0

    if args.command == "tinygrad-isa-map":
        result = map_tinygrad_packets_to_isa(
            report,
            tinygrad_path=args.tinygrad_path,
            tool=args.tool,
            stream_index=args.stream_index,
            packet_limit=args.packet_limit,
            mapped_limit=args.mapped_limit,
        )
        if args.as_json:
            return _json_output(result)
        print(render_tinygrad_isa_map(result, limit=args.limit))
        return 0

    if args.command == "dispatch-isa-map":
        result = map_dispatch_spans_to_isa(
            report,
            tinygrad_path=args.tinygrad_path,
            tool=args.tool,
            stream_index=args.stream_index,
            dispatch_limit=args.dispatch_limit,
            context_packets=args.context_packets,
            tail_packets=args.tail_packets,
            packet_limit=args.packet_limit,
            mapped_limit=args.mapped_limit,
        )
        if args.as_json:
            return _json_output(result)
        print(render_dispatch_isa_map(result, limit=args.limit))
        return 0

    if args.command == "markers":
        result = filter_markers(scan_markers(report, stream_limit=args.stream_limit), confidence=args.confidence)
        if args.as_json:
            return _json_output(result)
        print(render_markers(result, limit=args.limit))
        return 0

    if args.command == "extract-code-objects":
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for chunk in report["code_object_databases"]:
            for record in chunk["records"]:
                payload = materialize_code_object_payload(report, record, pad_elf=True)
                path = out_dir / f"code-object-{record['index']:03d}.elf"
                path.write_bytes(payload)
                written.append(str(path))
        if args.as_json:
            return _json_output(written)
        for path in written:
            print(path)
        return 0

    if args.command == "disassemble-code-objects":
        tool = shutil.which(args.tool) or shutil.which("llvm-objdump-18") or shutil.which("llvm-objdump")
        if tool is None:
            parser.error("llvm-objdump tool not found")
        results = []
        records = []
        for chunk in report["code_object_databases"]:
            records.extend(chunk["records"])
        with tempfile.TemporaryDirectory(prefix="rgp-analyzer-disasm-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            for record in records[: args.limit]:
                payload = materialize_code_object_payload(report, record, pad_elf=True)
                path = tmpdir_path / f"code-object-{record['index']:03d}.elf"
                path.write_bytes(payload)
                cmd = [tool, "-d", str(path)]
                if args.symbol:
                    cmd = [tool, "-d", f"--disassemble-symbols={args.symbol}", str(path)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                results.append(
                    {
                        "index": record["index"],
                        "path": str(path),
                        "tool": tool,
                        "returncode": proc.returncode,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                    }
                )
        if args.as_json:
            return _json_output(results)
        for item in results:
            print(f"code_object[{item['index']}] tool={item['tool']} returncode={item['returncode']}")
            if item["stderr"].strip():
                print(item["stderr"].rstrip())
            print(item["stdout"].rstrip())
            print()
        return 0

    if args.command == "isa-summary":
        tool = shutil.which(args.tool) or shutil.which("llvm-objdump-18") or shutil.which("llvm-objdump")
        if tool is None:
            parser.error("llvm-objdump tool not found")
        results = []
        records = []
        for chunk in report["code_object_databases"]:
            records.extend(chunk["records"])
        with tempfile.TemporaryDirectory(prefix="rgp-analyzer-isa-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            for record in records[: args.limit]:
                payload = materialize_code_object_payload(report, record, pad_elf=True)
                path = tmpdir_path / f"code-object-{record['index']:03d}.elf"
                path.write_bytes(payload)
                cmd = [tool, "-d", str(path)]
                if args.symbol:
                    cmd = [tool, "-d", f"--disassemble-symbols={args.symbol}", str(path)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                summary = summarize_isa_text(proc.stdout)
                results.append(
                    {
                        "index": record["index"],
                        "tool": tool,
                        "returncode": proc.returncode,
                        "summary": summary,
                    }
                )
        if args.as_json:
            return _json_output(results)
        lines = ["isa_summary:"]
        for item in results:
            summary = item["summary"]
            lines.append(
                f"  code_object[{item['index']}] instructions={summary['instruction_count']} unique_opcodes={summary['unique_opcodes']}"
            )
            lines.append(
                f"    vector_alu={summary['vector_alu']} scalar_ops={summary['scalar_ops']} "
                + f"buffer_ops={summary['buffer_ops']} lds_ops={summary['lds_ops']} "
                + f"flat_ops={summary['flat_ops']} image_ops={summary['image_ops']} "
                + f"wait_ops={summary['wait_ops']} branch_ops={summary['branch_ops']}"
            )
            top = ", ".join(f"{name}:{count}" for name, count in list(summary["opcode_counts"].items())[:12])
            lines.append(f"    top_opcodes={top}")
        print("\n".join(lines))
        return 0

    if args.command == "extract-sqtt":
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        blob = report["_blob"]
        written = []
        for index, chunk in enumerate(report["sqtt_data_chunks"]):
            payload = blob[chunk["payload_offset"] : chunk["payload_end"]]
            path = out_dir / f"sqtt-{index:03d}.bin"
            path.write_bytes(payload)
            written.append(str(path))
        if args.as_json:
            return _json_output(written)
        for path in written:
            print(path)
        return 0

    if args.command == "resource-summary":
        results = extract_resource_metadata(report, tool=args.tool, limit=args.limit)
        if args.as_json:
            return _json_output(results)
        print("resource_summary:")
        for item in results:
            print(
                f"  code_object[{item['index']}] entry_point={item['entry_point']} "
                f"vgpr={item['vgpr_count']} sgpr={item['sgpr_count']} "
                f"lds={item['lds_size']} scratch={item['scratch_memory_size']} "
                f"wavefront={item['wavefront_size']} spill_threshold={item['spill_threshold']}"
            )
            if item.get("internal_pipeline_hash"):
                print(
                    f"    internal_pipeline_hash={item['internal_pipeline_hash'][0]}:{item['internal_pipeline_hash'][1]} "
                    f"api_shader_hash={item.get('api_shader_hash')}"
                )
        return 0

    if args.command == "shader-triage":
        triage = shader_triage(
            report,
            helper=args.helper,
            build_helper=args.build_helper,
            decoder_lib_dir=args.decoder_lib_dir,
            isa_tool=args.isa_tool,
            readelf_tool=args.readelf_tool,
            limit=args.limit,
            hotspot_limit=args.hotspot_limit,
        )
        if args.as_json:
            return _json_output(triage)
        print("shader_triage:")
        for item in triage["findings"]:
            print(f"  - {item}")
        return 0

    if args.command == "decode-sqtt":
        resolved_helper = resolve_helper_path(decode_repo_root(), args.helper, args.build_helper)
        result = run_decode_helper(
            report,
            helper=resolved_helper,
            decoder_lib_dir=args.decoder_lib_dir or default_decoder_lib_dir(),
            stream_limit=args.stream_limit,
            hotspot_limit=args.hotspot_limit,
            strict=args.strict,
            as_json=True,
            keep_temp=args.keep_temp,
        )
        if args.keep_temp:
            print(f"manifest: {result['manifest_path']}")
            print(f"temp_root: {result['temp_root']}")
        if result.get("json"):
            stitched = result["json"].get("stitched")
            if args.as_json:
                return _json_output(result["json"])
            print("decode_sqtt:")
            print(f"  status: {result['json'].get('status')}")
            print(f"  decoder_lib_dir: {result['json'].get('decoder_lib_dir')}")
            print(f"  code_object_count: {result['json'].get('code_object_count')}")
            print(f"  code_object_load_failures: {result['json'].get('code_object_load_failures')}")
            print(f"  stream_count: {result['json'].get('stream_count')}")
            for warning in result["json"].get("warnings", []):
                print(f"  warning: {warning}")
            for stream in result["json"].get("streams", []):
                print(
                    f"  stream[{stream['index']}] se={stream['shader_engine_index']} cu={stream['compute_unit_index']} "
                    f"bytes={stream['bytes']} waves={stream['wave_records']} instructions={stream['instructions']}"
                )
                cats = stream.get("category_counts", {})
                if cats:
                    cat_text = ", ".join(f"{name}:{count}" for name, count in cats.items())
                    print(f"    categories={cat_text}")
            if result["json"].get("bridge"):
                print("bridge:")
                for entry in result["json"]["bridge"].get("entries", [])[:8]:
                    print(
                        f"  code_object[{entry['code_object_index']}] match={entry['match_kind']} "
                        f"entry_point={entry.get('entry_point')} api_pso_hash={entry.get('api_pso_hash')} "
                        f"load_addr=0x{int(entry.get('load_addr', 0)):x}"
                    )
            if stitched:
                print(render_stitched_hotspots(stitched))
        elif result["stdout"].strip():
            print(result["stdout"].rstrip())
        if result["stderr"].strip():
            print(result["stderr"].rstrip())
        return int(result["returncode"])

    if args.command == "compare":
        baseline = parse_rgp(args.baseline_rgp)
        candidate = parse_rgp(args.candidate_rgp)
        diff = compare_reports(baseline, candidate)
        if args.as_json:
            return _json_output(diff)
        print(render_compare(baseline, candidate))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
