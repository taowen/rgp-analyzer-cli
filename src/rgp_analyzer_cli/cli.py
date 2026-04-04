from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import compare_v1_v2
from .metrics_reference import (
    metrics_reference_payload,
    render_metrics_reference_markdown,
    render_metrics_reference_report_section,
)
from .models import CompareOptions, DecodeOptions, DispatchIsaOptions, ShaderFocusOptions, TriageOptions
from .render import (
    render_compare_region_focus_payload,
    render_code_object_isa_payload,
    render_compare_capture_payload,
    render_compare_shader_focus_payload,
    render_decode_payload,
    render_dispatch_payload,
    render_inspect,
    render_resource_payload,
    render_region_focus_payload,
    render_shader_focus_payload,
    render_stitch_report,
    render_triage_payload,
)
from .services import (
    code_object_isa_payload,
    compare_capture_payload,
    compare_shader_focus_payload,
    decode_payload,
    dispatch_isa_payload,
    inspect_payload,
    load_capture,
    resource_payload,
    shader_focus_payload,
    stitch_payload,
    triage_payload,
)


def _json_output(payload: object) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rgp-analyzer", description="CLI for RGP analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_capture(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("rgp_file", type=Path)
        cmd.add_argument("--json", action="store_true", dest="as_json")

    inspect_parser = subparsers.add_parser("inspect")
    add_capture(inspect_parser)
    inspect_parser.add_argument("--limit", type=int, default=10)

    stitch_parser = subparsers.add_parser("stitch-report")
    add_capture(stitch_parser)
    stitch_parser.add_argument("--limit", type=int, default=10)

    decode_parser = subparsers.add_parser("decode-sqtt")
    add_capture(decode_parser)
    decode_parser.add_argument("--build-helper", action="store_true")
    decode_parser.add_argument("--helper", type=Path, default=None)
    decode_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    decode_parser.add_argument("--stream-limit", type=int, default=0)
    decode_parser.add_argument("--hotspot-limit", type=int, default=8)
    decode_parser.add_argument("--no-cache", action="store_true")

    resource_parser = subparsers.add_parser("resource-summary")
    add_capture(resource_parser)
    resource_parser.add_argument("--limit", type=int, default=10)
    resource_parser.add_argument("--tool", default=None)

    dispatch_parser = subparsers.add_parser("dispatch-isa-map")
    add_capture(dispatch_parser)
    dispatch_parser.add_argument("--stream-index", type=int, default=0)
    dispatch_parser.add_argument("--dispatch-limit", type=int, default=8)
    dispatch_parser.add_argument("--context-packets", type=int, default=64)
    dispatch_parser.add_argument("--tail-packets", type=int, default=512)
    dispatch_parser.add_argument("--packet-limit", type=int, default=20000)
    dispatch_parser.add_argument("--mapped-limit", type=int, default=32)
    dispatch_parser.add_argument("--tool", default=None)
    dispatch_parser.add_argument("--limit", type=int, default=16)
    dispatch_parser.add_argument("--no-cache", action="store_true")

    triage_parser = subparsers.add_parser("shader-triage")
    add_capture(triage_parser)
    triage_parser.add_argument("--build-helper", action="store_true")
    triage_parser.add_argument("--helper", type=Path, default=None)
    triage_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    triage_parser.add_argument("--isa-tool", default=None)
    triage_parser.add_argument("--readelf-tool", default=None)
    triage_parser.add_argument("--limit", type=int, default=10)
    triage_parser.add_argument("--hotspot-limit", type=int, default=8)
    triage_parser.add_argument("--no-cache", action="store_true")

    compare_parser = subparsers.add_parser("compare-v1-v2")
    compare_parser.add_argument("rgp_file", type=Path)
    compare_parser.add_argument("--json", action="store_true", dest="as_json")

    compare_capture_parser = subparsers.add_parser("compare-captures")
    compare_capture_parser.add_argument("baseline_rgp", type=Path)
    compare_capture_parser.add_argument("candidate_rgp", type=Path)
    compare_capture_parser.add_argument("--json", action="store_true", dest="as_json")
    compare_capture_parser.add_argument("--build-helper", action="store_true")
    compare_capture_parser.add_argument("--helper", type=Path, default=None)
    compare_capture_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    compare_capture_parser.add_argument("--isa-tool", default=None)
    compare_capture_parser.add_argument("--readelf-tool", default=None)
    compare_capture_parser.add_argument("--limit", type=int, default=10)
    compare_capture_parser.add_argument("--hotspot-limit", type=int, default=8)
    compare_capture_parser.add_argument("--no-cache", action="store_true")

    shader_focus_parser = subparsers.add_parser("shader-focus")
    add_capture(shader_focus_parser)
    shader_focus_parser.add_argument("--build-helper", action="store_true")
    shader_focus_parser.add_argument("--helper", type=Path, default=None)
    shader_focus_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    shader_focus_parser.add_argument("--isa-tool", default=None)
    shader_focus_parser.add_argument("--readelf-tool", default=None)
    shader_focus_parser.add_argument("--limit", type=int, default=10)
    shader_focus_parser.add_argument("--hotspot-limit", type=int, default=8)
    shader_focus_parser.add_argument("--code-object-index", type=int, default=None)
    shader_focus_parser.add_argument("--source-file", type=Path, default=None)
    shader_focus_parser.add_argument("--source-excerpt", action="store_true")
    shader_focus_parser.add_argument("--no-cache", action="store_true")

    region_focus_parser = subparsers.add_parser("region-focus")
    add_capture(region_focus_parser)
    region_focus_parser.add_argument("--build-helper", action="store_true")
    region_focus_parser.add_argument("--helper", type=Path, default=None)
    region_focus_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    region_focus_parser.add_argument("--isa-tool", default=None)
    region_focus_parser.add_argument("--readelf-tool", default=None)
    region_focus_parser.add_argument("--limit", type=int, default=10)
    region_focus_parser.add_argument("--hotspot-limit", type=int, default=8)
    region_focus_parser.add_argument("--code-object-index", type=int, default=None)
    region_focus_parser.add_argument("--source-file", type=Path, required=True)
    region_focus_parser.add_argument("--source-excerpt", action="store_true")
    region_focus_parser.add_argument("--no-cache", action="store_true")

    compare_shader_focus_parser = subparsers.add_parser("compare-shader-focus")
    compare_shader_focus_parser.add_argument("baseline_rgp", type=Path)
    compare_shader_focus_parser.add_argument("candidate_rgp", type=Path)
    compare_shader_focus_parser.add_argument("--json", action="store_true", dest="as_json")
    compare_shader_focus_parser.add_argument("--build-helper", action="store_true")
    compare_shader_focus_parser.add_argument("--helper", type=Path, default=None)
    compare_shader_focus_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    compare_shader_focus_parser.add_argument("--isa-tool", default=None)
    compare_shader_focus_parser.add_argument("--readelf-tool", default=None)
    compare_shader_focus_parser.add_argument("--limit", type=int, default=10)
    compare_shader_focus_parser.add_argument("--hotspot-limit", type=int, default=8)
    compare_shader_focus_parser.add_argument("--code-object-index", type=int, default=None)
    compare_shader_focus_parser.add_argument("--source-file", type=Path, default=None)
    compare_shader_focus_parser.add_argument("--source-excerpt", action="store_true")
    compare_shader_focus_parser.add_argument("--no-cache", action="store_true")

    compare_region_focus_parser = subparsers.add_parser("compare-region-focus")
    compare_region_focus_parser.add_argument("baseline_rgp", type=Path)
    compare_region_focus_parser.add_argument("candidate_rgp", type=Path)
    compare_region_focus_parser.add_argument("--json", action="store_true", dest="as_json")
    compare_region_focus_parser.add_argument("--build-helper", action="store_true")
    compare_region_focus_parser.add_argument("--helper", type=Path, default=None)
    compare_region_focus_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    compare_region_focus_parser.add_argument("--isa-tool", default=None)
    compare_region_focus_parser.add_argument("--readelf-tool", default=None)
    compare_region_focus_parser.add_argument("--limit", type=int, default=10)
    compare_region_focus_parser.add_argument("--hotspot-limit", type=int, default=8)
    compare_region_focus_parser.add_argument("--code-object-index", type=int, default=None)
    compare_region_focus_parser.add_argument("--source-file", type=Path, required=True)
    compare_region_focus_parser.add_argument("--source-excerpt", action="store_true")
    compare_region_focus_parser.add_argument("--no-cache", action="store_true")

    code_object_isa_parser = subparsers.add_parser("code-object-isa")
    add_capture(code_object_isa_parser)
    code_object_isa_parser.add_argument("--build-helper", action="store_true")
    code_object_isa_parser.add_argument("--helper", type=Path, default=None)
    code_object_isa_parser.add_argument("--decoder-lib-dir", type=Path, default=None)
    code_object_isa_parser.add_argument("--isa-tool", default=None)
    code_object_isa_parser.add_argument("--readelf-tool", default=None)
    code_object_isa_parser.add_argument("--limit", type=int, default=10)
    code_object_isa_parser.add_argument("--hotspot-limit", type=int, default=8)
    code_object_isa_parser.add_argument("--code-object-index", type=int, default=None)
    code_object_isa_parser.add_argument("--source-file", type=Path, default=None)
    code_object_isa_parser.add_argument("--symbol", default="_amdgpu_cs_main")
    code_object_isa_parser.add_argument("--isa-limit", type=int, default=32)
    code_object_isa_parser.add_argument("--no-cache", action="store_true")

    metrics_parser = subparsers.add_parser("metrics-doc")
    metrics_parser.add_argument("--json", action="store_true", dest="as_json")
    metrics_parser.add_argument("--format", choices=("report", "markdown"), default="report")
    metrics_parser.add_argument("--section-title", default="Metrics Reference")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "compare-v1-v2":
        payload = compare_v1_v2(args.rgp_file)
        if args.as_json:
            return _json_output(payload)
        print("compare_v1:")
        print(f"  capture: {payload['capture']}")
        print(f"  all_match: {payload['all_match']}")
        for name, ok in payload["checks"].items():
            print(f"  {name}: {ok}")
        return 0

    if args.command == "compare-captures":
        baseline = load_capture(args.baseline_rgp)
        candidate = load_capture(args.candidate_rgp)
        payload = compare_capture_payload(
            baseline,
            candidate,
            CompareOptions(
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
                isa_tool=args.isa_tool,
                readelf_tool=args.readelf_tool,
                limit=args.limit,
                hotspot_limit=args.hotspot_limit,
                use_cache=not args.no_cache,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_compare_capture_payload(payload))
        return 0

    if args.command == "compare-shader-focus":
        baseline = load_capture(args.baseline_rgp)
        candidate = load_capture(args.candidate_rgp)
        payload = compare_shader_focus_payload(
            baseline,
            candidate,
            CompareOptions(
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
                isa_tool=args.isa_tool,
                readelf_tool=args.readelf_tool,
                limit=args.limit,
                hotspot_limit=args.hotspot_limit,
                code_object_index=args.code_object_index,
                source_file=args.source_file,
                use_cache=not args.no_cache,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_compare_shader_focus_payload(payload, source_excerpt=args.source_excerpt))
        return 0

    if args.command == "compare-region-focus":
        baseline = load_capture(args.baseline_rgp)
        candidate = load_capture(args.candidate_rgp)
        payload = compare_shader_focus_payload(
            baseline,
            candidate,
            CompareOptions(
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
                isa_tool=args.isa_tool,
                readelf_tool=args.readelf_tool,
                limit=args.limit,
                hotspot_limit=args.hotspot_limit,
                code_object_index=args.code_object_index,
                source_file=args.source_file,
                use_cache=not args.no_cache,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_compare_region_focus_payload(payload, source_excerpt=args.source_excerpt))
        return 0

    if args.command == "code-object-isa":
        session = load_capture(args.rgp_file)
        payload = code_object_isa_payload(
            session,
            ShaderFocusOptions(
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
                isa_tool=args.isa_tool,
                readelf_tool=args.readelf_tool,
                limit=args.limit,
                hotspot_limit=args.hotspot_limit,
                code_object_index=args.code_object_index,
                source_file=args.source_file,
                use_cache=not args.no_cache,
            ),
            limit=args.isa_limit,
            symbol=args.symbol,
        )
        if args.as_json:
            return _json_output(payload)
        print(render_code_object_isa_payload(payload))
        return 0

    if args.command == "metrics-doc":
        payload = metrics_reference_payload()
        if args.as_json:
            return _json_output(payload)
        if args.format == "markdown":
            print(render_metrics_reference_markdown(payload, title=args.section_title), end="")
        else:
            print(render_metrics_reference_report_section(payload, title=args.section_title))
        return 0

    session = load_capture(args.rgp_file)

    if args.command == "inspect":
        payload = inspect_payload(session)
        if args.as_json:
            return _json_output(payload)
        print(render_inspect(session.report, limit=args.limit))
        return 0

    if args.command == "stitch-report":
        payload = stitch_payload(session)
        if args.as_json:
            return _json_output(payload)
        print(render_stitch_report(session.report, limit=args.limit))
        return 0

    if args.command == "decode-sqtt":
        payload = decode_payload(
            session,
            DecodeOptions(
                stream_limit=args.stream_limit,
                hotspot_limit=args.hotspot_limit,
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
                use_cache=not args.no_cache,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_decode_payload(payload))
        return 0

    if args.command == "resource-summary":
        payload = resource_payload(session, limit=args.limit, tool=args.tool)
        if args.as_json:
            return _json_output(payload)
        print(render_resource_payload(payload))
        return 0

    if args.command == "dispatch-isa-map":
        payload = dispatch_isa_payload(
            session,
            DispatchIsaOptions(
                stream_index=args.stream_index,
                dispatch_limit=args.dispatch_limit,
                context_packets=args.context_packets,
                tail_packets=args.tail_packets,
                packet_limit=args.packet_limit,
                mapped_limit=args.mapped_limit,
                tool=args.tool,
                use_cache=not args.no_cache,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_dispatch_payload(payload, limit=args.limit))
        return 0

    if args.command == "shader-triage":
        payload = triage_payload(
            session,
            TriageOptions(
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
                isa_tool=args.isa_tool,
                readelf_tool=args.readelf_tool,
                limit=args.limit,
                hotspot_limit=args.hotspot_limit,
                use_cache=not args.no_cache,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_triage_payload(payload))
        return 0

    if args.command == "shader-focus":
        payload = shader_focus_payload(
            session,
            ShaderFocusOptions(
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
                isa_tool=args.isa_tool,
                readelf_tool=args.readelf_tool,
                limit=args.limit,
                hotspot_limit=args.hotspot_limit,
                code_object_index=args.code_object_index,
                source_file=args.source_file,
                use_cache=not args.no_cache,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_shader_focus_payload(payload, source_excerpt=args.source_excerpt))
        return 0

    if args.command == "region-focus":
        payload = shader_focus_payload(
            session,
            ShaderFocusOptions(
                build_helper=args.build_helper,
                helper=args.helper,
                decoder_lib_dir=args.decoder_lib_dir,
                isa_tool=args.isa_tool,
                readelf_tool=args.readelf_tool,
                limit=args.limit,
                hotspot_limit=args.hotspot_limit,
                code_object_index=args.code_object_index,
                source_file=args.source_file,
                use_cache=not args.no_cache,
            ),
        )
        if args.as_json:
            return _json_output(payload)
        print(render_region_focus_payload(payload, source_excerpt=args.source_excerpt))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2
