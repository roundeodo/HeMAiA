#!/usr/bin/env python3
"""Validate and analyze dynamic-MoE runtime timing records."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


V2_RECORD_RE = re.compile(
    r"\[MOE_TIMING_RECORD\]\s+"
    r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([0-9a-fA-F]+)\s+(\d+)"
)
V3_RECORD_RE = re.compile(
    r"\[MTR3\]\s+"
    r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([0-9a-fA-F]+)\s+(\d+)"
)
VERSION_RE = re.compile(r"\[MOE_TIMING_BEGIN\]\s+version=(\d+)")
LEVEL_RE = re.compile(
    r"\[MOE_TIMING_BEGIN\]\s+version=\d+\s+level=(\d+)"
)
END_RE = re.compile(
    r"\[MOE_TIMING_END\]\s+records=(\d+)(?:\s+expected=(\d+))?"
)

STAGE_NAMES = {
    1: "gather_s1",
    2: "load_s1",
    3: "compute_s1",
    4: "prefetch_s2",
    5: "compute_s2",
    6: "load_s3",
    7: "compute_s3",
    8: "prefetch_s4",
    9: "compute_s4",
    10: "store",
    11: "config_s1",
    12: "config_s3",
    13: "cluster_begin",
    14: "cluster_end",
}
RESOURCE_NAMES = {
    0: "none",
    1: "idma",
    2: "xdma",
    3: "dma_both",
    4: "versacore",
    5: "config",
}
FLAG_NAMES = {
    0: "active",
    1: "skipped",
    2: "ctrl_skip",
    3: "invalid_call",
    4: "no_prefetch",
    5: "no_store",
}
STAGE_GROUPS = (
    ("S1", frozenset((2, 3, 11))),
    ("S2", frozenset((4, 5))),
    ("S3", frozenset((6, 7, 12))),
    ("S4", frozenset((8, 9))),
    ("STORE", frozenset((10,))),
)
REQUIRED_SLOT_STAGES = frozenset().union(*(ids for _, ids in STAGE_GROUPS))
SCOPE_BEGIN = 13
SCOPE_END = 14
UINT32_MASK = (1 << 32) - 1


def delta(end: int, start: int) -> int:
    return (end - start) & UINT32_MASK


def format_ticks(cycles: int, cycles_per_tick: float) -> str:
    return f"{cycles / cycles_per_tick:.6f}"


def format_flags(flags: int) -> str:
    names = [name for bit, name in FLAG_NAMES.items() if flags & (1 << bit)]
    return ",".join(names) if names else "none"


def decode_common(
    *,
    node: int | None,
    meta: int,
    task: int,
    start: int,
    total: int,
    resource_offset: int,
    resource_cycles: int,
    peer_wait: int,
    units: int,
    flags: int,
    result: int,
) -> dict[str, int | None]:
    return {
        "node": node,
        "stage": meta & 0xFF,
        "resource": (meta >> 8) & 0xF,
        "cluster": (meta >> 12) & 0xF,
        "core": (meta >> 16) & 0xF,
        "block": (meta >> 20) & 0xFFF,
        "slot": task & 0x3F,
        "expert": (task >> 6) & 0xFF,
        "ntokens": (task >> 14) & 0x3FF,
        "start": start,
        "end": (start + total) & UINT32_MASK,
        "total": total,
        "resource_offset": resource_offset,
        "resource_cycles": resource_cycles,
        "peer_wait": peer_wait,
        "units": units,
        "flags": flags,
        "result": result,
    }


def decode_v2(match: re.Match[str]) -> dict[str, int | None]:
    return decode_common(
        node=None,
        meta=int(match[1], 16),
        task=int(match[2], 16),
        start=int(match[3]),
        total=int(match[4]),
        resource_offset=int(match[5]),
        resource_cycles=int(match[6]),
        peer_wait=int(match[7]),
        units=int(match[8]),
        flags=int(match[9], 16),
        result=int(match[10]),
    )


def decode_v3(match: re.Match[str]) -> dict[str, int | None]:
    return decode_common(
        node=int(match[1], 16),
        meta=int(match[2], 16),
        task=int(match[3], 16),
        start=int(match[4]),
        total=int(match[5]),
        resource_offset=int(match[6]),
        resource_cycles=int(match[7]),
        peer_wait=int(match[8]),
        units=int(match[9]),
        flags=int(match[10], 16),
        result=int(match[11]),
    )


def parse_records(text: str, version: int) -> list[dict[str, int | None]]:
    if version == 2:
        return [decode_v2(match) for match in V2_RECORD_RE.finditer(text)]
    if version == 3:
        return [decode_v3(match) for match in V3_RECORD_RE.finditer(text)]
    raise SystemExit(f"unsupported MOE timing schema version {version}")


def print_v2_rejection(
    records: list[dict[str, int | None]], details: bool
) -> None:
    print("\nVALIDATION status=LEGACY_INCOMPLETE")
    print("  schema v2 cannot produce a valid global, slot-stage, or task report")
    print("  optimized multi-block stages overwrite one scratchpad per block")
    print("  config tasks, cluster scope markers, and DFG node IDs are absent")
    print("  no gap, idle-time, global-time, or stage-efficiency value is emitted")
    if details:
        print("\nLEGACY RAW RECORDS")
        for record in records:
            print(
                f"  C{record['cluster']} core={record['core']} "
                f"slot={record['slot']} eid={record['expert']} "
                f"stage={STAGE_NAMES.get(record['stage'], 'unknown')} "
                f"start={record['start']} total={record['total']}"
            )


def find_scopes(
    records: list[dict[str, int | None]],
) -> tuple[dict[int, dict[str, int]], list[str]]:
    by_cluster: dict[int, list[dict[str, int | None]]] = defaultdict(list)
    for record in records:
        by_cluster[int(record["cluster"])].append(record)

    scopes: dict[int, dict[str, int]] = {}
    errors: list[str] = []
    for cluster, group in sorted(by_cluster.items()):
        begins = [r for r in group if r["stage"] == SCOPE_BEGIN]
        ends = [r for r in group if r["stage"] == SCOPE_END]
        if not begins and not ends:
            continue
        if len(begins) != 1 or len(ends) != 1:
            errors.append(
                f"C{cluster}: expected one cluster_begin and one cluster_end, "
                f"found {len(begins)} and {len(ends)}"
            )
            continue
        begin, end = begins[0], ends[0]
        if begin["core"] != end["core"]:
            errors.append(
                f"C{cluster}: scope markers use different cores "
                f"({begin['core']} vs {end['core']})"
            )
            continue
        epoch = int(begin["end"])
        stop = int(end["start"])
        scopes[cluster] = {
            "epoch": epoch,
            "stop": stop,
            "duration": delta(stop, epoch),
            "core": int(begin["core"]),
            "begin_node": int(begin["node"]),
            "end_node": int(end["node"]),
        }
    if len(scopes) < 2:
        errors.append(
            f"global time requires two valid cluster-local scopes, found {len(scopes)}"
        )
    return scopes, errors


def add_scope_offsets(
    records: list[dict[str, int | None]], scopes: dict[int, dict[str, int]]
) -> list[str]:
    errors: list[str] = []
    for record in records:
        cluster = int(record["cluster"])
        if cluster not in scopes or record["stage"] in (SCOPE_BEGIN, SCOPE_END):
            continue
        scope = scopes[cluster]
        start_offset = delta(int(record["start"]), scope["epoch"])
        end_offset = start_offset + int(record["total"])
        record["start_offset"] = start_offset
        record["end_offset"] = end_offset
        if end_offset > scope["duration"]:
            errors.append(
                f"node {record['node']} in C{cluster} ends outside its cluster scope"
            )
    return errors


def print_global(
    scopes: dict[int, dict[str, int]], cycles_per_tick: float
) -> None:
    print("\nGLOBAL EXECUTION TIME")
    print("  definition=max(per-cluster local mcycle scope); no cross-cluster subtraction")
    for cluster, scope in sorted(scopes.items()):
        print(
            f"  C{cluster} core={scope['core']} begin_node={scope['begin_node']} "
            f"end_node={scope['end_node']} cycles={scope['duration']} "
            f"ticks={format_ticks(scope['duration'], cycles_per_tick)}"
        )
    if len(scopes) >= 2:
        critical_cluster, critical = max(
            scopes.items(), key=lambda item: item[1]["duration"]
        )
        print(
            f"  GLOBAL=max(C-local) C{critical_cluster} cycles={critical['duration']} "
            f"ticks={format_ticks(critical['duration'], cycles_per_tick)}"
        )
    else:
        print("  GLOBAL=UNAVAILABLE")


def interval_union(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    start, end = intervals[0]
    total = 0
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def active_slots(
    records: list[dict[str, int | None]], scopes: dict[int, dict[str, int]]
) -> dict[tuple[int, int], list[dict[str, int | None]]]:
    slots: dict[tuple[int, int], list[dict[str, int | None]]] = defaultdict(list)
    for record in records:
        if (
            int(record["cluster"]) in scopes
            and int(record["ntokens"]) > 0
            and record["stage"] not in (SCOPE_BEGIN, SCOPE_END)
        ):
            slots[(int(record["cluster"]), int(record["slot"]))].append(record)
    return slots


def validate_slots(
    slots: dict[tuple[int, int], list[dict[str, int | None]]]
) -> list[str]:
    errors: list[str] = []
    for (cluster, slot), group in sorted(slots.items()):
        stages = Counter(int(record["stage"]) for record in group)
        missing = sorted(REQUIRED_SLOT_STAGES - stages.keys())
        if missing:
            errors.append(
                f"C{cluster} slot={slot}: missing task records "
                + ",".join(STAGE_NAMES[stage] for stage in missing)
            )
        if slot == 0 and stages[1] == 0:
            errors.append(f"C{cluster} slot=0: missing gather_s1 record")
    return errors


def validate_timing_level(
    records: list[dict[str, int | None]], timing_level: int
) -> list[str]:
    if timing_level not in (1, 2):
        return [f"unsupported MOE runtime timing level {timing_level}"]
    if timing_level == 1:
        unexpected = [
            record
            for record in records
            if int(record["stage"]) not in (SCOPE_BEGIN, SCOPE_END)
        ]
        if unexpected:
            return [
                f"level 1 capture contains {len(unexpected)} non-scope records"
            ]
    return []


def print_slot_and_stage_report(
    slots: dict[tuple[int, int], list[dict[str, int | None]]],
    cycles_per_tick: float,
) -> None:
    print("\nSLOT AND STAGE WALL TIMES")
    print("  stage wall=max(node end)-min(node start) inside one cluster clock domain")
    print("  stages may overlap; stage wall values must not be summed")
    for (cluster, slot), group in sorted(slots.items()):
        start = min(int(record["start_offset"]) for record in group)
        end = max(int(record["end_offset"]) for record in group)
        covered = interval_union(
            [
                (int(record["start_offset"]), int(record["end_offset"]))
                for record in group
            ]
        )
        first = group[0]
        print(
            f"  C{cluster} slot={slot:2d} eid={first['expert']:2d} "
            f"ntok={first['ntokens']:3d} task_cycles={end - start:8d} "
            f"task_ticks={format_ticks(end - start, cycles_per_tick)} "
            f"recorded_union={covered:8d}"
        )
        if slot == 0:
            gather = [record for record in group if record["stage"] == 1]
            if gather:
                g_start = min(int(record["start_offset"]) for record in gather)
                g_end = max(int(record["end_offset"]) for record in gather)
                print(
                    f"      INPUT  start={g_start:8d} end={g_end:8d} "
                    f"wall={g_end - g_start:8d} "
                    f"ticks={format_ticks(g_end - g_start, cycles_per_tick)}"
                )
        for name, stage_ids in STAGE_GROUPS:
            stage_records = [r for r in group if r["stage"] in stage_ids]
            if not stage_records:
                print(f"      {name:6s} MISSING")
                continue
            stage_start = min(int(r["start_offset"]) for r in stage_records)
            stage_end = max(int(r["end_offset"]) for r in stage_records)
            print(
                f"      {name:6s} start={stage_start:8d} end={stage_end:8d} "
                f"wall={stage_end - stage_start:8d} "
                f"ticks={format_ticks(stage_end - stage_start, cycles_per_tick)}"
            )


def print_task_details(
    slots: dict[tuple[int, int], list[dict[str, int | None]]],
    cycles_per_tick: float,
) -> None:
    print("\nDFG TASK TIMES")
    for (cluster, slot), group in sorted(slots.items()):
        print(f"  C{cluster} slot={slot}")
        for record in sorted(
            group,
            key=lambda item: (int(item["start_offset"]), int(item["node"])),
        ):
            resource = RESOURCE_NAMES.get(int(record["resource"]), "unknown")
            print(
                f"    node={record['node']:4d} core={record['core']} "
                f"stage={STAGE_NAMES.get(int(record['stage']), 'unknown'):12s} "
                f"start={record['start_offset']:8d} end={record['end_offset']:8d} "
                f"cycles={record['total']:7d} "
                f"ticks={format_ticks(int(record['total']), cycles_per_tick)} "
                f"resource={resource:10s} resource_span={record['resource_cycles']:7d} "
                f"wait={record['peer_wait']:6d} flags={format_flags(int(record['flags']))} "
                f"rc={record['result']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="UART or simulator log")
    parser.add_argument(
        "--details", action="store_true", help="print every DFG task record"
    )
    parser.add_argument(
        "--cycles-per-tick",
        type=float,
        default=8192.0,
        help="cycles represented by one structural-model tick (default: 8192)",
    )
    # Accepted for command-line compatibility with the removed MAC-efficiency report.
    parser.add_argument("--params", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--peak-mac-per-cluster-cc", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--individual-cluster-count", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cycles_per_tick <= 0:
        raise SystemExit("--cycles-per-tick must be positive")

    text = args.log.read_text(errors="replace")
    version_match = VERSION_RE.search(text)
    if version_match is None:
        raise SystemExit("no [MOE_TIMING_BEGIN] schema version found")
    version = int(version_match[1])
    level_match = LEVEL_RE.search(text)
    # Schema-v3 logs produced before timing levels were introduced are full
    # captures and therefore have level-2 semantics.
    timing_level = int(level_match[1]) if level_match else 2
    records = parse_records(text, version)
    if not records:
        raise SystemExit(f"no schema-v{version} timing records found")
    end_match = END_RE.search(text)
    reported = int(end_match[1]) if end_match else None
    expected_tasks = (
        int(end_match[2]) if end_match and end_match[2] is not None else None
    )
    uart_complete = reported is not None and reported == len(records)
    task_capture_complete = (
        version == 3
        and reported is not None
        and expected_tasks is not None
        and reported == expected_tasks
    )
    reported_text = str(reported) if reported is not None else "missing-end-marker"
    expected_text = str(expected_tasks) if expected_tasks is not None else "unavailable"
    print(
        f"TIMING RECORDS schema=v{version} level={timing_level} "
        f"parsed={len(records)} "
        f"printed={reported_text} expected_tasks={expected_text} "
        f"uart_complete={'yes' if uart_complete else 'no'} "
        f"task_capture_complete={'yes' if task_capture_complete else 'no'}"
    )

    if version == 2:
        print_v2_rejection(records, args.details)
        return

    errors = validate_timing_level(records, timing_level)
    scopes, scope_errors = find_scopes(records)
    errors.extend(scope_errors)
    errors.extend(add_scope_offsets(records, scopes))
    slots = active_slots(records, scopes) if timing_level == 2 else {}
    if timing_level == 2:
        errors.extend(validate_slots(slots))
    if not uart_complete:
        errors.append("UART capture is incomplete")
    if expected_tasks is None:
        errors.append("schema-v3 end marker has no expected task count")
    elif not task_capture_complete:
        errors.append(
            f"only {reported} of {expected_tasks} profiled DFG tasks wrote records"
        )

    print(f"\nVALIDATION status={'VALID' if not errors else 'INCOMPLETE'}")
    print("  clock_domain=cluster-local mcycle")
    print("  cross_cluster_timestamp_subtraction=forbidden")
    for error in errors:
        print(f"  ERROR: {error}")

    print_global(scopes, args.cycles_per_tick)
    if timing_level == 1:
        print("\nDETAIL TIMING disabled at MOE_RUNTIME_TIMING=1")
        print("  rebuild with MOE_RUNTIME_TIMING=2 for slot/stage/task records")
        return
    if slots:
        print_slot_and_stage_report(slots, args.cycles_per_tick)
        if args.details:
            print_task_details(slots, args.cycles_per_tick)
    else:
        print("\nSLOT AND STAGE WALL TIMES unavailable: no active slot records")


if __name__ == "__main__":
    main()
