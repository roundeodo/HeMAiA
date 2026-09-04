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
LEVEL_RE = re.compile(r"\[MOE_TIMING_BEGIN\]\s+version=\d+\s+level=(\d+)")
END_RE = re.compile(r"\[MOE_TIMING_END\]\s+records=(\d+)(?:\s+expected=(\d+))?")

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
    15: "load_s3_prefetch_s4",
    16: "workload_begin",
    17: "shared_begin",
    18: "shared_end",
    19: "workload_end",
    20: "m8_fixed_a_begin",
    21: "m8_fixed_a_end",
    22: "m8_fixed_b_begin",
    23: "m8_fixed_b_end",
    24: "m8_fixed_c_begin",
    25: "m8_fixed_c_end",
    26: "m8_distilled_begin",
    27: "m8_distilled_end",
    28: "m32_fixed_a_begin",
    29: "m32_fixed_a_end",
    30: "m32_fixed_b_begin",
    31: "m32_fixed_b_end",
    32: "m32_fixed_c_begin",
    33: "m32_fixed_c_end",
    34: "m32_distilled_begin",
    35: "m32_distilled_end",
    36: "scheduler_routed_begin",
    37: "scheduler_routed_end",
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
    ("S3+S4PF", frozenset((15,))),
)
REQUIRED_SLOT_STAGES = frozenset((2, 3, 4, 5, 7, 9, 10, 11, 12))
REQUIRED_STAGE_ALTERNATIVES = (
    (frozenset((6, 15)), "load_s3|load_s3_prefetch_s4"),
    (frozenset((8, 15)), "prefetch_s4|load_s3_prefetch_s4"),
)
SCOPE_BEGIN = 13
SCOPE_END = 14
WORKLOAD_BEGIN = 16
SHARED_BEGIN = 17
SHARED_END = 18
WORKLOAD_END = 19
SCHEDULER_ROUTED_BEGIN = 36
SCHEDULER_ROUTED_END = 37
M8_SCOPE_PAIRS = (
    (20, 21, "FIXED_A_A"),
    (22, 23, "FIXED_B_B"),
    (24, 25, "FIXED_C_C"),
    (26, 27, "DISTILLED"),
)
M32_SCOPE_PAIRS = (
    (28, 29, "FIXED_A_A"),
    (30, 31, "FIXED_B_B"),
    (32, 33, "FIXED_C_C"),
    (34, 35, "DISTILLED"),
)
COMPARISON_SCOPE_GROUPS = (
    ("M8", M8_SCOPE_PAIRS),
    ("M32", M32_SCOPE_PAIRS),
)
LEVEL1_SCOPE_STAGES = frozenset(
    (
        SCOPE_BEGIN,
        SCOPE_END,
        WORKLOAD_BEGIN,
        SHARED_BEGIN,
        SHARED_END,
        WORKLOAD_END,
        SCHEDULER_ROUTED_BEGIN,
        SCHEDULER_ROUTED_END,
        *(stage for pair in M8_SCOPE_PAIRS for stage in pair[:2]),
        *(stage for pair in M32_SCOPE_PAIRS for stage in pair[:2]),
    )
)
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


def print_v2_rejection(records: list[dict[str, int | None]], details: bool) -> None:
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


def find_named_scopes(
    records: list[dict[str, int | None]],
    begin_stage: int,
    end_stage: int,
    label: str,
) -> tuple[dict[int, dict[str, int]], list[str]]:
    by_cluster: dict[int, list[dict[str, int | None]]] = defaultdict(list)
    for record in records:
        if int(record["stage"]) in (begin_stage, end_stage):
            by_cluster[int(record["cluster"])].append(record)

    scopes: dict[int, dict[str, int]] = {}
    errors: list[str] = []
    for cluster, group in sorted(by_cluster.items()):
        begins = [r for r in group if int(r["stage"]) == begin_stage]
        ends = [r for r in group if int(r["stage"]) == end_stage]
        if len(begins) != 1 or len(ends) != 1:
            errors.append(
                f"{label} C{cluster}: expected one begin/end pair, "
                f"found {len(begins)} and {len(ends)}"
            )
            continue
        begin, end = begins[0], ends[0]
        if begin["core"] != end["core"]:
            errors.append(f"{label} C{cluster}: begin/end use different cores")
            continue
        epoch = int(begin["end"])
        stop = int(end["start"])
        scopes[cluster] = {
            "begin_start": int(begin["start"]),
            "epoch": epoch,
            "stop": stop,
            "duration": delta(stop, epoch),
            "core": int(begin["core"]),
            "begin_node": int(begin["node"]),
            "end_node": int(end["node"]),
        }
    if not scopes:
        errors.append(f"{label}: no valid scope found")
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


def print_global(scopes: dict[int, dict[str, int]], cycles_per_tick: float) -> None:
    print("\nROUTED EXPERT PIPELINE TIME")
    print("  definition=max(C2,C3 local scopes); no cross-cluster subtraction")
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
            f"  ROUTED=max(C-local) C{critical_cluster} cycles={critical['duration']} "
            f"ticks={format_ticks(critical['duration'], cycles_per_tick)}"
        )
        # Keep the legacy parser key while making the routed-only scope clear.
        print(
            f"  GLOBAL=max(C-local) C{critical_cluster} cycles={critical['duration']} "
            f"ticks={format_ticks(critical['duration'], cycles_per_tick)}"
        )
    else:
        print("  ROUTED=UNAVAILABLE")


def print_full_workload_scopes(
    workload_scopes: dict[int, dict[str, int]],
    shared_scopes: dict[int, dict[str, int]],
    cycles_per_tick: float,
) -> None:
    print("\nROUTER + SHARED MEASUREMENT")
    common_clusters = sorted(set(workload_scopes) & set(shared_scopes))
    if not common_clusters:
        print("  ROUTER_SHARED=UNAVAILABLE")
        return
    cluster = common_clusters[0]
    workload = workload_scopes[cluster]
    shared = shared_scopes[cluster]
    router_cycles = delta(shared["begin_start"], workload["epoch"])
    shared_cycles = shared["duration"]
    combined_cycles = router_cycles + shared_cycles
    marker_overhead = workload["duration"] - combined_cycles
    print(
        f"  ROUTER C{cluster} cycles={router_cycles} "
        f"ticks={format_ticks(router_cycles, cycles_per_tick)}"
    )
    print(
        f"  SHARED C{cluster} cycles={shared_cycles} "
        f"ticks={format_ticks(shared_cycles, cycles_per_tick)}"
    )
    print(
        f"  ROUTER_PLUS_SHARED cycles={combined_cycles} "
        f"ticks={format_ticks(combined_cycles, cycles_per_tick)}"
    )
    print(
        f"  RAW_WORKLOAD_SCOPE cycles={workload['duration']} "
        f"boundary_marker_cycles={marker_overhead}"
    )


def print_comparison_scopes(
    scopes_by_label: dict[str, dict[int, dict[str, int]]],
    cycles_per_tick: float,
    token_label: str,
    scope_pairs,
) -> None:
    print(f"\n{token_label} FIXED-SHAPE VS DISTILLED")
    print("  definition=end_marker.start - begin_marker.end")
    for _begin, _end, label in scope_pairs:
        scopes = scopes_by_label.get(label, {})
        if not scopes:
            print(f"  {label}=UNAVAILABLE")
            continue
        cluster, scope = next(iter(sorted(scopes.items())))
        print(
            f"  {label} C{cluster} cycles={scope['duration']} "
            f"ticks={format_ticks(scope['duration'], cycles_per_tick)}"
        )


def print_scheduler_routed_scope(
    scopes: dict[int, dict[str, int]],
    cycles_per_tick: float,
    scheduler_only_cycles: int | None = None,
    routed_only_cycles: int | None = None,
) -> None:
    print("\nSCHEDULER + ROUTED EXPERT LINKED TIME")
    print("  definition=end_marker.start - begin_marker.end")
    print("  includes=Bingo dispatch + MoEPrepare + MoEExecute + routed expert chains")
    if not scopes:
        print("  SCHEDULER_ROUTED=UNAVAILABLE")
        return
    cluster, scope = next(iter(sorted(scopes.items())))
    print(
        f"  SCHEDULER_ROUTED C{cluster} cycles={scope['duration']} "
        f"ticks={format_ticks(scope['duration'], cycles_per_tick)}"
    )
    if scheduler_only_cycles is not None and routed_only_cycles is not None:
        separate_sum = scheduler_only_cycles + routed_only_cycles
        difference = int(scope["duration"]) - separate_sum
        percent = (100.0 * difference / separate_sum) if separate_sum else 0.0
        print(f"  SEPARATE_SUM cycles={separate_sum}")
        print(
            f"  LINKED_MINUS_SEPARATE cycles={difference} "
            f"percent_of_separate={percent:.3f}%"
        )


def print_m8_comparison_scopes(
    scopes_by_label: dict[str, dict[int, dict[str, int]]],
    cycles_per_tick: float,
) -> None:
    print_comparison_scopes(scopes_by_label, cycles_per_tick, "M8", M8_SCOPE_PAIRS)


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
    slots: dict[tuple[int, int], list[dict[str, int | None]]],
) -> list[str]:
    errors: list[str] = []
    for (cluster, slot), group in sorted(slots.items()):
        stages = Counter(int(record["stage"]) for record in group)
        missing = sorted(REQUIRED_SLOT_STAGES - stages.keys())
        missing_alternatives = [
            name
            for alternatives, name in REQUIRED_STAGE_ALTERNATIVES
            if alternatives.isdisjoint(stages)
        ]
        if missing or missing_alternatives:
            errors.append(
                f"C{cluster} slot={slot}: missing task records "
                + ",".join(
                    [STAGE_NAMES[stage] for stage in missing] + missing_alternatives
                )
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
            if int(record["stage"]) not in LEVEL1_SCOPE_STAGES
        ]
        if unexpected:
            return [f"level 1 capture contains {len(unexpected)} non-scope records"]
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
    parser.add_argument(
        "--scheduler-only-cycles",
        type=int,
        help="previous scheduler-only continuous-interval result",
    )
    parser.add_argument(
        "--routed-only-cycles",
        type=int,
        help="previous routed-only continuous-interval result",
    )
    args = parser.parse_args()
    if args.cycles_per_tick <= 0:
        raise SystemExit("--cycles-per-tick must be positive")
    if (args.scheduler_only_cycles is None) != (args.routed_only_cycles is None):
        raise SystemExit(
            "--scheduler-only-cycles and --routed-only-cycles must be provided together"
        )

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
    stages = {int(record["stage"]) for record in records}
    scopes = {}
    if stages & {SCOPE_BEGIN, SCOPE_END}:
        scopes, scope_errors = find_scopes(records)
        errors.extend(scope_errors)
        errors.extend(add_scope_offsets(records, scopes))
    workload_scopes = {}
    shared_scopes = {}
    if stages & {WORKLOAD_BEGIN, WORKLOAD_END, SHARED_BEGIN, SHARED_END}:
        workload_scopes, workload_errors = find_named_scopes(
            records, WORKLOAD_BEGIN, WORKLOAD_END, "router/shared workload"
        )
        shared_scopes, shared_errors = find_named_scopes(
            records, SHARED_BEGIN, SHARED_END, "shared branch"
        )
        errors.extend(workload_errors)
        errors.extend(shared_errors)
    comparison_scopes = {}
    scheduler_routed_scopes = {}
    if stages & {SCHEDULER_ROUTED_BEGIN, SCHEDULER_ROUTED_END}:
        scheduler_routed_scopes, scheduler_routed_errors = find_named_scopes(
            records,
            SCHEDULER_ROUTED_BEGIN,
            SCHEDULER_ROUTED_END,
            "scheduler+routed linked",
        )
        errors.extend(scheduler_routed_errors)
    for token_label, scope_pairs in COMPARISON_SCOPE_GROUPS:
        comparison_stages = {stage for pair in scope_pairs for stage in pair[:2]}
        if stages & comparison_stages:
            scopes_by_label = {}
            for begin_stage, end_stage, label in scope_pairs:
                run_scopes, run_errors = find_named_scopes(
                    records, begin_stage, end_stage, f"{token_label} {label}"
                )
                scopes_by_label[label] = run_scopes
                errors.extend(run_errors)
            comparison_scopes[token_label] = (scope_pairs, scopes_by_label)
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

    if workload_scopes or shared_scopes:
        print_full_workload_scopes(workload_scopes, shared_scopes, args.cycles_per_tick)
    for token_label, (scope_pairs, scopes_by_label) in comparison_scopes.items():
        print_comparison_scopes(
            scopes_by_label, args.cycles_per_tick, token_label, scope_pairs
        )
    if scheduler_routed_scopes:
        print_scheduler_routed_scope(
            scheduler_routed_scopes,
            args.cycles_per_tick,
            args.scheduler_only_cycles,
            args.routed_only_cycles,
        )
    if scopes:
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
