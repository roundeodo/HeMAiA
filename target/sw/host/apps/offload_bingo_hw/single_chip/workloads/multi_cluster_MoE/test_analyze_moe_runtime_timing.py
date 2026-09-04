#!/usr/bin/env python3
"""Focused tests for the schema-v3 MoE runtime timing analyzer."""

from __future__ import annotations

import contextlib
import io
import unittest

import analyze_moe_runtime_timing as timing


def make_record(
    node: int,
    cluster: int,
    stage: int,
    start: int,
    total: int,
    *,
    slot: int = 0,
    expert: int = 1,
    ntokens: int = 8,
) -> dict[str, int | None]:
    meta = stage | (cluster << 12)
    task = slot | (expert << 6) | (ntokens << 14)
    return timing.decode_common(
        node=node,
        meta=meta,
        task=task,
        start=start,
        total=total,
        resource_offset=0,
        resource_cycles=0,
        peer_wait=0,
        units=0,
        flags=0,
        result=0,
    )


def make_cluster(
    cluster: int, epoch: int, duration: int, first_node: int
) -> list[dict[str, int | None]]:
    records = [
        make_record(
            first_node,
            cluster,
            timing.SCOPE_BEGIN,
            (epoch - 3) & timing.UINT32_MASK,
            3,
        )
    ]
    for offset, stage in enumerate((1, 2, 11, 3, 4, 5, 6, 12, 7, 8, 9, 10), 1):
        records.append(
            make_record(
                first_node + offset,
                cluster,
                stage,
                (epoch + offset) & timing.UINT32_MASK,
                1,
            )
        )
    records.append(
        make_record(
            first_node + 13,
            cluster,
            timing.SCOPE_END,
            (epoch + duration) & timing.UINT32_MASK,
            2,
        )
    )
    return records


class TimingAnalyzerTest(unittest.TestCase):
    def test_timing_level_is_parsed_from_begin_marker(self) -> None:
        match = timing.LEVEL_RE.search("[MOE_TIMING_BEGIN] version=3 level=1")
        self.assertIsNotNone(match)
        self.assertEqual("1", match[1])

    def test_v3_end_marker_reports_printed_and_expected_counts(self) -> None:
        match = timing.END_RE.search("[MOE_TIMING_END] records=479 expected=479")
        self.assertIsNotNone(match)
        self.assertEqual("479", match[1])
        self.assertEqual("479", match[2])

    def test_global_uses_max_cluster_local_span_with_wrap(self) -> None:
        records = make_cluster(2, 0xFFFFFFF8, 56, 100)
        records += make_cluster(3, 1000, 45, 200)

        scopes, errors = timing.find_scopes(records)
        errors += timing.add_scope_offsets(records, scopes)

        self.assertEqual([], errors)
        self.assertEqual(56, scopes[2]["duration"])
        self.assertEqual(45, scopes[3]["duration"])
        self.assertEqual(56, max(scope["duration"] for scope in scopes.values()))

    def test_level_one_accepts_only_four_scope_records(self) -> None:
        records = [
            make_record(1, 2, timing.SCOPE_BEGIN, 100, 2),
            make_record(2, 2, timing.SCOPE_END, 202, 2),
            make_record(3, 3, timing.SCOPE_BEGIN, 500, 2),
            make_record(4, 3, timing.SCOPE_END, 592, 2),
        ]
        self.assertEqual([], timing.validate_timing_level(records, 1))
        scopes, errors = timing.find_scopes(records)
        self.assertEqual([], errors)
        self.assertEqual(100, scopes[2]["duration"])
        self.assertEqual(90, scopes[3]["duration"])
        self.assertEqual({}, timing.active_slots(records, scopes))

    def test_level_one_accepts_full_workload_scope_records(self) -> None:
        records = [
            make_record(1, 3, timing.WORKLOAD_BEGIN, 100, 2),
            make_record(2, 3, timing.WORKLOAD_END, 1102, 2),
            make_record(3, 0, timing.SHARED_BEGIN, 200, 2),
            make_record(4, 0, timing.SHARED_END, 702, 2),
            make_record(5, 1, timing.SHARED_BEGIN, 300, 2),
            make_record(6, 1, timing.SHARED_END, 752, 2),
            make_record(7, 2, timing.SCOPE_BEGIN, 400, 2),
            make_record(8, 2, timing.SCOPE_END, 902, 2),
            make_record(9, 3, timing.SCOPE_BEGIN, 500, 2),
            make_record(10, 3, timing.SCOPE_END, 952, 2),
        ]
        self.assertEqual([], timing.validate_timing_level(records, 1))
        workload, errors = timing.find_named_scopes(
            records, timing.WORKLOAD_BEGIN, timing.WORKLOAD_END, "overall"
        )
        shared, shared_errors = timing.find_named_scopes(
            records, timing.SHARED_BEGIN, timing.SHARED_END, "shared"
        )
        self.assertEqual([], errors + shared_errors)
        self.assertEqual(1000, workload[3]["duration"])
        self.assertEqual(500, shared[0]["duration"])
        self.assertEqual(450, shared[1]["duration"])

    def test_level_one_accepts_and_reports_all_m8_comparison_scopes(self) -> None:
        records = []
        expected = {}
        for index, (begin, end, label) in enumerate(timing.M8_SCOPE_PAIRS):
            start = 1000 + index * 1000
            duration = 100 + index * 10
            records += [
                make_record(20 + 2 * index, 0, begin, start, 2),
                make_record(21 + 2 * index, 0, end, start + 2 + duration, 2),
            ]
            expected[label] = duration
        self.assertEqual([], timing.validate_timing_level(records, 1))

        scopes_by_label = {}
        errors = []
        for begin, end, label in timing.M8_SCOPE_PAIRS:
            scopes, scope_errors = timing.find_named_scopes(records, begin, end, label)
            scopes_by_label[label] = scopes
            errors += scope_errors
            self.assertEqual(expected[label], scopes[0]["duration"])
        self.assertEqual([], errors)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            timing.print_m8_comparison_scopes(scopes_by_label, 1.0)
        report = output.getvalue()
        self.assertIn("FIXED_A_A C0 cycles=100", report)
        self.assertIn("FIXED_B_B C0 cycles=110", report)
        self.assertIn("FIXED_C_C C0 cycles=120", report)
        self.assertIn("DISTILLED C0 cycles=130", report)

    def test_level_one_accepts_and_reports_all_m32_comparison_scopes(self) -> None:
        records = []
        scopes_by_label = {}
        for index, (begin, end, label) in enumerate(timing.M32_SCOPE_PAIRS):
            start = 5000 + index * 1000
            duration = 200 + index * 10
            records += [
                make_record(28 + 2 * index, 0, begin, start, 2),
                make_record(29 + 2 * index, 0, end, start + 2 + duration, 2),
            ]
        self.assertEqual([], timing.validate_timing_level(records, 1))
        for begin, end, label in timing.M32_SCOPE_PAIRS:
            scopes, errors = timing.find_named_scopes(records, begin, end, label)
            self.assertEqual([], errors)
            scopes_by_label[label] = scopes

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            timing.print_comparison_scopes(
                scopes_by_label, 1.0, "M32", timing.M32_SCOPE_PAIRS
            )
        report = output.getvalue()
        self.assertIn("M32 FIXED-SHAPE VS DISTILLED", report)
        self.assertIn("FIXED_A_A C0 cycles=200", report)
        self.assertIn("DISTILLED C0 cycles=230", report)

    def test_router_shared_mode_reports_boundary_excluding_marker_cost(self) -> None:
        records = [
            make_record(1, 3, timing.WORKLOAD_BEGIN, 100, 2),
            make_record(2, 3, timing.SHARED_BEGIN, 250, 3),
            make_record(3, 3, timing.SHARED_END, 700, 2),
            make_record(4, 3, timing.WORKLOAD_END, 704, 2),
        ]
        workload, errors = timing.find_named_scopes(
            records, timing.WORKLOAD_BEGIN, timing.WORKLOAD_END, "workload"
        )
        shared, shared_errors = timing.find_named_scopes(
            records, timing.SHARED_BEGIN, timing.SHARED_END, "shared"
        )
        self.assertEqual([], errors + shared_errors)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            timing.print_full_workload_scopes(workload, shared, 1.0)
        report = output.getvalue()
        self.assertIn("ROUTER C3 cycles=148", report)
        self.assertIn("SHARED C3 cycles=447", report)
        self.assertIn("ROUTER_PLUS_SHARED cycles=595", report)
        self.assertIn("boundary_marker_cycles=7", report)

    def test_scheduler_routed_scope_reports_continuous_linked_time(self) -> None:
        records = [
            make_record(1, 3, timing.SCHEDULER_ROUTED_BEGIN, 100, 3),
            make_record(2, 3, timing.SCHEDULER_ROUTED_END, 1103, 2),
        ]
        scopes, errors = timing.find_named_scopes(
            records,
            timing.SCHEDULER_ROUTED_BEGIN,
            timing.SCHEDULER_ROUTED_END,
            "scheduler+routed linked",
        )
        self.assertEqual([], errors)
        self.assertEqual(1000, scopes[3]["duration"])
        self.assertEqual([], timing.validate_timing_level(records, 1))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            timing.print_scheduler_routed_scope(scopes, 1.0, 100, 850)
        report = output.getvalue()
        self.assertIn("SCHEDULER_ROUTED C3 cycles=1000", report)
        self.assertIn("MoEPrepare + MoEExecute + routed expert chains", report)
        self.assertIn("SEPARATE_SUM cycles=950", report)
        self.assertIn("LINKED_MINUS_SEPARATE cycles=50", report)

    def test_level_one_rejects_detail_records(self) -> None:
        records = [make_record(1, 2, 2, 100, 2)]
        self.assertEqual(
            ["level 1 capture contains 1 non-scope records"],
            timing.validate_timing_level(records, 1),
        )

    def test_all_slot_stages_and_task_offsets_are_complete(self) -> None:
        records = make_cluster(2, 500, 80, 100)
        records += make_cluster(3, 9000, 70, 200)
        scopes, errors = timing.find_scopes(records)
        errors += timing.add_scope_offsets(records, scopes)
        slots = timing.active_slots(records, scopes)
        errors += timing.validate_slots(slots)

        self.assertEqual([], errors)
        self.assertEqual({(2, 0), (3, 0)}, set(slots))
        self.assertEqual(1, min(record["start_offset"] for record in slots[(2, 0)]))

    def test_fused_s3_s4pf_record_satisfies_both_stage_requirements(self) -> None:
        records = make_cluster(2, 500, 80, 100)
        records = [record for record in records if record["stage"] not in (6, 8)]
        records.append(make_record(200, 2, 15, 520, 12))
        records += make_cluster(3, 900, 70, 300)

        scopes, errors = timing.find_scopes(records)
        errors += timing.add_scope_offsets(records, scopes)
        slots = timing.active_slots(records, scopes)
        errors += timing.validate_slots(slots)

        self.assertEqual([], errors)

    def test_valid_report_does_not_infer_gap_or_idle(self) -> None:
        records = make_cluster(2, 100, 80, 100)
        records += make_cluster(3, 5000, 70, 200)
        scopes, errors = timing.find_scopes(records)
        errors += timing.add_scope_offsets(records, scopes)
        slots = timing.active_slots(records, scopes)
        errors += timing.validate_slots(slots)
        self.assertEqual([], errors)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            timing.print_global(scopes, 1.0)
            timing.print_slot_and_stage_report(slots, 1.0)

        report = output.getvalue().lower()
        self.assertNotIn("gap", report)
        self.assertNotIn("idle", report)
        self.assertIn("global=max(c-local) c2 cycles=80", report)


if __name__ == "__main__":
    unittest.main()
