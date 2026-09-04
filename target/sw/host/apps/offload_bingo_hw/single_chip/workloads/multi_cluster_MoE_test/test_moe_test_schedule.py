#!/usr/bin/env python3
"""Focused schedule-contract tests for the FPGA showcase profiles."""

import pathlib
import sys
import unittest

import moe_test_schedule as schedule


class MoeTestScheduleTest(unittest.TestCase):
    def test_m32_distribution_is_the_production_routing_case(self) -> None:
        case = schedule.get_routing_case(32, schedule.EXPERT_COUNT)
        self.assertEqual("m32_moe_style", case.name)
        self.assertEqual(case.expected_counts, schedule.M32_SCALED_SKEW_COUNTS)
        self.assertEqual(case.token_major_top2, schedule.M32_SCALED_SKEW_TOP2)
        self.assertEqual(64, sum(schedule.M32_SCALED_SKEW_COUNTS))
        self.assertEqual(29, sum(c > 0 for c in schedule.M32_SCALED_SKEW_COUNTS))
        self.assertEqual(
            (8, 5, 4, 3) + (2,) * 14 + (1,) * 10 + (0,) * 35 + (6,),
            schedule.M32_SCALED_SKEW_COUNTS,
        )

    def test_m32_comparison_contains_four_audited_runs(self) -> None:
        schedules = schedule.build_m32_comparison_schedules()
        self.assertEqual(schedule.M32_COMPARISON_RUN_PROFILES, tuple(schedules))
        expected = {
            schedule.M32_FIXED_A_PROFILE: 180,
            schedule.M32_FIXED_B_PROFILE: 99,
            schedule.M32_FIXED_C_PROFILE: 108,
            schedule.M32_DISTILLED_PROFILE: 87,
        }
        for profile, queues in schedules.items():
            with self.subTest(profile=profile):
                audit = schedule.audit_m32_comparison_schedule(queues, profile)
                self.assertEqual(expected[profile], audit["makespan_ticks"])
                self.assertEqual(64, audit["routed_tokens"])
                self.assertEqual(29, audit["active_experts"])

    def test_m32_comparison_runs_are_independently_selectable(self) -> None:
        schedules = schedule.build_m32_comparison_schedules()
        self.assertNotIn(schedule.M32_COMPARISON_PROFILE, schedule.SCHEDULE_PROFILES)
        for profile in schedule.M32_COMPARISON_RUN_PROFILES:
            with self.subTest(profile=profile):
                self.assertIn(profile, schedule.SCHEDULE_PROFILES)
                self.assertEqual(
                    schedules[profile], schedule.build_schedule_profile(profile)
                )

    def test_m32_distilled_stream_matches_normative_model(self) -> None:
        model_dir = pathlib.Path(__file__).resolve().parents[9] / "schedule_algorithm"
        if not model_dir.is_dir():
            self.skipTest("normative schedule_algorithm directory is unavailable")
        sys.path.insert(0, str(model_dir))
        try:
            from scheduler_rtl_distilled_policy import schedule as model_schedule
            from scheduler_rtl_distilled_types import TICK_CC
        finally:
            sys.path.pop(0)

        distribution = {
            eid: ntokens
            for eid, ntokens in enumerate(schedule.M32_SCALED_SKEW_COUNTS)
            if ntokens
        }
        result = model_schedule(
            distribution,
            initial_cache_c2=-1,
            initial_cache_c3=-1,
        )
        shape_id = {"A": schedule.SHAPE_A, "B": schedule.SHAPE_B, "C": schedule.SHAPE_C}
        model_queues = {"c0": [], "c1": []}
        for step in result.steps:
            self.assertFalse(step.s4pf_actions)
            action = step.action
            for cluster_name, prefix in (("c0", "c2"), ("c1", "c3")):
                expert_id = getattr(action, f"{prefix}_eid")
                if expert_id < 0:
                    continue
                model_queues[cluster_name].append(
                    (
                        expert_id,
                        getattr(action, f"{prefix}_ntok"),
                        getattr(action, f"{prefix}_start") // TICK_CC,
                        shape_id[getattr(action, f"{prefix}_shape_s1").name[0]],
                        shape_id[getattr(action, f"{prefix}_shape_s3").name[0]],
                        int(getattr(action, f"{prefix}_dma_s1")),
                        int(getattr(action, f"{prefix}_dma_s3")),
                        int(getattr(action, f"{prefix}_s2pf_dma")),
                        bool(getattr(action, f"{prefix}_s1_cached")),
                        bool(getattr(action, f"{prefix}_s3_cached")),
                    )
                )

        frozen = schedule.build_m32_distilled_schedule()
        for cluster_name in ("c0", "c1"):
            frozen_signature = [
                (
                    slot.expert_id,
                    slot.ntokens,
                    slot.reference_start_tick,
                    slot.s1_shape,
                    slot.s3_shape,
                    slot.s1_dma,
                    slot.s3_dma,
                    slot.s2_prefetch_dma,
                    slot.skip_s1,
                    slot.skip_s3,
                )
                for slot in frozen[cluster_name]
            ]
            self.assertEqual(model_queues[cluster_name], frozen_signature)
        self.assertEqual(
            schedule.M32_DISTILLED_EXPECTED_MAKESPAN_TICKS,
            result.makespan_cc // TICK_CC,
        )

    def test_m8_distribution_preserves_production_top2_order(self) -> None:
        self.assertEqual(
            (
                (0, 63),
                (0, 63),
                (0, 63),
                (0, 63),
                (0, 1),
                (0, 1),
                (63, 1),
                (63, 1),
            ),
            schedule.M8_4_2_2_TOP2,
        )
        self.assertEqual(
            (6, 4) + (0,) * 61 + (6,),
            schedule.M8_4_2_2_COUNTS,
        )

    def test_m8_comparison_contains_four_audited_runs(self) -> None:
        schedules = schedule.build_m8_comparison_schedules()
        self.assertEqual(
            schedule.M8_COMPARISON_RUN_PROFILES,
            tuple(schedules),
        )
        expected = {
            schedule.M8_FIXED_A_PROFILE: 24,
            schedule.M8_FIXED_B_PROFILE: 15,
            schedule.M8_FIXED_C_PROFILE: 21,
            schedule.M8_DISTILLED_PROFILE: 15,
        }
        for profile, queues in schedules.items():
            with self.subTest(profile=profile):
                audit = schedule.audit_m8_comparison_schedule(queues, profile)
                self.assertEqual(expected[profile], audit["makespan_ticks"])
                self.assertEqual(16, audit["routed_tokens"])

    def test_m8_distilled_stream_matches_normative_model_result(self) -> None:
        queues = schedule.build_m8_distilled_schedule()
        self.assertEqual((0, 1), tuple(s.expert_id for s in queues["c0"]))
        self.assertEqual((63,), tuple(s.expert_id for s in queues["c1"]))
        self.assertEqual(
            (
                (schedule.SHAPE_B, schedule.SHAPE_B),
                (schedule.SHAPE_C, schedule.SHAPE_C),
            ),
            tuple((s.s1_shape, s.s3_shape) for s in queues["c0"]),
        )
        self.assertEqual(
            (("c1", 0, 63, "S3"), ("c0", 1, 1, "S1")),
            schedule.cross_cluster_dma_release_edges(queues)[0],
        )

    def test_s2pf_early_mode_requires_positive_s1_slack(self) -> None:
        cases = (
            (schedule.SHAPE_A, schedule.DMA_IDMA, 2),
            (schedule.SHAPE_A, schedule.DMA_BOTH, 2),
            (schedule.SHAPE_B, schedule.DMA_IDMA, 0),
            (schedule.SHAPE_B, schedule.DMA_BOTH, 2),
            (schedule.SHAPE_C, schedule.DMA_IDMA, 0),
            (schedule.SHAPE_C, schedule.DMA_BOTH, 0),
        )
        for shape, dma, expected in cases:
            with self.subTest(shape=shape, dma=dma):
                self.assertEqual(
                    expected,
                    schedule.s2pf_s1_overlap_steps(
                        skip_s1=False,
                        s1_shape=shape,
                        s1_dma=dma,
                        s2_prefetch_dma=schedule.DMA_IDMA,
                    ),
                )
        self.assertEqual(
            0,
            schedule.s2pf_s1_overlap_steps(
                skip_s1=True,
                s1_shape=schedule.SHAPE_A,
                s1_dma=schedule.DMA_NONE,
                s2_prefetch_dma=schedule.DMA_XDMA,
            ),
        )

    def test_s2pf_both_profile_is_available(self) -> None:
        self.assertIn(schedule.S2PF_BOTH_PROFILE, schedule.SCHEDULE_PROFILES)

    def test_s2pf_both_changes_only_c0_slot0(self) -> None:
        baseline = {
            ("c0", 0): schedule.DMA_IDMA,
            ("c1", 0): schedule.DMA_NONE,
            ("c0", 1): schedule.DMA_IDMA,
            ("c1", 1): schedule.DMA_XDMA,
        }
        selected = {
            key: schedule.select_two_slot_s2pf_binding(
                schedule.S2PF_BOTH_PROFILE, *key, binding
            )
            for key, binding in baseline.items()
        }
        self.assertEqual(schedule.DMA_BOTH, selected[("c0", 0)])
        self.assertEqual(schedule.DMA_NONE, selected[("c1", 0)])
        self.assertEqual(schedule.DMA_IDMA, selected[("c0", 1)])
        self.assertEqual(schedule.DMA_XDMA, selected[("c1", 1)])

    def test_static_desc_matches_case0_policy0(self) -> None:
        queues = schedule.build_static_desc_schedule()
        audit = schedule.audit_static_desc_schedule(queues)

        self.assertEqual({"c0": 159, "c1": 162}, audit["queue_ticks"])
        self.assertEqual({"c0": 22, "c1": 21}, audit["cluster_local_slots"])
        self.assertEqual(43, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual((), audit["dma_release_edges"])

    def test_static_desc_has_only_fixed_physical_parameters(self) -> None:
        queues = schedule.build_static_desc_schedule()
        for cluster_name, slots in queues.items():
            expected_dma = (
                schedule.DMA_IDMA if cluster_name == "c0" else schedule.DMA_XDMA
            )
            for slot in slots:
                self.assertEqual(schedule.SHAPE_B, slot.s1_shape)
                self.assertEqual(schedule.SHAPE_B, slot.s3_shape)
                self.assertEqual(expected_dma, slot.s1_dma)
                self.assertEqual(expected_dma, slot.s3_dma)
                self.assertEqual(schedule.DMA_NONE, slot.s2_prefetch_dma)
                self.assertEqual(schedule.DMA_NONE, slot.s4_prefetch_dma)

    def test_static_desc_uses_exported_token_routing(self) -> None:
        routing = schedule.STATIC_DESC_TOKEN_IDS_BY_EXPERT
        self.assertEqual((0, 1, 2, 3, 4), routing[0][:5])
        self.assertEqual((24, 35, 50), routing[3])
        self.assertEqual((69,), routing[42])
        owners = [[] for _ in range(70)]
        for expert_id, token_ids in enumerate(routing):
            for token_id in token_ids:
                owners[token_id].append(expert_id)
        self.assertTrue(all(len(token_owners) == 2 for token_owners in owners))

    def test_static_desc_structural_lower_bound_is_177_75_ticks(self) -> None:
        queues = schedule.build_static_desc_schedule()
        audit = schedule.audit_static_desc_schedule(queues)

        self.assertEqual(
            {"c0": 702, "c1": 711},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            177.75,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )

    def test_m70_three_hot_static_desc_matches_case1_policy0(self) -> None:
        queues = schedule.build_m70_three_hot_static_desc_schedule()
        audit = schedule.audit_m70_three_hot_static_desc_schedule(queues)

        self.assertEqual({"c0": 132, "c1": 126}, audit["queue_ticks"])
        self.assertEqual({"c0": 10, "c1": 13}, audit["cluster_local_slots"])
        self.assertEqual(23, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual((), audit["dma_release_edges"])
        self.assertEqual(
            (0, 2, 8, 10, 12, 14, 16, 18, 20, 22),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (1, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m70_three_hot_static_desc_uses_exported_routing(self) -> None:
        routing = schedule.M70_THREE_HOT_TOKEN_IDS_BY_EXPERT
        self.assertEqual((0, 1, 3, 4), routing[0][:4])
        self.assertEqual((34, 38, 41, 45, 48, 60), routing[3])
        self.assertEqual((58, 69), routing[22])
        owners = [[] for _ in range(70)]
        for expert_id, token_ids in enumerate(routing):
            for token_id in token_ids:
                owners[token_id].append(expert_id)
        self.assertTrue(all(len(token_owners) == 2 for token_owners in owners))

    def test_m70_three_hot_static_desc_structural_lower_bound_is_139_5_ticks(
        self,
    ) -> None:
        queues = schedule.build_m70_three_hot_static_desc_schedule()
        audit = schedule.audit_m70_three_hot_static_desc_schedule(queues)

        self.assertEqual(
            {"c0": 558, "c1": 543},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            139.5,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )

    def test_m70_three_hot_dynamic_desc_matches_case1_policy1(self) -> None:
        queues = schedule.build_m70_three_hot_dynamic_desc_schedule()
        audit = schedule.audit_m70_three_hot_dynamic_desc_schedule(queues)

        self.assertEqual({"c0": 126, "c1": 126}, audit["queue_ticks"])
        self.assertEqual({"c0": 9, "c1": 14}, audit["cluster_local_slots"])
        self.assertEqual(23, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(
            (0, 2, 9, 11, 13, 15, 17, 19, 21),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (1, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m70_skip_elided_profile_reuses_exact_dynamic_schedule(self) -> None:
        self.assertEqual(
            schedule.build_m70_three_hot_dynamic_desc_schedule(),
            schedule.build_schedule_profile(
                schedule.M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE
            ),
        )

    def test_m70_three_hot_dynamic_desc_preserves_prefetch_contract(self) -> None:
        queues = schedule.build_m70_three_hot_dynamic_desc_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            {0: 2, 1: 2},
            {eid: slots[eid].s2pf_s1_overlap_steps for eid in (0, 1)},
        )
        self.assertEqual(
            {
                0: (schedule.DMA_IDMA, 2),
                1: (schedule.DMA_XDMA, 3),
                3: (schedule.DMA_BOTH, 4),
                4: (schedule.DMA_BOTH, 5),
                5: (schedule.DMA_BOTH, 6),
                6: (schedule.DMA_BOTH, 7),
            },
            {
                eid: (slot.s4_prefetch_dma, slot.s4_prefetch_target_eid)
                for eid, slot in slots.items()
                if slot.s4_prefetch_dma != schedule.DMA_NONE
            },
        )
        self.assertEqual(
            {2, 3, 4, 5, 6, 7},
            {eid for eid, slot in slots.items() if slot.skip_s1},
        )

    def test_m70_three_hot_dynamic_desc_dma_release_edges(self) -> None:
        queues = schedule.build_m70_three_hot_dynamic_desc_schedule()
        audit = schedule.audit_m70_three_hot_dynamic_desc_schedule(queues)

        self.assertEqual(4, len(audit["dma_release_edges"]))
        self.assertIn(
            (("c1", 3, 5, "S4PF"), ("c0", 1, 2, "S3")),
            audit["dma_release_edges"],
        )
        self.assertIn(
            (("c1", 6, 8, "S3"), ("c0", 2, 9, "S1")),
            audit["dma_release_edges"],
        )

    def test_m70_three_hot_dynamic_desc_structural_lower_bound_is_136_5_ticks(
        self,
    ) -> None:
        queues = schedule.build_m70_three_hot_dynamic_desc_schedule()
        audit = schedule.audit_m70_three_hot_dynamic_desc_schedule(queues)

        self.assertEqual(
            {"c0": 531, "c1": 546},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            136.5,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )

    def test_m70_three_hot_dynamic_two_ended_matches_case1_policy2(self) -> None:
        queues = schedule.build_m70_three_hot_dynamic_two_ended_schedule()
        audit = schedule.audit_m70_three_hot_dynamic_two_ended_schedule(queues)

        self.assertEqual({"c0": 127, "c1": 91}, audit["queue_ticks"])
        self.assertEqual({"c0": 3, "c1": 20}, audit["cluster_local_slots"])
        self.assertEqual(23, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual((0, 1, 2), tuple(slot.expert_id for slot in queues["c0"]))
        self.assertEqual(
            tuple(range(22, 2, -1)),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m70_three_hot_dynamic_two_ended_prefetch_contract(self) -> None:
        queues = schedule.build_m70_three_hot_dynamic_two_ended_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            {1, 2, 3, 4, 5, 6},
            {eid for eid, slot in slots.items() if slot.s2_prefetch_dma},
        )
        self.assertTrue(
            all(slot.s4_prefetch_dma == schedule.DMA_NONE for slot in slots.values())
        )
        self.assertTrue(all(not slot.skip_s1 for slot in slots.values()))

    def test_m70_three_hot_dynamic_two_ended_structural_bound(self) -> None:
        audit = schedule.audit_m70_three_hot_dynamic_two_ended_schedule(
            schedule.build_m70_three_hot_dynamic_two_ended_schedule()
        )

        self.assertEqual(
            {"c0": 517, "c1": 424},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(129.25, audit["structural_lower_bound_quarter_ticks"] / 4)
        self.assertEqual(7, len(audit["dma_release_edges"]))

    def test_m70_three_hot_full_scheduler_matches_case1_policy3(self) -> None:
        queues = schedule.build_m70_three_hot_full_scheduler_schedule()
        audit = schedule.audit_m70_three_hot_full_scheduler_schedule(queues)

        self.assertEqual({"c0": 105, "c1": 105}, audit["queue_ticks"])
        self.assertEqual({"c0": 7, "c1": 16}, audit["cluster_local_slots"])
        self.assertEqual(23, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(
            (2, 4, 1, 10, 9, 8, 7),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (3, *range(22, 11, -1), 5, 6, 11, 0),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m70_three_hot_full_scheduler_prefetch_contract(self) -> None:
        queues = schedule.build_m70_three_hot_full_scheduler_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            {
                0: schedule.DMA_BOTH,
                1: schedule.DMA_IDMA,
                2: schedule.DMA_IDMA,
                3: schedule.DMA_XDMA,
                4: schedule.DMA_IDMA,
                5: schedule.DMA_XDMA,
                6: schedule.DMA_XDMA,
            },
            {
                eid: slot.s2_prefetch_dma
                for eid, slot in slots.items()
                if slot.s2_prefetch_dma != schedule.DMA_NONE
            },
        )
        self.assertEqual(2, slots[0].s2pf_s1_overlap_steps)
        self.assertTrue(
            all(slots[eid].s2pf_s1_overlap_steps == 0 for eid in range(1, 7))
        )
        self.assertTrue(
            all(slot.s4_prefetch_dma == schedule.DMA_NONE for slot in slots.values())
        )

    def test_m70_three_hot_full_scheduler_structural_bound(self) -> None:
        audit = schedule.audit_m70_three_hot_full_scheduler_schedule(
            schedule.build_m70_three_hot_full_scheduler_schedule()
        )

        self.assertEqual(
            {"c0": 441, "c1": 468},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(117.0, audit["structural_lower_bound_quarter_ticks"] / 4)
        self.assertEqual(4, len(audit["dma_release_edges"]))

    def test_m92_parameter_order_static_desc_matches_case2_policy0(self) -> None:
        queues = schedule.build_m92_parameter_order_static_desc_schedule()
        audit = schedule.audit_m92_parameter_order_static_desc_schedule(queues)

        self.assertEqual({"c0": 198, "c1": 192}, audit["queue_ticks"])
        self.assertEqual({"c0": 15, "c1": 23}, audit["cluster_local_slots"])
        self.assertEqual(38, audit["task_count"])
        self.assertEqual(184, audit["routed_tokens"])
        self.assertEqual(
            (0, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
                36,
            ),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m92_parameter_order_static_desc_uses_exact_routing(self) -> None:
        routing = schedule.M92_PARAMETER_ORDER_TOKEN_IDS_BY_EXPERT

        self.assertEqual(tuple(range(76)), routing[0])
        self.assertEqual((*range(39), 71), routing[1])
        self.assertEqual((39, 72), routing[2])
        self.assertEqual((90,), routing[34])
        self.assertEqual((), routing[63])
        owners = [[] for _ in range(92)]
        for expert_id, token_ids in enumerate(routing):
            for token_id in token_ids:
                owners[token_id].append(expert_id)
        self.assertTrue(all(len(token_owners) == 2 for token_owners in owners))

    def test_m92_parameter_order_static_desc_structural_bound(self) -> None:
        audit = schedule.audit_m92_parameter_order_static_desc_schedule(
            schedule.build_m92_parameter_order_static_desc_schedule()
        )

        self.assertEqual(
            {"c0": 837, "c1": 837},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            209.25,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )
        self.assertEqual((), audit["dma_release_edges"])

    def test_m60_high_skew_static_desc_matches_case3_policy0(self) -> None:
        queues = schedule.build_m60_high_skew_static_desc_schedule()
        audit = schedule.audit_m60_high_skew_static_desc_schedule(queues)

        self.assertIn(
            schedule.M60_HIGH_SKEW_STATIC_DESC_PROFILE,
            schedule.SCHEDULE_PROFILES,
        )
        self.assertEqual({"c0": 135, "c1": 138}, audit["queue_ticks"])
        self.assertEqual({"c0": 14, "c1": 16}, audit["cluster_local_slots"])
        self.assertEqual(30, audit["task_count"])
        self.assertEqual(120, audit["routed_tokens"])
        self.assertEqual((), audit["dma_release_edges"])
        self.assertEqual(
            (0, 3, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (1, 2, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m60_high_skew_static_desc_uses_exact_routing(self) -> None:
        routing = schedule.M60_HIGH_SKEW_TOKEN_IDS_BY_EXPERT

        self.assertEqual((*range(35), 45), routing[0])
        self.assertEqual((25, 28, 31, 34, 36, 46), routing[3])
        self.assertEqual((59,), routing[29])
        self.assertEqual((), routing[63])
        owners = [[] for _ in range(60)]
        for expert_id, token_ids in enumerate(routing):
            for token_id in token_ids:
                owners[token_id].append(expert_id)
        self.assertTrue(all(len(token_owners) == 2 for token_owners in owners))

    def test_m60_high_skew_static_desc_structural_bound(self) -> None:
        queues = schedule.build_schedule_profile(
            schedule.M60_HIGH_SKEW_STATIC_DESC_PROFILE
        )
        audit = schedule.audit_m60_high_skew_static_desc_schedule(queues)

        self.assertEqual(
            {"c0": 582, "c1": 600},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            150.0,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )
        for cluster_name, slots in queues.items():
            expected_dma = (
                schedule.DMA_IDMA if cluster_name == "c0" else schedule.DMA_XDMA
            )
            for slot in slots:
                self.assertEqual(schedule.SHAPE_B, slot.s1_shape)
                self.assertEqual(schedule.SHAPE_B, slot.s3_shape)
                self.assertEqual(expected_dma, slot.s1_dma)
                self.assertEqual(expected_dma, slot.s3_dma)
                self.assertEqual(schedule.DMA_NONE, slot.s2_prefetch_dma)
                self.assertEqual(schedule.DMA_NONE, slot.s4_prefetch_dma)

    def test_m60_high_skew_dynamic_desc_matches_case3_policy1(self) -> None:
        queues = schedule.build_m60_high_skew_dynamic_desc_schedule()
        audit = schedule.audit_m60_high_skew_dynamic_desc_schedule(queues)

        self.assertIn(
            schedule.M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE,
            schedule.SCHEDULE_PROFILES,
        )
        self.assertEqual({"c0": 133, "c1": 130}, audit["queue_ticks"])
        self.assertEqual({"c0": 14, "c1": 16}, audit["cluster_local_slots"])
        self.assertEqual(30, audit["task_count"])
        self.assertEqual(120, audit["routed_tokens"])
        self.assertEqual(
            (0, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (1, 2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m60_high_skew_dynamic_desc_prefetch_contract(self) -> None:
        queues = schedule.build_m60_high_skew_dynamic_desc_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            {0: 2, 1: 2},
            {eid: slots[eid].s2pf_s1_overlap_steps for eid in (0, 1)},
        )
        self.assertEqual(
            {
                0: (schedule.DMA_IDMA, 3),
                1: (schedule.DMA_XDMA, 2),
                2: (schedule.DMA_XDMA, 4),
            },
            {
                eid: (slot.s4_prefetch_dma, slot.s4_prefetch_target_eid)
                for eid, slot in slots.items()
                if slot.s4_prefetch_dma != schedule.DMA_NONE
            },
        )
        self.assertEqual(
            {2, 3, 4},
            {eid for eid, slot in slots.items() if slot.skip_s1},
        )

    def test_m60_high_skew_dynamic_desc_structural_bound(self) -> None:
        audit = schedule.audit_m60_high_skew_dynamic_desc_schedule(
            schedule.build_schedule_profile(schedule.M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE)
        )

        self.assertEqual(
            {"c0": 574, "c1": 568},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            143.5,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )
        self.assertEqual(26, len(audit["dma_release_edges"]))

    def test_m60_high_skew_dynamic_two_ended_matches_case3_policy2(
        self,
    ) -> None:
        queues = schedule.build_m60_high_skew_dynamic_two_ended_schedule()
        audit = schedule.audit_m60_high_skew_dynamic_two_ended_schedule(queues)

        self.assertIn(
            schedule.M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE,
            schedule.SCHEDULE_PROFILES,
        )
        self.assertEqual({"c0": 111, "c1": 94}, audit["queue_ticks"])
        self.assertEqual({"c0": 3, "c1": 27}, audit["cluster_local_slots"])
        self.assertEqual(30, audit["task_count"])
        self.assertEqual(120, audit["routed_tokens"])
        self.assertEqual(
            (0, 1, 2),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            tuple(range(29, 2, -1)),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m60_high_skew_dynamic_two_ended_prefetch_contract(self) -> None:
        queues = schedule.build_m60_high_skew_dynamic_two_ended_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            {1: 2, 2: 2, 3: 0},
            {eid: slots[eid].s2pf_s1_overlap_steps for eid in (1, 2, 3)},
        )
        self.assertTrue(
            all(slot.s4_prefetch_dma == schedule.DMA_NONE for slot in slots.values())
        )
        self.assertTrue(all(not slot.skip_s1 for slot in slots.values()))

    def test_m60_high_skew_dynamic_two_ended_structural_bound(self) -> None:
        audit = schedule.audit_m60_high_skew_dynamic_two_ended_schedule(
            schedule.build_schedule_profile(
                schedule.M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE
            )
        )

        self.assertEqual(
            {"c0": 453, "c1": 457},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            114.25,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )
        self.assertEqual(6, len(audit["dma_release_edges"]))

    def test_m60_high_skew_full_scheduler_matches_case3_policy3(self) -> None:
        queues = schedule.build_m60_high_skew_full_scheduler_schedule()
        audit = schedule.audit_m60_high_skew_full_scheduler_schedule(queues)

        self.assertIn(
            schedule.M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE,
            schedule.SCHEDULE_PROFILES,
        )
        self.assertEqual({"c0": 99, "c1": 99}, audit["queue_ticks"])
        self.assertEqual({"c0": 10, "c1": 20}, audit["cluster_local_slots"])
        self.assertEqual(30, audit["task_count"])
        self.assertEqual(120, audit["routed_tokens"])
        self.assertEqual(
            (0, *range(13, 5, -1), 2),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (29, *range(28, 13, -1), 1, 5, 4, 3),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m60_high_skew_full_scheduler_prefetch_contract(self) -> None:
        queues = schedule.build_m60_high_skew_full_scheduler_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            {
                0: schedule.DMA_IDMA,
                1: schedule.DMA_BOTH,
                2: schedule.DMA_BOTH,
            },
            {
                eid: slot.s2_prefetch_dma
                for eid, slot in slots.items()
                if slot.s2_prefetch_dma != schedule.DMA_NONE
            },
        )
        self.assertTrue(all(slots[eid].s2pf_s1_overlap_steps == 2 for eid in (0, 1, 2)))
        self.assertTrue(
            all(slot.s4_prefetch_dma == schedule.DMA_NONE for slot in slots.values())
        )
        self.assertTrue(all(not slot.skip_s1 for slot in slots.values()))

    def test_m60_high_skew_full_scheduler_structural_bound(self) -> None:
        audit = schedule.audit_m60_high_skew_full_scheduler_schedule(
            schedule.build_schedule_profile(
                schedule.M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE
            )
        )

        self.assertEqual(
            {"c0": 426, "c1": 456},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            114.0,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )
        self.assertEqual(
            {
                (("c0", 0, 0, "S2PF"), ("c1", 1, 28, "S1")),
                (("c1", 16, 1, "S2PF"), ("c0", 1, 13, "S1")),
                (("c0", 9, 2, "S2PF"), ("c1", 17, 5, "S1")),
            },
            set(audit["dma_release_edges"]),
        )

    def test_m92_parameter_order_dynamic_desc_matches_case2_policy1(self) -> None:
        queues = schedule.build_m92_parameter_order_dynamic_desc_schedule()
        audit = schedule.audit_m92_parameter_order_dynamic_desc_schedule(queues)

        self.assertEqual({"c0": 168, "c1": 168}, audit["queue_ticks"])
        self.assertEqual({"c0": 10, "c1": 28}, audit["cluster_local_slots"])
        self.assertEqual(38, audit["task_count"])
        self.assertEqual(184, audit["routed_tokens"])
        self.assertEqual(
            (0, 20, 22, 24, 26, 28, 30, 32, 34, 36),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                21,
                23,
                25,
                27,
                29,
                31,
                33,
                35,
                37,
            ),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m92_parameter_order_dynamic_desc_prefetch_contract(self) -> None:
        queues = schedule.build_m92_parameter_order_dynamic_desc_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(schedule.DMA_IDMA, slots[0].s2_prefetch_dma)
        self.assertEqual(schedule.DMA_XDMA, slots[1].s2_prefetch_dma)
        self.assertEqual(2, slots[0].s2pf_s1_overlap_steps)
        self.assertEqual(2, slots[1].s2pf_s1_overlap_steps)
        self.assertEqual(schedule.DMA_XDMA, slots[1].s4_prefetch_dma)
        self.assertEqual(2, slots[1].s4_prefetch_target_eid)
        self.assertTrue(slots[2].skip_s1)
        self.assertEqual({2}, {eid for eid, slot in slots.items() if slot.skip_s1})

    def test_m92_parameter_order_dynamic_desc_structural_bound(self) -> None:
        audit = schedule.audit_m92_parameter_order_dynamic_desc_schedule(
            schedule.build_m92_parameter_order_dynamic_desc_schedule()
        )

        self.assertEqual(
            {"c0": 702, "c1": 756},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            189.0,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )
        self.assertEqual(2, len(audit["dma_release_edges"]))

    def test_m92_parameter_order_dynamic_two_ended_matches_case2_policy2(
        self,
    ) -> None:
        queues = schedule.build_m92_parameter_order_dynamic_two_ended_schedule()
        audit = schedule.audit_m92_parameter_order_dynamic_two_ended_schedule(queues)

        self.assertEqual({"c0": 114, "c1": 172}, audit["queue_ticks"])
        self.assertEqual({"c0": 1, "c1": 37}, audit["cluster_local_slots"])
        self.assertEqual(38, audit["task_count"])
        self.assertEqual(184, audit["routed_tokens"])
        self.assertEqual((0,), tuple(slot.expert_id for slot in queues["c0"]))
        self.assertEqual(
            tuple(range(37, 0, -1)),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m92_parameter_order_dynamic_two_ended_prefetch_contract(
        self,
    ) -> None:
        queues = schedule.build_m92_parameter_order_dynamic_two_ended_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(schedule.DMA_BOTH, slots[1].s2_prefetch_dma)
        self.assertEqual(2, slots[1].s2pf_s1_overlap_steps)
        self.assertEqual(
            {1},
            {
                eid
                for eid, slot in slots.items()
                if slot.s2_prefetch_dma != schedule.DMA_NONE
            },
        )
        self.assertTrue(
            all(slot.s4_prefetch_dma == schedule.DMA_NONE for slot in slots.values())
        )
        self.assertTrue(all(not slot.skip_s1 for slot in slots.values()))

    def test_m92_parameter_order_dynamic_two_ended_structural_bound(
        self,
    ) -> None:
        audit = schedule.audit_m92_parameter_order_dynamic_two_ended_schedule(
            schedule.build_m92_parameter_order_dynamic_two_ended_schedule()
        )

        self.assertEqual(
            {"c0": 459, "c1": 799},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            199.75,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )
        self.assertEqual(3, len(audit["dma_release_edges"]))

    def test_m92_parameter_order_full_scheduler_matches_case2_policy3(
        self,
    ) -> None:
        queues = schedule.build_m92_parameter_order_full_scheduler_schedule()
        audit = schedule.audit_m92_parameter_order_full_scheduler_schedule(queues)

        self.assertEqual({"c0": 141, "c1": 144}, audit["queue_ticks"])
        self.assertEqual({"c0": 10, "c1": 28}, audit["cluster_local_slots"])
        self.assertEqual(38, audit["task_count"])
        self.assertEqual(184, audit["routed_tokens"])
        self.assertEqual(
            (0, *range(10, 1, -1)),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (37, *range(36, 10, -1), 1),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_m92_parameter_order_full_scheduler_prefetch_contract(
        self,
    ) -> None:
        queues = schedule.build_m92_parameter_order_full_scheduler_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            {0: schedule.DMA_IDMA, 1: schedule.DMA_BOTH},
            {
                eid: slot.s2_prefetch_dma
                for eid, slot in slots.items()
                if slot.s2_prefetch_dma != schedule.DMA_NONE
            },
        )
        self.assertEqual(2, slots[0].s2pf_s1_overlap_steps)
        self.assertEqual(2, slots[1].s2pf_s1_overlap_steps)
        self.assertTrue(
            all(slot.s4_prefetch_dma == schedule.DMA_NONE for slot in slots.values())
        )
        self.assertTrue(all(not slot.skip_s1 for slot in slots.values()))

    def test_m92_parameter_order_full_scheduler_structural_bound(
        self,
    ) -> None:
        audit = schedule.audit_m92_parameter_order_full_scheduler_schedule(
            schedule.build_m92_parameter_order_full_scheduler_schedule()
        )

        self.assertEqual(
            {"c0": 594, "c1": 660},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            165.0,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )
        self.assertEqual(2, len(audit["dma_release_edges"]))

    def test_dynamic_desc_matches_case0_policy1(self) -> None:
        queues = schedule.build_dynamic_desc_schedule()
        audit = schedule.audit_dynamic_desc_schedule(queues)

        self.assertEqual({"c0": 159, "c1": 159}, audit["queue_ticks"])
        self.assertEqual({"c0": 22, "c1": 21}, audit["cluster_local_slots"])
        self.assertEqual(43, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(21, len(audit["dma_release_edges"]))
        early_s2pf = tuple(
            slot
            for slots in queues.values()
            for slot in slots
            if slot.s2pf_s1_overlap_steps != 0
        )
        self.assertEqual(20, len(early_s2pf))
        self.assertTrue(all(slot.s1_dma == slot.s2_prefetch_dma for slot in early_s2pf))
        self.assertTrue(
            all(
                slot.s2pf_s1_overlap_steps == 2
                for slots in queues.values()
                for slot in slots
                if slot.s2_prefetch_dma != schedule.DMA_NONE
            )
        )

    def test_dynamic_desc_preserves_preload_contract(self) -> None:
        queues = schedule.build_dynamic_desc_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            (schedule.DMA_XDMA, 2),
            (slots[1].s4_prefetch_dma, slots[1].s4_prefetch_target_eid),
        )
        self.assertEqual(
            (schedule.DMA_BOTH, 4),
            (slots[3].s4_prefetch_dma, slots[3].s4_prefetch_target_eid),
        )
        self.assertTrue(slots[2].skip_s1)
        self.assertTrue(slots[4].skip_s1)
        self.assertEqual(schedule.DMA_NONE, slots[2].s2_prefetch_dma)
        self.assertEqual(schedule.DMA_BOTH, slots[2].s3_dma)
        self.assertEqual(
            (
                schedule.SHAPE_C,
                schedule.SHAPE_C,
                schedule.DMA_BOTH,
                schedule.DMA_BOTH,
            ),
            (
                slots[22].s1_shape,
                slots[22].s3_shape,
                slots[22].s1_dma,
                slots[22].s3_dma,
            ),
        )

    def test_dynamic_desc_s4pf_is_in_dma_release_graph(self) -> None:
        queues = schedule.build_dynamic_desc_schedule()
        edges = set(schedule.cross_cluster_dma_release_edges(queues))

        self.assertIn(
            (("c1", 0, 1, "S4PF"), ("c0", 1, 3, "S1")),
            edges,
        )
        self.assertIn(
            (("c0", 1, 3, "S4PF"), ("c1", 1, 2, "S3")),
            edges,
        )

    def test_dynamic_desc_structural_lower_bound_is_175_5_ticks(self) -> None:
        queues = schedule.build_dynamic_desc_schedule()
        audit = schedule.audit_dynamic_desc_schedule(queues)

        self.assertEqual(
            {"c0": 702, "c1": 699},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            175.5,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )

    def test_dynamic_two_ended_matches_case0_policy2(self) -> None:
        queues = schedule.build_dynamic_two_ended_schedule()
        audit = schedule.audit_dynamic_two_ended_schedule(queues)

        self.assertEqual({"c0": 137, "c1": 134}, audit["queue_ticks"])
        self.assertEqual({"c0": 12, "c1": 31}, audit["cluster_local_slots"])
        self.assertEqual(43, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(26, len(audit["dma_release_edges"]))
        self.assertEqual(
            tuple(range(12)),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            tuple(range(42, 11, -1)),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_dynamic_two_ended_preserves_prefetch_contract(self) -> None:
        queues = schedule.build_dynamic_two_ended_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }
        early_s2pf = tuple(
            slot for slot in slots.values() if slot.s2pf_s1_overlap_steps != 0
        )

        self.assertEqual(19, len(early_s2pf))
        self.assertTrue(all(slot.s2pf_s1_overlap_steps == 2 for slot in early_s2pf))
        self.assertEqual(
            (schedule.DMA_BOTH, 20),
            (slots[21].s4_prefetch_dma, slots[21].s4_prefetch_target_eid),
        )
        self.assertTrue(slots[20].skip_s1)
        self.assertEqual(schedule.DMA_NONE, slots[20].s1_dma)
        self.assertEqual(schedule.DMA_BOTH, slots[20].s3_dma)

    def test_dynamic_two_ended_structural_lower_bound_is_157_25_ticks(
        self,
    ) -> None:
        queues = schedule.build_dynamic_two_ended_schedule()
        audit = schedule.audit_dynamic_two_ended_schedule(queues)

        self.assertEqual(
            {"c0": 584, "c1": 629},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            157.25,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )

    def test_full_scheduler_matches_case0_policy3(self) -> None:
        queues = schedule.build_full_scheduler_schedule()
        audit = schedule.audit_full_scheduler_schedule(queues)

        self.assertEqual({"c0": 129, "c1": 129}, audit["queue_ticks"])
        self.assertEqual({"c0": 11, "c1": 32}, audit["cluster_local_slots"])
        self.assertEqual(43, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(6, len(audit["dma_release_edges"]))
        self.assertEqual(
            (0, 1, 2, 5, 8, 10, 12, 14, 16, 18, 20),
            tuple(slot.expert_id for slot in queues["c0"]),
        )
        self.assertEqual(
            (
                4,
                *range(42, 33, -1),
                6,
                *range(33, 26, -1),
                3,
                *range(26, 21, -1),
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                21,
            ),
            tuple(slot.expert_id for slot in queues["c1"]),
        )

    def test_full_scheduler_preserves_prefetch_contract(self) -> None:
        queues = schedule.build_full_scheduler_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            {0, 1, 2},
            {
                expert_id
                for expert_id, slot in slots.items()
                if slot.s2_prefetch_dma != schedule.DMA_NONE
            },
        )
        self.assertEqual(
            {0: 2, 1: 2, 2: 0},
            {eid: slots[eid].s2pf_s1_overlap_steps for eid in (0, 1, 2)},
        )
        self.assertTrue(
            all(slots[eid].s2_prefetch_dma == schedule.DMA_IDMA for eid in (0, 1, 2))
        )
        self.assertTrue(
            all(slot.s4_prefetch_dma == schedule.DMA_NONE for slot in slots.values())
        )

    def test_full_scheduler_structural_lower_bound_is_153_ticks(self) -> None:
        queues = schedule.build_full_scheduler_schedule()
        audit = schedule.audit_full_scheduler_schedule(queues)

        self.assertEqual(
            {"c0": 549, "c1": 612},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(
            153,
            audit["structural_lower_bound_quarter_ticks"] / 4,
        )

    def test_existing_descending_profile_is_unchanged(self) -> None:
        queues = schedule.build_high_to_low_schedule()
        audit = schedule.audit_high_to_low_schedule(queues)

        self.assertEqual({"c0": 163, "c1": 160}, audit["queue_ticks"])
        self.assertEqual(43, audit["active_experts"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(
            179.5,
            schedule.HIGH_TO_LOW_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS / 4,
        )

    def test_ascending_profile_matches_handoff_shape(self) -> None:
        queues = schedule.build_low_to_high_schedule()
        audit = schedule.audit_low_to_high_schedule(queues)

        self.assertEqual({"c0": 165, "c1": 165}, audit["queue_ticks"])
        self.assertEqual(43, audit["active_experts"])
        self.assertEqual(44, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(22, len(queues["c0"]))
        self.assertEqual(22, len(queues["c1"]))

    def test_ascending_e0_split_is_disjoint_and_complete(self) -> None:
        queues = schedule.build_low_to_high_schedule()
        slices = sorted(
            (slot.token_start_rank, slot.ntokens, slot.cluster_name)
            for slots in queues.values()
            for slot in slots
            if slot.expert_id == 0
        )
        self.assertEqual([(0, 11, "c0"), (11, 11, "c1")], slices)

    def test_ascending_e2_preserves_explicit_dma_bindings(self) -> None:
        queues = schedule.build_low_to_high_schedule()
        e2 = next(slot for slot in queues["c1"] if slot.expert_id == 2)
        timeline = schedule._task_timeline(e2, e2.reference_start_tick)

        self.assertEqual(schedule.SHAPE_C, e2.s1_shape)
        self.assertEqual(schedule.DMA_XDMA, e2.s1_dma)
        self.assertEqual(schedule.DMA_IDMA, e2.s3_dma)
        self.assertEqual((120, 144), (timeline.start, timeline.task_end))

    def test_ascending_structural_lower_bound_is_181_5_ticks(self) -> None:
        quarters = schedule.LOW_TO_HIGH_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
        self.assertEqual(
            4 * 165 + 22 * schedule.STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
            quarters,
        )
        self.assertEqual(181.5, quarters / 4)

    def test_ends_inward_profile_matches_handoff_shape(self) -> None:
        queues = schedule.build_ends_inward_schedule()
        audit = schedule.audit_ends_inward_schedule(queues)

        self.assertEqual({"c0": 166, "c1": 166}, audit["queue_ticks"])
        self.assertEqual({"c0": 21, "c1": 22}, audit["cluster_local_slots"])
        self.assertEqual(43, audit["task_count"])
        self.assertEqual(140, audit["routed_tokens"])
        self.assertEqual(7, len(audit["dma_release_edges"]))

    def test_ends_inward_preserves_special_stage_profiles(self) -> None:
        queues = schedule.build_ends_inward_schedule()
        slots = {
            slot.expert_id: slot
            for cluster_slots in queues.values()
            for slot in cluster_slots
        }

        self.assertEqual(
            (schedule.SHAPE_C, schedule.SHAPE_B, schedule.DMA_BOTH),
            (slots[41].s1_shape, slots[41].s3_shape, slots[41].s1_dma),
        )
        self.assertEqual(schedule.DMA_BOTH, slots[42].s3_dma)
        self.assertEqual(
            (schedule.SHAPE_A, schedule.SHAPE_C, schedule.DMA_IDMA),
            (slots[3].s1_shape, slots[3].s3_shape, slots[3].s3_dma),
        )

    def test_ends_inward_structural_lower_bound_is_182_5_ticks(self) -> None:
        queues = schedule.build_ends_inward_schedule()
        audit = schedule.audit_ends_inward_schedule(queues)

        self.assertEqual(
            {"c0": 727, "c1": 730},
            audit["structural_cluster_quarter_ticks"],
        )
        self.assertEqual(182.5, audit["structural_lower_bound_quarter_ticks"] / 4)


if __name__ == "__main__":
    unittest.main()
