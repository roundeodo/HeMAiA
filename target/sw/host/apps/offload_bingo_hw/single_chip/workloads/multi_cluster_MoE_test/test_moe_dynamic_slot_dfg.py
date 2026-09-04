import sys
import unittest
from pathlib import Path
from unittest.mock import patch

WORKLOADS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOADS_DIR))
SW_DIR = next(path for path in Path(__file__).resolve().parents if path.name == "sw")
sys.path.insert(0, str(SW_DIR / "host/runtime/libbingo/mini_compiler"))
KERNEL_DIR = (
    SW_DIR / "device/apps/snax/snax-bingo-offload/libsnaxkernel/offload_hw_kernels"
)

from moe_dynamic_slot_dfg import (  # noqa: E402
    build_dynamic_expert_skip_s1_elided_slot_chain,
    build_dynamic_expert_slot_chain,
)
import moe_high_to_low_workload as high_to_low  # noqa: E402
from moe_high_to_low_workload import (  # noqa: E402
    DMA_RELEASE_EDGE_PROFILES,
    _add_dma_release_edges,
)
from moe_test_schedule import (  # noqa: E402
    DMA_NONE,
    M32_DISTILLED_PROFILE,
    M32_FIXED_A_PROFILE,
    M32_FIXED_B_PROFILE,
    M32_FIXED_C_PROFILE,
    M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE,
    M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE,
    M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE,
    M60_HIGH_SKEW_STATIC_DESC_PROFILE,
    M92_PARAMETER_ORDER_DYNAMIC_DESC_PROFILE,
    M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_PROFILE,
    M92_PARAMETER_ORDER_FULL_SCHEDULER_PROFILE,
    M92_PARAMETER_ORDER_STATIC_DESC_PROFILE,
    SHAPE_A,
    build_m32_fixed_shape_schedule,
    build_m70_three_hot_dynamic_desc_schedule,
    build_m60_high_skew_dynamic_desc_schedule,
    build_m60_high_skew_dynamic_two_ended_schedule,
    build_m60_high_skew_full_scheduler_schedule,
    build_m92_parameter_order_full_scheduler_schedule,
    cross_cluster_dma_release_edges,
)


class MoeDynamicSlotDfgTest(unittest.TestCase):
    def test_pipeline_ctrl_is_reset_before_global_timing_and_gather(self) -> None:
        full_queues = build_m32_fixed_shape_schedule(SHAPE_A)
        queues = {name: (slots[0],) for name, slots in full_queues.items()}

        class FakeHandle:
            def __init__(self, name):
                self.name = name

            def get_c_var_name(self):
                return f"ptr_{self.name}"

        mh = {
            "input": FakeHandle("input"),
            "prod_slot_token_refs": FakeHandle("refs"),
            "pipeline_ctrl_zero": FakeHandle("pipeline_ctrl_zero"),
        }
        for prefix in ("c0", "c1"):
            for stem in (
                "layout",
                "gate",
                "up",
                "down",
                "gate_out",
                "pipeline_ctrl",
                "prod_output_l3",
                "prod_static_l3",
                "prod_runtime_l3",
                "prod_static_l1",
                "prod_runtime_l1",
                "token_refs_l1",
            ):
                mh[f"{prefix}_{stem}"] = FakeHandle(f"{prefix}_{stem}")

        class FakeDfg:
            def __init__(self):
                self.edges = []

            def bingo_add_edge(self, source, target):
                self.edges.append((source, target))

            def add_edge(self, source, target, descriptor_sequence=False):
                del descriptor_sequence
                self.edges.append((source, target))

        dfg = FakeDfg()
        emitted = {}

        def fake_add_node(_dfg, cluster, core, kernel, args, node_name=""):
            emitted[node_name] = (cluster, core, kernel, args)
            return node_name

        def fake_slot_chain(*, add_edge, input_ready, label_prefix, **kwargs):
            del kwargs
            store = f"{label_prefix}_STORE"
            add_edge(input_ready, store)
            return {"store": store}

        p = {
            "schedule_profile": M32_FIXED_A_PROFILE,
            "prod_token_refs_bytes": 1024,
            "s1_block_count": 8,
            "s3_block_count": 4,
        }
        with patch.object(
            high_to_low, "_add_node", side_effect=fake_add_node
        ), patch.object(
            high_to_low,
            "build_dynamic_expert_slot_chain",
            side_effect=fake_slot_chain,
        ), patch.object(
            high_to_low, "audit_m32_comparison_schedule"
        ):
            high_to_low.add_high_to_low_schedule(
                dfg,
                p,
                mh,
                queues,
                "optimized",
                timing_stages=(28, 29),
            )

        begin = "M32_FIXED_A_GLOBAL_BEGIN"
        for prefix in ("C0", "C1"):
            load = f"{prefix}_M32_FIXED_A_LOAD_DYNAMIC_ABI"
            reset = f"{prefix}_M32_FIXED_A_RESET_PIPELINE_CTRL"
            gather = f"{prefix}_M32_FIXED_A_SLOT0_GATHER"
            self.assertIn((load, reset), dfg.edges)
            self.assertIn((reset, begin), dfg.edges)
            self.assertIn((begin, gather), dfg.edges)
            reset_args = emitted[reset][3]
            self.assertIs(reset_args.src_addr, mh["pipeline_ctrl_zero"])
            self.assertEqual(1024, reset_args.size)

    def test_standalone_m32_runs_use_one_global_timing_pair(self) -> None:
        expected_stages = {
            M32_FIXED_A_PROFILE: (28, 29),
            M32_FIXED_B_PROFILE: (30, 31),
            M32_FIXED_C_PROFILE: (32, 33),
            M32_DISTILLED_PROFILE: (34, 35),
        }
        for profile, stages in expected_stages.items():
            with self.subTest(profile=profile):
                with patch.object(
                    high_to_low,
                    "add_high_to_low_schedule",
                    return_value=["check"],
                ) as add_schedule:
                    result = high_to_low.add_m32_comparison_run(
                        "dfg",
                        {"schedule_profile": profile},
                        "memory",
                        "queues",
                        "implementation",
                    )
                self.assertEqual(["check"], result)
                self.assertEqual(stages, add_schedule.call_args.kwargs["timing_stages"])

    def test_m92_dma_release_edges_are_lowered_for_dynamic_profiles(self) -> None:
        self.assertTrue(
            {
                M92_PARAMETER_ORDER_DYNAMIC_DESC_PROFILE,
                M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_PROFILE,
                M92_PARAMETER_ORDER_FULL_SCHEDULER_PROFILE,
            }.issubset(DMA_RELEASE_EDGE_PROFILES)
        )
        self.assertNotIn(
            M92_PARAMETER_ORDER_STATIC_DESC_PROFILE,
            DMA_RELEASE_EDGE_PROFILES,
        )
        self.assertNotIn(
            M60_HIGH_SKEW_STATIC_DESC_PROFILE,
            DMA_RELEASE_EDGE_PROFILES,
        )
        self.assertIn(
            M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE,
            DMA_RELEASE_EDGE_PROFILES,
        )
        self.assertIn(
            M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE,
            DMA_RELEASE_EDGE_PROFILES,
        )
        self.assertIn(
            M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE,
            DMA_RELEASE_EDGE_PROFILES,
        )

    def test_m60_dynamic_desc_release_edges_reach_stage_nodes(self) -> None:
        queues = build_m60_high_skew_dynamic_desc_schedule()
        release_edges = cross_cluster_dma_release_edges(queues)
        slot_chains = {}
        for cluster_name, slots in queues.items():
            for slot in slots:
                prefix = f"{cluster_name}_S{slot.local_slot}"
                slot_chains[(cluster_name, slot.local_slot)] = {
                    "s1_load": f"{prefix}_S1",
                    "s2_prefetch": f"{prefix}_S2PF",
                    "s3_load": f"{prefix}_S3",
                    "s4_prepare": f"{prefix}_S4PF",
                }

        lowered_edges = set()

        class FakeDfg:
            @staticmethod
            def bingo_add_edge(source, target):
                lowered_edges.add((source, target))

        _add_dma_release_edges(FakeDfg(), slot_chains, release_edges)

        self.assertEqual(26, len(lowered_edges))
        self.assertIn(("c0_S0_S4PF", "c1_S1_S3"), lowered_edges)
        self.assertIn(("c1_S15_S3", "c0_S13_S1"), lowered_edges)

    def test_m60_dynamic_two_ended_release_edges_reach_stage_nodes(
        self,
    ) -> None:
        queues = build_m60_high_skew_dynamic_two_ended_schedule()
        release_edges = cross_cluster_dma_release_edges(queues)
        slot_chains = {}
        for cluster_name, slots in queues.items():
            for slot in slots:
                prefix = f"{cluster_name}_S{slot.local_slot}"
                slot_chains[(cluster_name, slot.local_slot)] = {
                    "s1_load": f"{prefix}_S1",
                    "s2_prefetch": f"{prefix}_S2PF",
                    "s3_load": f"{prefix}_S3",
                    "s4_prepare": f"{prefix}_S4PF",
                }

        lowered_edges = set()

        class FakeDfg:
            @staticmethod
            def bingo_add_edge(source, target):
                lowered_edges.add((source, target))

        _add_dma_release_edges(FakeDfg(), slot_chains, release_edges)

        self.assertEqual(6, len(lowered_edges))
        self.assertIn(("c0_S0_S1", "c1_S0_S3"), lowered_edges)
        self.assertIn(("c1_S26_S2PF", "c0_S2_S1"), lowered_edges)

    def test_m60_full_scheduler_release_edges_reach_stage_nodes(self) -> None:
        queues = build_m60_high_skew_full_scheduler_schedule()
        release_edges = cross_cluster_dma_release_edges(queues)
        slot_chains = {}
        for cluster_name, slots in queues.items():
            for slot in slots:
                prefix = f"{cluster_name}_S{slot.local_slot}"
                slot_chains[(cluster_name, slot.local_slot)] = {
                    "s1_load": f"{prefix}_S1",
                    "s2_prefetch": f"{prefix}_S2PF",
                    "s3_load": f"{prefix}_S3",
                    "s4_prepare": f"{prefix}_S4PF",
                }

        lowered_edges = set()

        class FakeDfg:
            @staticmethod
            def bingo_add_edge(source, target):
                lowered_edges.add((source, target))

        _add_dma_release_edges(FakeDfg(), slot_chains, release_edges)

        self.assertEqual(
            {
                ("c0_S0_S2PF", "c1_S1_S1"),
                ("c1_S16_S2PF", "c0_S1_S1"),
                ("c0_S9_S2PF", "c1_S17_S1"),
            },
            lowered_edges,
        )

    def test_m92_full_scheduler_release_edges_reach_expected_stage_nodes(
        self,
    ) -> None:
        queues = build_m92_parameter_order_full_scheduler_schedule()
        release_edges = cross_cluster_dma_release_edges(queues)
        slot_chains = {}
        for cluster_name, slots in queues.items():
            for slot in slots:
                prefix = f"{cluster_name}_S{slot.local_slot}"
                slot_chains[(cluster_name, slot.local_slot)] = {
                    "s1_load": f"{prefix}_S1",
                    "s2_prefetch": f"{prefix}_S2PF",
                    "s3_load": f"{prefix}_S3",
                    "s4_prepare": f"{prefix}_S4PF",
                }

        lowered_edges = set()

        class FakeDfg:
            @staticmethod
            def bingo_add_edge(source, target):
                lowered_edges.add((source, target))

        _add_dma_release_edges(FakeDfg(), slot_chains, release_edges)

        self.assertEqual(
            {
                ("c0_S0_S2PF", "c1_S1_S1"),
                ("c1_S27_S2PF", "c0_S1_S1"),
            },
            lowered_edges,
        )

    def test_m70_skip_elision_removes_only_empty_s1_stage_nodes(self) -> None:
        queues = build_m70_three_hot_dynamic_desc_schedule()
        original_counts = {}
        elided_counts = {}

        for cluster_name, slots in queues.items():
            original_nodes = []
            elided_nodes = []
            for slot in slots:
                common = {
                    "add_edge": lambda _source, _target: None,
                    "add_descriptor_sequence": lambda _source, _target: None,
                    "make_block_args": lambda block: block,
                    "input_ready": f"{cluster_name}_{slot.local_slot}_READY",
                    "s1_block_count": 8,
                    "s3_block_count": 4,
                    "dma_core_id": 1,
                    "gemm_core_id": 0,
                    "emit_s2_prefetch_task": (
                        slot.s2_prefetch_dma != DMA_NONE
                        and slot.s2pf_s1_overlap_steps == 0
                    ),
                    "s2pf_starts_after_s1_dma": (slot.s2pf_starts_after_s1_dma),
                    "label_prefix": (
                        f"{cluster_name}_S{slot.local_slot}_E{slot.expert_id}"
                    ),
                }
                build_dynamic_expert_slot_chain(
                    add_node=lambda _core, _kernel, _args, label: (
                        original_nodes.append(label) or label
                    ),
                    implementation="optimized",
                    **common,
                )
                if slot.skip_s1:
                    build_dynamic_expert_skip_s1_elided_slot_chain(
                        add_node=lambda _core, _kernel, _args, label: (
                            elided_nodes.append(label) or label
                        ),
                        **common,
                    )
                else:
                    build_dynamic_expert_slot_chain(
                        add_node=lambda _core, _kernel, _args, label: (
                            elided_nodes.append(label) or label
                        ),
                        implementation="optimized",
                        **common,
                    )

            original_counts[cluster_name] = len(original_nodes)
            elided_counts[cluster_name] = len(elided_nodes)

        self.assertEqual({"c0": 90, "c1": 140}, original_counts)
        self.assertEqual({"c0": 87, "c1": 125}, elided_counts)

    def test_m70_skip_s1_slots_have_preceding_s4_compute(self) -> None:
        queues = build_m70_three_hot_dynamic_desc_schedule()
        for slots in queues.values():
            self.assertFalse(slots[0].skip_s1)
            for previous, current in zip(slots, slots[1:]):
                if current.skip_s1:
                    self.assertNotEqual(0, previous.s4_m_exec)

    def test_elided_s1_reconnects_s2_directly_to_slot_entry(self) -> None:
        completion_edges = set()
        created_labels = []

        chain = build_dynamic_expert_skip_s1_elided_slot_chain(
            add_node=lambda _core, _kernel, _args, label: (
                created_labels.append(label) or label
            ),
            add_edge=lambda source, target: completion_edges.add((source, target)),
            add_descriptor_sequence=lambda _source, _target: None,
            make_block_args=lambda block: block,
            input_ready="INPUT_READY",
            s1_block_count=8,
            s3_block_count=4,
            dma_core_id=1,
            gemm_core_id=0,
            emit_s2_prefetch_task=False,
            s2pf_starts_after_s1_dma=False,
            label_prefix="SLOT",
        )

        self.assertFalse(any("_S1_" in label for label in created_labels))
        self.assertIn(("INPUT_READY", chain["s2_compute"]), completion_edges)

    def test_s2pf_phase_batches_cover_every_slice(self) -> None:
        for binding in (1, 2, 3):
            for block_count in range(1, 9):
                actual = []
                for phase in range(2):
                    blocks = list(range(phase, block_count, 2))
                    if binding == 3:
                        actual.extend((block, 0, "idma") for block in blocks)
                        actual.extend((block, 1, "xdma") for block in blocks)
                    else:
                        lane = "idma" if binding == 1 else "xdma"
                        for side in range(2):
                            actual.extend((block, side, lane) for block in blocks)

                expected = [
                    (
                        block,
                        side,
                        (
                            "idma"
                            if binding == 1 or (binding == 3 and side == 0)
                            else "xdma"
                        ),
                    )
                    for block in range(block_count)
                    for side in range(2)
                ]
                self.assertEqual(sorted(expected), sorted(actual))

    def test_s2pf_phase_shape_preserves_every_block_row_address(self) -> None:
        block_bytes = 256
        row_bytes = 64
        tcdm_row_bytes = 512
        phase_delta = 0x20000
        half_bytes = 8 * block_bytes

        for block_count in range(1, 9):
            for side in range(2):
                expected = set()
                for block in range(block_count):
                    phase = block & 1
                    block_in_phase = block >> 1
                    for row in range(block_bytes // row_bytes):
                        expected.add(
                            (
                                side * half_bytes
                                + block * block_bytes
                                + row * row_bytes,
                                side * row_bytes
                                + (phase ^ 1) * phase_delta
                                + block_in_phase
                                * block_bytes
                                * (tcdm_row_bytes // row_bytes)
                                + row * tcdm_row_bytes,
                            )
                        )

                actual = set()
                for phase in range(2):
                    phase_blocks = (block_count + 1 - phase) // 2
                    for block_in_phase in range(phase_blocks):
                        for row in range(block_bytes // row_bytes):
                            actual.add(
                                (
                                    side * half_bytes
                                    + phase * block_bytes
                                    + block_in_phase * 2 * block_bytes
                                    + row * row_bytes,
                                    side * row_bytes
                                    + (phase ^ 1) * phase_delta
                                    + block_in_phase
                                    * block_bytes
                                    * (tcdm_row_bytes // row_bytes)
                                    + row * tcdm_row_bytes,
                                )
                            )

                self.assertEqual(expected, actual)

    def test_s2pf_and_s2_compute_join_only_at_s3(self) -> None:
        completion_edges = set()

        def add_node(_core, _kernel, _args, label):
            return label

        chain = build_dynamic_expert_slot_chain(
            add_node=add_node,
            add_edge=lambda source, target: completion_edges.add((source, target)),
            add_descriptor_sequence=lambda _source, _target: None,
            make_block_args=lambda block: block,
            input_ready="INPUT_READY",
            s1_block_count=8,
            s3_block_count=4,
            dma_core_id=1,
            gemm_core_id=0,
            emit_s2_prefetch_task=True,
            s2pf_starts_after_s1_dma=True,
            implementation="optimized",
            label_prefix="SLOT",
        )

        self.assertNotIn(
            (chain["s2_prefetch"], chain["s2_compute"]),
            completion_edges,
        )
        self.assertNotIn(
            (chain["s2_compute"], chain["s2_prefetch"]),
            completion_edges,
        )
        s3_config = "SLOT_S3_CONFIG_BLOCK0_DURING_LOAD0"
        for successor in (chain["s3_load"], s3_config):
            self.assertIn((chain["s2_prefetch"], successor), completion_edges)
            self.assertIn((chain["s2_compute"], successor), completion_edges)

    def test_s2pf_and_gate_up_compute_use_distinct_weight_buffers(self) -> None:
        s1 = (KERNEL_DIR / "moe_dynamic_stage_s1.h").read_text()
        compute = (KERNEL_DIR / "moe_dynamic_stage_compute.h").read_text()
        dma = (KERNEL_DIR / "moe_dynamic_dma.h").read_text()

        self.assertIn("s2->down_dst_base = st->l1_b_down_addr;", s1)
        self.assertIn("s2->down_dst_base", dma)
        self.assertIn("st->l1_b_gate_addr", compute)
        self.assertIn("st->l1_b_up_addr", compute)

    def test_s2_prefetch_starts_at_s1_dma_completion(self) -> None:
        completion_edges = set()
        descriptor_sequences = set()

        def add_node(_core, _kernel, _args, label):
            return label

        chain = build_dynamic_expert_slot_chain(
            add_node=add_node,
            add_edge=lambda source, target: completion_edges.add((source, target)),
            add_descriptor_sequence=lambda source, target: (
                descriptor_sequences.add((source, target))
            ),
            make_block_args=lambda block: block,
            input_ready="INPUT_READY",
            s1_block_count=8,
            s3_block_count=4,
            dma_core_id=1,
            gemm_core_id=0,
            emit_s2_prefetch_task=True,
            s2pf_starts_after_s1_dma=True,
            implementation="optimized",
            label_prefix="SLOT",
        )

        self.assertIn((chain["s1_load"], chain["s2_prefetch"]), completion_edges)
        self.assertNotIn((chain["s1_compute"], chain["s2_prefetch"]), completion_edges)
        self.assertIn((chain["s1_load"], chain["s2_compute"]), completion_edges)
        self.assertIn((chain["s1_compute"], chain["s2_compute"]), completion_edges)
        self.assertIn((chain["s1_load"], chain["s1_compute"]), descriptor_sequences)

    def test_fused_s2_prefetch_uses_s1_load_completion_directly(self) -> None:
        completion_edges = set()
        created_labels = []

        def add_node(_core, _kernel, _args, label):
            created_labels.append(label)
            return label

        chain = build_dynamic_expert_slot_chain(
            add_node=add_node,
            add_edge=lambda source, target: completion_edges.add((source, target)),
            add_descriptor_sequence=lambda _source, _target: None,
            make_block_args=lambda block: block,
            input_ready="INPUT_READY",
            s1_block_count=8,
            s3_block_count=4,
            dma_core_id=1,
            gemm_core_id=0,
            emit_s2_prefetch_task=False,
            s2pf_starts_after_s1_dma=True,
            implementation="optimized",
            label_prefix="SLOT",
        )

        self.assertEqual(chain["s1_load"], chain["s2_prefetch"])
        self.assertNotIn("SLOT_S2_DOWN_PREFETCH", created_labels)
        self.assertNotIn((chain["s1_load"], chain["s1_load"]), completion_edges)
        for successor in (
            chain["s3_load"],
            "SLOT_S3_CONFIG_BLOCK0_DURING_LOAD0",
        ):
            self.assertIn((chain["s1_load"], successor), completion_edges)

    def test_s2_prefetch_waits_for_s1_compute_without_slack(self) -> None:
        completion_edges = set()

        def add_node(_core, _kernel, _args, label):
            return label

        chain = build_dynamic_expert_slot_chain(
            add_node=add_node,
            add_edge=lambda source, target: completion_edges.add((source, target)),
            add_descriptor_sequence=lambda _source, _target: None,
            make_block_args=lambda block: block,
            input_ready="INPUT_READY",
            s1_block_count=8,
            s3_block_count=4,
            dma_core_id=1,
            gemm_core_id=0,
            emit_s2_prefetch_task=True,
            s2pf_starts_after_s1_dma=False,
            implementation="optimized",
            label_prefix="SLOT",
        )

        self.assertIn((chain["s1_load"], chain["s2_prefetch"]), completion_edges)
        self.assertIn((chain["s1_compute"], chain["s2_prefetch"]), completion_edges)

    def test_s1_and_s2_use_independent_pipeline_cookies(self) -> None:
        defs = (KERNEL_DIR / "moe_dynamic_defs.h").read_text()
        s1 = (KERNEL_DIR / "moe_dynamic_stage_s1.h").read_text()

        self.assertIn("volatile uint32_t load_done;", defs)
        self.assertIn("volatile uint32_t compute_done;", defs)
        self.assertIn("&s1->load_done", s1)
        self.assertIn("&s1->compute_done", s1)

        pipeline_start = s1.index("__moe_wait_s1_load_bank(")
        pipeline_end = s1.index(
            "MOE_PROFILE_RESOURCE_END(stage_profile);", pipeline_start
        )
        pipeline = s1[pipeline_start:pipeline_end]
        self.assertNotIn("sync->compute_done", pipeline)
        self.assertNotIn("sync->prefetch_done", pipeline)

    def test_s2pf_has_no_s1_or_s2_compute_cookie_dependency(self) -> None:
        defs = (KERNEL_DIR / "moe_dynamic_defs.h").read_text()
        s1 = (KERNEL_DIR / "moe_dynamic_stage_s1.h").read_text()
        prefetch = (KERNEL_DIR / "moe_dynamic_stage_prefetch.h").read_text()
        compute = (KERNEL_DIR / "moe_dynamic_stage_compute.h").read_text()
        s3 = (KERNEL_DIR / "moe_dynamic_stage_s3.h").read_text()
        s2pf = prefetch[: prefetch.index("__moe_s4pf_phase_marker")]
        s2pf_run = s2pf[
            s2pf.index("__moe_s2pf_submit_idma_phase(") : s2pf.index(
                "SNAX_LIB_DEFINE uint32_t"
            )
        ]

        self.assertNotIn("uint32_t s1_overlap_steps;", defs)
        self.assertNotIn("uint32_t transfer_count;", defs)
        self.assertNotIn("uint32_t transfers_per_step;", defs)
        self.assertIn("s2->prefetch_done = 0u;", s1)
        self.assertIn("&s1->compute_done, final + 1u", s1)
        self.assertNotIn("&s2->prefetch_done", s1)
        self.assertNotIn("&s1->compute_done", s2pf_run)
        self.assertNotIn("&s2->compute_done", s2pf_run)
        self.assertNotIn("&s2->prefetch_done", s2pf_run)
        self.assertNotIn("&s2->compute_done", compute)
        self.assertNotIn("&s2->prefetch_done", compute)
        self.assertNotIn("__moe_s2pf_wait_for_phase", s2pf)
        self.assertNotIn("__moe_s2pf_publish_phase", s2pf)
        self.assertNotIn("~0u", s1)
        self.assertNotIn("transfer_count", s2pf)
        self.assertNotIn("transfers_per_step", s2pf)
        self.assertIn("__moe_s2pf_run_both(", s2pf)
        self.assertIn("__moe_s2pf_run_xdma(", s2pf)
        self.assertIn("__moe_s2pf_run_idma(", s2pf)
        self.assertEqual(s2pf_run.count("__moe_dyn_wait_both_2d("), 1)
        self.assertEqual(s2pf_run.count("__moe_dyn_wait_single_2d_xdma("), 1)
        self.assertEqual(s2pf_run.count("__moe_dyn_wait_single_2d_idma("), 1)
        self.assertIn("MOE_DYN_CTRL_S2PF_RUNTIME_RELEASE", s1)
        self.assertIn("MOE_DYN_CTRL_S2PF_RUNTIME_RELEASE", prefetch)
        self.assertIn("MOE_DYN_CTRL_S2PF_EARLY", s1)
        self.assertIn("MOE_DYN_CTRL_S2PF_EARLY", prefetch)
        self.assertIn("&sync->prefetch_done", s3)
        self.assertIn("&sync->compute_done", s3)

    def test_early_s2pf_is_fused_into_s1_dma_gaps(self) -> None:
        s1 = (KERNEL_DIR / "moe_dynamic_stage_s1.h").read_text()
        prefetch = (KERNEL_DIR / "moe_dynamic_stage_prefetch.h").read_text()

        self.assertIn("__moe_run_early_s2pf_group", s1)
        self.assertIn("__moe_load_s1_both_with_early_s2pf", s1)
        self.assertIn("__moe_load_s1_xdma_with_early_s2pf", s1)
        self.assertIn("__moe_load_s1_idma_with_early_s2pf", s1)
        self.assertIn("s2->valid = 0u;", s1)
        self.assertIn("if (s2->valid == 0u)", prefetch)

    def test_s2pf_release_mode_is_lowered_by_all_host_paths(self) -> None:
        abi = (
            SW_DIR / "host/runtime/libbingo/include/libbingo/device_kernel_args.h"
        ).read_text()
        protocol = (KERNEL_DIR / "moe_dynamic_protocol.h").read_text()
        helper = (SW_DIR / "host/runtime/host_moe_s2pf_mode.h").read_text()
        hw = (SW_DIR / "host/runtime/host_moe_hw_path.h").read_text()
        sw = (SW_DIR / "host/runtime/host_moe_sw_path.h").read_text()

        self.assertIn("BINGO_MOE_DYN_CTRL_S2PF_EARLY_BIT 20u", abi)
        self.assertIn("BINGO_MOE_DYN_CTRL_S2PF_RUNTIME_RELEASE_BIT 21u", abi)
        self.assertIn("MOE_DYN_CTRL_S2PF_EARLY", protocol)
        self.assertIn("MOE_DYN_CTRL_S2PF_RUNTIME_RELEASE", protocol)
        self.assertEqual(helper.count("s1_compute_ticks > s1_dma_ticks"), 1)
        for lowering in (hw, sw):
            self.assertIn('#include "host_moe_s2pf_mode.h"', lowering)
            self.assertIn("__host_moe_s2pf_runtime_ctrl(", lowering)
            self.assertNotIn("s1_compute_ticks", lowering)
            self.assertNotIn("s1_dma_ticks", lowering)
            self.assertNotIn("HOST_MOE_DYN_CTRL_S2PF_", lowering)

    def test_hw_scheduler_init_starts_on_last_valid_window(self) -> None:
        hw = (SW_DIR / "host/runtime/host_moe_hw_path.h").read_text()
        init = hw[hw.index("writed(moe_sched_pack_config(") :]
        init = init[: init.index("next_top_pos = hot_count;")]

        self.assertIn("if (rem_count <= 4u)", init)
        self.assertIn("if (rem_count <= 8u)", init)
        self.assertIn("if (rem_count <= 12u)", init)
        self.assertEqual(init.count("MOE_SCHED_WINDOW0"), 2)
        self.assertEqual(init.count("MOE_SCHED_WINDOW1"), 2)
        self.assertEqual(init.count("MOE_SCHED_WINDOW2"), 2)
        self.assertEqual(init.count("MOE_SCHED_WINDOW3_START"), 1)
        self.assertEqual(init.count("moe_sched_fence();"), 5)

    def test_s2pf_xdma_pipeline_contract_is_present(self) -> None:
        defs = (KERNEL_DIR / "moe_dynamic_defs.h").read_text()
        protocol = (KERNEL_DIR / "moe_dynamic_protocol.h").read_text()
        dma = (KERNEL_DIR / "moe_dynamic_dma.h").read_text()
        s1 = (KERNEL_DIR / "moe_dynamic_stage_s1.h").read_text()
        s3 = (KERNEL_DIR / "moe_dynamic_stage_s3.h").read_text()
        prefetch = (KERNEL_DIR / "moe_dynamic_stage_prefetch.h").read_text()
        compute = (KERNEL_DIR / "moe_dynamic_stage_compute.h").read_text()
        store = (KERNEL_DIR / "moe_dynamic_stage_store.h").read_text()

        self.assertIn("MOE_XDMA_PREPARED_S2PF", defs)
        self.assertNotIn("__moe_prepare_after_s2pf_xdma", protocol)
        self.assertIn("__moe_dyn_prepare_after_s2pf_xdma", dma)
        self.assertIn("cfg->s3_call[0].valid", dma)
        self.assertIn("__moe_dyn_prepare_s2pf_xdma(blk)", s1)
        self.assertIn("__moe_prepare_slot_entry_xdma", s1)
        self.assertIn("__moe_dyn_prepare_after_s2pf_xdma(blk, cfg, st)", s1)
        self.assertEqual(
            store.count("__moe_prepare_slot_entry_xdma(&next_blk, next_cfg, st)"),
            2,
        )
        self.assertNotIn("__moe_dyn_prepare_s1_xdma(&next_blk, next_cfg, st)", store)
        self.assertIn("__moe_dyn_prepare_s2pf_both_xdma_address", dma)
        self.assertIn("__moe_dyn_prepare_s2pf_single_xdma_address", dma)
        self.assertIn("__moe_prepare_s2pf_xdma_phase_shape", protocol)
        self.assertIn("__moe_prepare_s2pf_xdma_phase_shape", dma)
        self.assertIn("__moe_prepare_s2pf_xdma_phase_shape", prefetch)
        self.assertIn("__moe_s2pf_start_both_phase", prefetch)
        self.assertIn("__moe_dyn_prepare_after_s2pf_xdma(blk, cfg, st)", prefetch)
        both_start = dma.index("__moe_dyn_start_both_2d_preloaded_xdma(")
        both_end = dma.index("__moe_dyn_start_pair_2d_idma(", both_start)
        both_body = dma[both_start:both_end]
        self.assertLess(
            both_body.index("__moe_dyn_xdma_start_remote_begin()"),
            both_body.index("snrt_dma_start_2d_wideptr("),
        )
        self.assertLess(
            both_body.index("snrt_dma_start_2d_wideptr("),
            both_body.index("__moe_dyn_xdma_start_remote_commit(previous)"),
        )
        self.assertIn("__moe_dyn_xdma_start_remote_begin()", prefetch)
        self.assertIn("__moe_dyn_xdma_start_remote_commit(previous)", prefetch)
        self.assertNotIn("xdma_start()", prefetch)
        self.assertNotIn("xdma_ready", prefetch)
        self.assertNotIn("transfers_per_step", prefetch)
        self.assertNotIn("transfer_count", prefetch)
        self.assertNotIn("__moe_s2pf_run_xdma_mode", prefetch)
        self.assertNotIn("if (both", prefetch)
        self.assertNotIn("__moe_dyn_copy_pair_2d", s1)
        self.assertNotIn("__moe_dyn_copy_pair_2d", s3)
        self.assertNotIn("__moe_dyn_2d_pair_pending_t", s1)
        self.assertNotIn("__moe_dyn_2d_pair_pending_t", s3)
        self.assertNotIn("__moe_dyn_2d_pair_pending_t", prefetch)
        self.assertNotIn("wait_idma", prefetch)
        self.assertNotIn("store_prepared == 0u", store)
        self.assertIn("MOE_S4_CSR_LAYOUT_PHASE_BATCHED, 0u", compute)
        self.assertNotIn(
            "!__moe_csr_stage_is_prepared(blk, MOE_CSR_PREPARED_S4)",
            compute,
        )


if __name__ == "__main__":
    unittest.main()
