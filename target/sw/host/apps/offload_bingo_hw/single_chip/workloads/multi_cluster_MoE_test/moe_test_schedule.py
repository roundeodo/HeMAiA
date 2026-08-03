"""Static schedule profiles for the focused multi-cluster MoE workload."""

from dataclasses import dataclass, replace


BASELINE_PROFILE = "baseline"
S2PF_BOTH_PROFILE = "s2pf_both"
STATIC_DESC_PROFILE = "static_desc"
DYNAMIC_DESC_PROFILE = "dynamic_desc"
HIGH_TO_LOW_PROFILE = "high_to_low"
LOW_TO_HIGH_PROFILE = "low_to_high"
ENDS_INWARD_PROFILE = "ends_inward"
C_TAIL_SMOKE_PROFILE = "c_tail_smoke"
S1_STAGE_SMOKE_PROFILE = "s1_stage_smoke"
SCHEDULE_PROFILES = (
    BASELINE_PROFILE,
    S2PF_BOTH_PROFILE,
    STATIC_DESC_PROFILE,
    DYNAMIC_DESC_PROFILE,
    HIGH_TO_LOW_PROFILE,
    LOW_TO_HIGH_PROFILE,
    ENDS_INWARD_PROFILE,
    C_TAIL_SMOKE_PROFILE,
    S1_STAGE_SMOKE_PROFILE,
)

DMA_NONE = 0
DMA_IDMA = 1
DMA_XDMA = 2
DMA_BOTH = 3

SHAPE_A = 0
SHAPE_B = 1
SHAPE_C = 2
SHAPE_NAMES = ("A", "B", "C")
SHAPE_M = (8, 4, 2)
S1_COMPUTE_TICKS = (8, 4, 2)
S1_DMA_TICKS = (4, 4, 2)
S3_COMPUTE_TICKS = (4, 2, 1)
S3_DMA_TICKS = (2, 2, 1)
S2PF_TICKS = 2
S2PF_BOTH_TICKS = 1
S4PF_SINGLE_TICKS = 4
S4PF_BOTH_TICKS = 2

EXPERT_COUNT = 64
STATIC_DESC_EXPECTED_MAKESPAN_TICKS = 162
DYNAMIC_DESC_EXPECTED_MAKESPAN_TICKS = 159
HIGH_TO_LOW_EXPECTED_MAKESPAN_TICKS = 163
LOW_TO_HIGH_EXPECTED_MAKESPAN_TICKS = 165
ENDS_INWARD_EXPECTED_MAKESPAN_TICKS = 166
# Same mandatory DFG/API allowance used for the accepted high-to-low bound:
# three quarter-ticks per cluster-local slot.
STRUCTURAL_API_QUARTER_TICKS_PER_SLOT = 3
STATIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 711
DYNAMIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 702
HIGH_TO_LOW_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = (
    4 * HIGH_TO_LOW_EXPECTED_MAKESPAN_TICKS
    + 22 * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
)
LOW_TO_HIGH_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = (
    4 * LOW_TO_HIGH_EXPECTED_MAKESPAN_TICKS
    + 22 * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
)
ENDS_INWARD_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = (
    4 * ENDS_INWARD_EXPECTED_MAKESPAN_TICKS
    + 22 * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
)


def select_two_slot_s2pf_binding(
    schedule_profile: str,
    cluster_name: str,
    slot_idx: int,
    baseline_binding: int,
) -> int:
    """Change only the first C0 slot's known-good IDMA S2PF to BOTH."""
    if (
        schedule_profile == S2PF_BOTH_PROFILE
        and cluster_name == "c0"
        and slot_idx == 0
    ):
        if baseline_binding != DMA_IDMA:
            raise ValueError("focused S2PF BOTH source must be the C0 IDMA path")
        return DMA_BOTH
    return baseline_binding
HIGH_TO_LOW_COUNTS = (
    (22, 18, 14)
    + (3,) * 19
    + (2,) * 8
    + (1,) * 13
    + (0,) * 21
)

# Exact token routing exported for
# certified_olmoe_triple_hot_long_cold_tail. Entries E43-E63 are inactive.
STATIC_DESC_TOKEN_IDS_BY_EXPERT = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 22, 33, 48),
    (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 15, 17, 18, 20, 21, 23, 34, 49),
    (5, 7, 9, 11, 13, 14, 16, 17, 19, 20, 22, 23, 34, 49),
    (24, 35, 50), (24, 35, 50), (25, 36, 51), (25, 36, 51),
    (26, 37, 52), (26, 37, 52), (27, 38, 53), (27, 38, 53),
    (28, 39, 54), (28, 39, 54), (29, 40, 55), (29, 40, 55),
    (30, 41, 56), (30, 41, 56), (31, 42, 57), (31, 42, 57),
    (32, 43, 58), (32, 43, 58), (33, 44, 59),
    (44, 59), (45, 60), (45, 60), (46, 61), (46, 61),
    (47, 62), (47, 62), (48, 63),
    (63,), (64,), (64,), (65,), (65,), (66,), (66,),
    (67,), (67,), (68,), (68,), (69,), (69,),
    *(() for _ in range(21)),
)

# Literal cluster-local streams from case 0 / STATIC_DESC. Physical parameters
# are intentionally absent here because this policy fixes every task to B/B,
# C2 to IDMA, C3 to XDMA, and disables S2PF/S4PF.
STATIC_DESC_CLUSTER_EIDS = {
    "c0": (0, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
           31, 33, 35, 37, 39, 41),
    "c1": (1, 2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
           32, 34, 36, 38, 40, 42),
}


@dataclass(frozen=True)
class DescendingTask:
    """One task from fixed_orders.descending in the distilled showcase."""

    cluster_name: str
    expert_id: int
    ntokens: int
    start_tick: int
    end_tick: int
    s1_shape: int
    s3_shape: int
    s2_prefetch_dma: int = DMA_NONE


# Literal replay of Idea_Model/results/policy_search/
# scheduler_rtl_distilled_showcase.json:fixed_orders.descending.  The order,
# assignment, shapes, prefetch choices, and resource stalls are all intentional.
HIGH_TO_LOW_HISTORY = (
    DescendingTask("c0", 0, 22, 0, 33, SHAPE_A, SHAPE_B, DMA_IDMA),
    DescendingTask("c1", 1, 18, 0, 27, SHAPE_A, SHAPE_B, DMA_XDMA),
    DescendingTask("c1", 2, 14, 27, 48, SHAPE_A, SHAPE_B, DMA_XDMA),
    DescendingTask("c0", 3, 3, 33, 43, SHAPE_A, SHAPE_B),
    DescendingTask("c0", 4, 3, 43, 49, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 5, 3, 48, 54, SHAPE_B, SHAPE_B),
    DescendingTask("c0", 6, 3, 49, 55, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 7, 3, 54, 60, SHAPE_B, SHAPE_B),
    DescendingTask("c0", 8, 3, 55, 61, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 9, 3, 60, 66, SHAPE_B, SHAPE_B),
    DescendingTask("c0", 10, 3, 61, 67, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 11, 3, 66, 72, SHAPE_B, SHAPE_B),
    DescendingTask("c0", 12, 3, 67, 73, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 13, 3, 72, 78, SHAPE_B, SHAPE_B),
    DescendingTask("c0", 14, 3, 73, 79, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 15, 3, 78, 84, SHAPE_B, SHAPE_B),
    DescendingTask("c0", 16, 3, 79, 85, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 17, 3, 84, 90, SHAPE_B, SHAPE_B),
    DescendingTask("c0", 18, 3, 85, 91, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 19, 3, 90, 96, SHAPE_B, SHAPE_B),
    DescendingTask("c0", 20, 3, 91, 97, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 21, 3, 96, 102, SHAPE_B, SHAPE_B),
    DescendingTask("c0", 22, 2, 97, 103, SHAPE_B, SHAPE_B),
    DescendingTask("c1", 23, 2, 103, 106, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 24, 2, 106, 109, SHAPE_C, SHAPE_C),
    DescendingTask("c1", 25, 2, 109, 112, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 26, 2, 112, 115, SHAPE_C, SHAPE_C),
    DescendingTask("c1", 27, 2, 115, 118, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 28, 2, 118, 121, SHAPE_C, SHAPE_C),
    DescendingTask("c1", 29, 2, 121, 124, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 30, 1, 124, 127, SHAPE_C, SHAPE_C),
    DescendingTask("c1", 31, 1, 127, 130, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 32, 1, 130, 133, SHAPE_C, SHAPE_C),
    DescendingTask("c1", 33, 1, 133, 136, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 34, 1, 136, 139, SHAPE_C, SHAPE_C),
    DescendingTask("c1", 35, 1, 139, 142, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 36, 1, 142, 145, SHAPE_C, SHAPE_C),
    DescendingTask("c1", 37, 1, 145, 148, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 38, 1, 148, 151, SHAPE_C, SHAPE_C),
    DescendingTask("c1", 39, 1, 151, 154, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 40, 1, 154, 157, SHAPE_C, SHAPE_C),
    DescendingTask("c1", 41, 1, 157, 160, SHAPE_C, SHAPE_C),
    DescendingTask("c0", 42, 1, 160, 163, SHAPE_C, SHAPE_C),
)

# The first cross-cluster stall is E22 -> E23.  From there every C/C task uses
# BOTH continuously, so this is one global DMA release chain.
HIGH_TO_LOW_DMA_SERIAL_EIDS = tuple(range(22, 43))


@dataclass(frozen=True)
class FixedOrderTask:
    """One task from a fixed-order FPGA showcase handoff stream."""

    cluster_name: str
    expert_id: int
    token_start_rank: int
    ntokens: int
    start_tick: int
    end_tick: int
    s1_shape: int
    s3_shape: int
    s2_prefetch_dma: int
    s1_dma: int
    s3_dma: int
    s4_prefetch_dma: int = DMA_NONE
    s4_prefetch_target_eid: int = -1
    skip_s1: bool = False


def _dynamic_desc_task(
    cluster_name: str,
    expert_id: int,
    start_tick: int,
    end_tick: int,
    s1_shape: int,
    s3_shape: int,
    s2_prefetch_dma: int,
    s1_dma: int,
    s3_dma: int,
    s4_prefetch_dma: int = DMA_NONE,
    s4_prefetch_target_eid: int = -1,
    skip_s1: bool = False,
) -> FixedOrderTask:
    return FixedOrderTask(
        cluster_name,
        expert_id,
        0,
        HIGH_TO_LOW_COUNTS[expert_id],
        start_tick,
        end_tick,
        s1_shape,
        s3_shape,
        s2_prefetch_dma,
        s1_dma,
        s3_dma,
        s4_prefetch_dma,
        s4_prefetch_target_eid,
        skip_s1,
    )


# Literal cluster-local replay of case 0 / DYNAMIC_DESC. The issue order is
# descending, while shape, DMA, S2PF, S4PF, and cache-hit fields are selected
# dynamically by the exported policy.
DYNAMIC_DESC_HISTORY = (
    _dynamic_desc_task("c0", 0, 0, 33, SHAPE_A, SHAPE_B, DMA_IDMA, DMA_IDMA, DMA_NONE),
    _dynamic_desc_task("c0", 3, 33, 39, SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE, DMA_BOTH, 4),
    _dynamic_desc_task("c0", 4, 39, 45, SHAPE_A, SHAPE_B, DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True),
    *(
        _dynamic_desc_task(
            "c0", expert_id, 45 + slot * 6, 51 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        )
        for slot, expert_id in enumerate(range(5, 22, 2))
    ),
    *(
        _dynamic_desc_task(
            "c0", expert_id, 99 + slot * 6, 105 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA,
        )
        for slot, expert_id in enumerate(range(23, 43, 2))
    ),
    _dynamic_desc_task("c1", 1, 0, 27, SHAPE_A, SHAPE_B, DMA_XDMA, DMA_XDMA, DMA_NONE, DMA_XDMA, 2),
    _dynamic_desc_task("c1", 2, 27, 48, SHAPE_A, SHAPE_B, DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True),
    *(
        _dynamic_desc_task(
            "c1", expert_id, 48 + slot * 6, 54 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        )
        for slot, expert_id in enumerate(range(6, 22, 2))
    ),
    _dynamic_desc_task("c1", 22, 96, 99, SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH),
    *(
        _dynamic_desc_task(
            "c1", expert_id, 99 + slot * 6, 105 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA,
        )
        for slot, expert_id in enumerate(range(24, 43, 2))
    ),
)


# Literal cluster-local replay of certified_triple_hot_long_cold_tail/ascending.
# E0 is the final 11+11 SPLIT and therefore appears once on each cluster.
LOW_TO_HIGH_HISTORY = (
    FixedOrderTask("c0", 41, 0, 1, 0, 6, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 39, 0, 1, 6, 12, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 37, 0, 1, 12, 18, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 35, 0, 1, 18, 24, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 33, 0, 1, 24, 30, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 31, 0, 1, 30, 36, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 29, 0, 2, 36, 42, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 27, 0, 2, 42, 48, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 25, 0, 2, 48, 54, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 23, 0, 2, 54, 60, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 21, 0, 3, 60, 66, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 19, 0, 3, 66, 72, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 17, 0, 3, 72, 78, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 15, 0, 3, 78, 84, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 13, 0, 3, 84, 90, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 11, 0, 3, 90, 96, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 9, 0, 3, 96, 102, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 7, 0, 3, 102, 108, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 5, 0, 3, 108, 114, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 3, 0, 3, 114, 120, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 1, 0, 18, 120, 147, SHAPE_A, SHAPE_B, DMA_BOTH, DMA_IDMA, DMA_NONE),
    FixedOrderTask("c0", 0, 0, 11, 147, 165, SHAPE_A, SHAPE_B, DMA_IDMA, DMA_IDMA, DMA_NONE),
    FixedOrderTask("c1", 42, 0, 1, 0, 6, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 40, 0, 1, 6, 12, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 38, 0, 1, 12, 18, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 36, 0, 1, 18, 24, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 34, 0, 1, 24, 30, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 32, 0, 1, 30, 36, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 30, 0, 1, 36, 42, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 28, 0, 2, 42, 48, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 26, 0, 2, 48, 54, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 24, 0, 2, 54, 60, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 22, 0, 2, 60, 66, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 20, 0, 3, 66, 72, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 18, 0, 3, 72, 78, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 16, 0, 3, 78, 84, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 14, 0, 3, 84, 90, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 12, 0, 3, 90, 96, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 10, 0, 3, 96, 102, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 8, 0, 3, 102, 108, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 6, 0, 3, 108, 114, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 4, 0, 3, 114, 120, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 2, 0, 14, 120, 144, SHAPE_C, SHAPE_C, DMA_NONE, DMA_XDMA, DMA_IDMA),
    FixedOrderTask("c1", 0, 11, 11, 147, 165, SHAPE_A, SHAPE_B, DMA_XDMA, DMA_XDMA, DMA_NONE),
)

# Literal cluster-local replay of certified_triple_hot_long_cold_tail/ends_inward.
ENDS_INWARD_HISTORY = (
    FixedOrderTask("c0", 0, 0, 22, 0, 33, SHAPE_A, SHAPE_B, DMA_BOTH, DMA_IDMA, DMA_NONE),
    FixedOrderTask("c0", 41, 0, 1, 33, 37, SHAPE_C, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE),
    FixedOrderTask("c0", 2, 0, 14, 37, 58, SHAPE_A, SHAPE_B, DMA_BOTH, DMA_IDMA, DMA_NONE),
    FixedOrderTask("c0", 4, 0, 3, 58, 64, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 5, 0, 3, 64, 70, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 6, 0, 3, 70, 76, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 7, 0, 3, 76, 82, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 8, 0, 3, 82, 88, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 9, 0, 3, 88, 94, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 10, 0, 3, 94, 100, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 11, 0, 3, 100, 106, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 12, 0, 3, 106, 112, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 13, 0, 3, 112, 118, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 14, 0, 3, 118, 124, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 15, 0, 3, 124, 130, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 16, 0, 3, 130, 136, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 17, 0, 3, 136, 142, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 18, 0, 3, 142, 148, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 19, 0, 3, 148, 154, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 20, 0, 3, 154, 160, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c0", 21, 0, 3, 160, 166, SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA),
    FixedOrderTask("c1", 42, 0, 1, 0, 10, SHAPE_A, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_BOTH),
    FixedOrderTask("c1", 1, 0, 18, 10, 37, SHAPE_A, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE),
    FixedOrderTask("c1", 40, 0, 1, 37, 47, SHAPE_A, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_BOTH),
    FixedOrderTask("c1", 3, 0, 3, 47, 58, SHAPE_A, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_IDMA),
    FixedOrderTask("c1", 39, 0, 1, 58, 64, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 38, 0, 1, 64, 70, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 37, 0, 1, 70, 76, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 36, 0, 1, 76, 82, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 35, 0, 1, 82, 88, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 34, 0, 1, 88, 94, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 33, 0, 1, 94, 100, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 32, 0, 1, 100, 106, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 31, 0, 1, 106, 112, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 30, 0, 1, 112, 118, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 29, 0, 2, 118, 124, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 28, 0, 2, 124, 130, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 27, 0, 2, 130, 136, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 26, 0, 2, 136, 142, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 25, 0, 2, 142, 148, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 24, 0, 2, 148, 154, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 23, 0, 2, 154, 160, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
    FixedOrderTask("c1", 22, 0, 2, 160, 166, SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA),
)


@dataclass(frozen=True)
class SlotSpec:
    cluster_name: str
    cluster_index: int
    local_slot: int
    expert_id: int
    ntokens: int
    profile: str
    s1_shape: int
    s3_shape: int
    s2_prefetch_dma: int
    reference_start_tick: int
    reference_end_tick: int
    token_start_rank: int = 0
    s1_dma_override: int = -1
    s3_dma_override: int = -1
    s4_prefetch_dma: int = DMA_NONE
    s4_prefetch_target_eid: int = -1
    skip_s1: bool = False

    @property
    def single_dma(self) -> int:
        return DMA_IDMA if self.cluster_index == 0 else DMA_XDMA

    @property
    def skip_s3(self) -> bool:
        return self.s2_prefetch_dma != DMA_NONE

    @property
    def s1_dma(self) -> int:
        if self.s1_dma_override >= 0:
            return self.s1_dma_override
        if self.skip_s1:
            return DMA_NONE
        return DMA_BOTH if self.s1_shape == SHAPE_C else self.single_dma

    @property
    def s3_dma(self) -> int:
        if self.s3_dma_override >= 0:
            return self.s3_dma_override
        if self.skip_s3:
            return DMA_NONE
        return DMA_BOTH if self.s3_shape == SHAPE_C else self.single_dma

    @property
    def s2_token_start(self) -> int:
        return 0 if self.skip_s1 else SHAPE_M[self.s1_shape]

    @property
    def s2_m_exec(self) -> int:
        remaining = max(self.ntokens - self.s2_token_start, 0)
        return (remaining + SHAPE_M[SHAPE_C] - 1) // SHAPE_M[SHAPE_C]

    @property
    def s4_token_start(self) -> int:
        return 0 if self.skip_s3 else SHAPE_M[self.s3_shape]

    @property
    def s4_m_exec(self) -> int:
        remaining = max(self.ntokens - self.s4_token_start, 0)
        return (remaining + SHAPE_M[SHAPE_C] - 1) // SHAPE_M[SHAPE_C]


@dataclass(frozen=True)
class TaskTimeline:
    start: int
    task_end: int
    dma1_end: int
    s2_end: int
    dma3_end: int


def _task_timeline(slot: SlotSpec, start: int) -> TaskTimeline:
    if slot.skip_s1:
        dma1_end = start
        s2_end = start + 2 * ((slot.ntokens + 1) // 2)
    else:
        tail = max(slot.ntokens - SHAPE_M[slot.s1_shape], 0)
        dma_ticks = 2 if slot.s1_dma == DMA_BOTH else 4
        dma1_end = start + dma_ticks
        s2_end = start + max(S1_COMPUTE_TICKS[slot.s1_shape], dma_ticks) + 2 * (
            (tail + 1) // 2
        )

    if slot.skip_s3:
        prefetch_ticks = (
            S2PF_BOTH_TICKS
            if slot.s2_prefetch_dma == DMA_BOTH
            else S2PF_TICKS
        )
        dma3_end = max(s2_end, dma1_end + prefetch_ticks)
        task_end = dma3_end + ((slot.ntokens + 1) // 2)
    else:
        tail = max(slot.ntokens - SHAPE_M[slot.s3_shape], 0)
        dma_ticks = 1 if slot.s3_dma == DMA_BOTH else 2
        dma3_end = s2_end + dma_ticks
        task_end = s2_end + max(S3_COMPUTE_TICKS[slot.s3_shape], dma_ticks) + (
            (tail + 1) // 2
        )
    return TaskTimeline(start, task_end, dma1_end, s2_end, dma3_end)


def _profile(task: DescendingTask) -> str:
    suffix = "+S2PF" if task.s2_prefetch_dma != DMA_NONE else ""
    return f"{SHAPE_NAMES[task.s1_shape]}/{SHAPE_NAMES[task.s3_shape]}{suffix}"


def build_static_desc_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    queues = {"c0": [], "c1": []}
    for cluster_name in ("c0", "c1"):
        cluster_index = 0 if cluster_name == "c0" else 1
        dma = DMA_IDMA if cluster_index == 0 else DMA_XDMA
        start_tick = 0
        for expert_id in STATIC_DESC_CLUSTER_EIDS[cluster_name]:
            slot = SlotSpec(
                cluster_name=cluster_name,
                cluster_index=cluster_index,
                local_slot=len(queues[cluster_name]),
                expert_id=expert_id,
                ntokens=HIGH_TO_LOW_COUNTS[expert_id],
                profile="B/B",
                s1_shape=SHAPE_B,
                s3_shape=SHAPE_B,
                s2_prefetch_dma=DMA_NONE,
                reference_start_tick=start_tick,
                reference_end_tick=-1,
                s1_dma_override=dma,
                s3_dma_override=dma,
            )
            end_tick = _task_timeline(slot, start_tick).task_end
            queues[cluster_name].append(
                replace(slot, reference_end_tick=end_tick)
            )
            start_tick = end_tick
    frozen = {name: tuple(slots) for name, slots in queues.items()}
    audit_static_desc_schedule(frozen)
    return frozen


def build_dynamic_desc_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    queues = {"c0": [], "c1": []}
    for task in DYNAMIC_DESC_HISTORY:
        cluster_index = 0 if task.cluster_name == "c0" else 1
        queues[task.cluster_name].append(
            SlotSpec(
                cluster_name=task.cluster_name,
                cluster_index=cluster_index,
                local_slot=len(queues[task.cluster_name]),
                expert_id=task.expert_id,
                ntokens=task.ntokens,
                profile=(
                    f"{SHAPE_NAMES[task.s1_shape]}/"
                    f"{SHAPE_NAMES[task.s3_shape]}"
                    f"{'+S2PF' if task.s2_prefetch_dma != DMA_NONE else ''}"
                    f"{'+S4PF' if task.s4_prefetch_dma != DMA_NONE else ''}"
                ),
                s1_shape=task.s1_shape,
                s3_shape=task.s3_shape,
                s2_prefetch_dma=task.s2_prefetch_dma,
                reference_start_tick=task.start_tick,
                reference_end_tick=task.end_tick,
                token_start_rank=task.token_start_rank,
                s1_dma_override=task.s1_dma,
                s3_dma_override=task.s3_dma,
                s4_prefetch_dma=task.s4_prefetch_dma,
                s4_prefetch_target_eid=task.s4_prefetch_target_eid,
                skip_s1=task.skip_s1,
            )
        )
    frozen = {name: tuple(slots) for name, slots in queues.items()}
    audit_dynamic_desc_schedule(frozen)
    return frozen


def build_high_to_low_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    queues = {"c0": [], "c1": []}
    for task in HIGH_TO_LOW_HISTORY:
        cluster_index = 0 if task.cluster_name == "c0" else 1
        queues[task.cluster_name].append(
            SlotSpec(
                cluster_name=task.cluster_name,
                cluster_index=cluster_index,
                local_slot=len(queues[task.cluster_name]),
                expert_id=task.expert_id,
                ntokens=task.ntokens,
                profile=_profile(task),
                s1_shape=task.s1_shape,
                s3_shape=task.s3_shape,
                s2_prefetch_dma=task.s2_prefetch_dma,
                reference_start_tick=task.start_tick,
                reference_end_tick=task.end_tick,
            )
        )
    frozen = {name: tuple(slots) for name, slots in queues.items()}
    audit_high_to_low_schedule(frozen)
    return frozen


def build_low_to_high_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    queues = {"c0": [], "c1": []}
    for task in LOW_TO_HIGH_HISTORY:
        cluster_index = 0 if task.cluster_name == "c0" else 1
        queues[task.cluster_name].append(
            SlotSpec(
                cluster_name=task.cluster_name,
                cluster_index=cluster_index,
                local_slot=len(queues[task.cluster_name]),
                expert_id=task.expert_id,
                ntokens=task.ntokens,
                profile=(
                    f"{SHAPE_NAMES[task.s1_shape]}/"
                    f"{SHAPE_NAMES[task.s3_shape]}"
                    f"{'+S2PF' if task.s2_prefetch_dma != DMA_NONE else ''}"
                ),
                s1_shape=task.s1_shape,
                s3_shape=task.s3_shape,
                s2_prefetch_dma=task.s2_prefetch_dma,
                reference_start_tick=task.start_tick,
                reference_end_tick=task.end_tick,
                token_start_rank=task.token_start_rank,
                s1_dma_override=task.s1_dma,
                s3_dma_override=task.s3_dma,
            )
        )
    frozen = {name: tuple(slots) for name, slots in queues.items()}
    audit_low_to_high_schedule(frozen)
    return frozen


def build_ends_inward_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    queues = {"c0": [], "c1": []}
    for task in ENDS_INWARD_HISTORY:
        cluster_index = 0 if task.cluster_name == "c0" else 1
        queues[task.cluster_name].append(
            SlotSpec(
                cluster_name=task.cluster_name,
                cluster_index=cluster_index,
                local_slot=len(queues[task.cluster_name]),
                expert_id=task.expert_id,
                ntokens=task.ntokens,
                profile=(
                    f"{SHAPE_NAMES[task.s1_shape]}/"
                    f"{SHAPE_NAMES[task.s3_shape]}"
                    f"{'+S2PF' if task.s2_prefetch_dma != DMA_NONE else ''}"
                ),
                s1_shape=task.s1_shape,
                s3_shape=task.s3_shape,
                s2_prefetch_dma=task.s2_prefetch_dma,
                reference_start_tick=task.start_tick,
                reference_end_tick=task.end_tick,
                token_start_rank=task.token_start_rank,
                s1_dma_override=task.s1_dma,
                s3_dma_override=task.s3_dma,
            )
        )
    frozen = {name: tuple(slots) for name, slots in queues.items()}
    audit_ends_inward_schedule(frozen)
    return frozen


def build_c_tail_smoke_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    """Minimal 2-token C/C BOTH-DMA diagnostic schedule."""
    c1 = SlotSpec(
        cluster_name="c1",
        cluster_index=1,
        local_slot=0,
        expert_id=23,
        ntokens=2,
        profile="C/C",
        s1_shape=SHAPE_C,
        s3_shape=SHAPE_C,
        s2_prefetch_dma=DMA_NONE,
        reference_start_tick=0,
        reference_end_tick=3,
    )
    c0 = SlotSpec(
        cluster_name="c0",
        cluster_index=0,
        local_slot=0,
        expert_id=24,
        ntokens=2,
        profile="C/C",
        s1_shape=SHAPE_C,
        s3_shape=SHAPE_C,
        s2_prefetch_dma=DMA_NONE,
        reference_start_tick=3,
        reference_end_tick=6,
    )
    return {"c0": (c0,), "c1": (c1,)}


def build_s1_stage_smoke_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    """Minimal S1-only version of the 2-token C/C BOTH-DMA tail."""
    return build_c_tail_smoke_schedule()


def build_schedule_profile(profile: str) -> dict[str, tuple[SlotSpec, ...]]:
    if profile == STATIC_DESC_PROFILE:
        return build_static_desc_schedule()
    if profile == DYNAMIC_DESC_PROFILE:
        return build_dynamic_desc_schedule()
    if profile == HIGH_TO_LOW_PROFILE:
        return build_high_to_low_schedule()
    if profile == LOW_TO_HIGH_PROFILE:
        return build_low_to_high_schedule()
    if profile == ENDS_INWARD_PROFILE:
        return build_ends_inward_schedule()
    if profile == C_TAIL_SMOKE_PROFILE:
        return build_c_tail_smoke_schedule()
    if profile == S1_STAGE_SMOKE_PROFILE:
        return build_s1_stage_smoke_schedule()
    raise ValueError(f"profile {profile!r} does not use SlotSpec queues")


def _dma_intervals(slot: SlotSpec) -> tuple[tuple[int, int, int, str], ...]:
    timeline = _task_timeline(slot, slot.reference_start_tick)
    intervals = []
    if slot.s1_dma != DMA_NONE:
        intervals.append(
            (timeline.start, timeline.dma1_end, slot.s1_dma, "S1")
        )
    if slot.s2_prefetch_dma != DMA_NONE:
        prefetch_ticks = (
            S2PF_BOTH_TICKS
            if slot.s2_prefetch_dma == DMA_BOTH
            else S2PF_TICKS
        )
        intervals.append(
            (
                timeline.dma1_end,
                timeline.dma1_end + prefetch_ticks,
                slot.s2_prefetch_dma,
                "S2PF",
            )
        )
    if slot.s3_dma != DMA_NONE:
        intervals.append(
            (timeline.s2_end, timeline.dma3_end, slot.s3_dma, "S3")
        )
    if slot.s4_prefetch_dma != DMA_NONE:
        prefetch_ticks = (
            S4PF_BOTH_TICKS
            if slot.s4_prefetch_dma == DMA_BOTH
            else S4PF_SINGLE_TICKS
        )
        intervals.append(
            (
                timeline.dma3_end,
                timeline.dma3_end + prefetch_ticks,
                slot.s4_prefetch_dma,
                "S4PF",
            )
        )
    return tuple(intervals)


def cross_cluster_dma_release_edges(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> tuple[tuple[tuple[str, int, int, str], tuple[str, int, int, str]], ...]:
    intervals = [
        (start, end, dma, slot.cluster_name, slot.local_slot, slot.expert_id, stage)
        for slots in queues.values()
        for slot in slots
        for start, end, dma, stage in _dma_intervals(slot)
    ]
    edges = set()
    for lane in (DMA_IDMA, DMA_XDMA):
        lane_intervals = sorted(
            (interval for interval in intervals if interval[2] & lane),
            key=lambda interval: (interval[0], interval[1], interval[3:]),
        )
        for previous, current in zip(lane_intervals, lane_intervals[1:]):
            if previous[1] > current[0]:
                raise AssertionError(
                    f"DMA lane {lane} overlaps: {previous} and {current}"
                )
            if previous[3] == current[3]:
                continue
            edges.add(
                (
                    (previous[3], previous[4], previous[5], previous[6]),
                    (current[3], current[4], current[5], current[6]),
                )
            )
    return tuple(sorted(edges))


def audit_static_desc_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(HIGH_TO_LOW_COUNTS) != EXPERT_COUNT or sum(HIGH_TO_LOW_COUNTS) != 140:
        raise AssertionError("STATIC_DESC distribution contract changed")
    if tuple(len(STATIC_DESC_TOKEN_IDS_BY_EXPERT[eid]) for eid in range(EXPERT_COUNT)) != HIGH_TO_LOW_COUNTS:
        raise AssertionError("STATIC_DESC token routing does not match expert loads")

    token_owners = [[] for _ in range(70)]
    for expert_id, token_ids in enumerate(STATIC_DESC_TOKEN_IDS_BY_EXPERT):
        for token_id in token_ids:
            if not 0 <= token_id < len(token_owners):
                raise AssertionError(f"E{expert_id} has invalid token {token_id}")
            token_owners[token_id].append(expert_id)
    if any(len(owners) != 2 for owners in token_owners):
        raise AssertionError("STATIC_DESC must route every input token twice")

    if tuple(len(queues[name]) for name in ("c0", "c1")) != (22, 21):
        raise AssertionError("STATIC_DESC must contain C2=22 and C3=21 slots")
    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 43 or set(slots_by_eid) != set(range(43)):
        raise AssertionError("STATIC_DESC must cover E0 through E42 once")
    for cluster_name in ("c0", "c1"):
        if tuple(slot.expert_id for slot in queues[cluster_name]) != STATIC_DESC_CLUSTER_EIDS[cluster_name]:
            raise AssertionError(f"{cluster_name} STATIC_DESC stream changed")

    for expert_id, slot in slots_by_eid.items():
        expected_dma = DMA_IDMA if slot.cluster_name == "c0" else DMA_XDMA
        if slot.ntokens != HIGH_TO_LOW_COUNTS[expert_id]:
            raise AssertionError(f"E{expert_id} token count changed")
        if (
            slot.s1_shape != SHAPE_B
            or slot.s3_shape != SHAPE_B
            or slot.s1_dma != expected_dma
            or slot.s3_dma != expected_dma
            or slot.s2_prefetch_dma != DMA_NONE
            or slot.s4_prefetch_dma != DMA_NONE
            or slot.skip_s1
            or slot.skip_s3
        ):
            raise AssertionError(f"E{expert_id} is not fixed B/B without prefetch")
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(f"E{expert_id} STATIC_DESC timeline changed")

    for cluster_name, slots in queues.items():
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} STATIC_DESC stream has a gap")

    if cross_cluster_dma_release_edges(queues):
        raise AssertionError("STATIC_DESC dedicated DMA lanes need no release edges")
    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 159, "c1": 162}:
        raise AssertionError(f"STATIC_DESC queue endpoints changed: {queue_ticks}")
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if structural_quarters != STATIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS:
        raise AssertionError("STATIC_DESC structural lower bound must be 177.75 ticks")
    return {
        "distribution": HIGH_TO_LOW_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(HIGH_TO_LOW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": (),
    }


def audit_dynamic_desc_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(HIGH_TO_LOW_COUNTS) != EXPERT_COUNT or sum(HIGH_TO_LOW_COUNTS) != 140:
        raise AssertionError("DYNAMIC_DESC distribution contract changed")
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (22, 21):
        raise AssertionError("DYNAMIC_DESC must contain C2=22 and C3=21 slots")

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 43 or set(slots_by_eid) != set(range(43)):
        raise AssertionError("DYNAMIC_DESC must cover E0 through E42 once")
    for cluster_name in ("c0", "c1"):
        if tuple(slot.expert_id for slot in queues[cluster_name]) != (
            STATIC_DESC_CLUSTER_EIDS[cluster_name]
        ):
            raise AssertionError(f"{cluster_name} DYNAMIC_DESC stream changed")

    for expert_id, slot in slots_by_eid.items():
        if slot.ntokens != HIGH_TO_LOW_COUNTS[expert_id]:
            raise AssertionError(f"E{expert_id} token count changed")
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{expert_id} timeline ends at {timeline.task_end}, "
                f"expected {slot.reference_end_tick}"
            )
        if slot.s4_prefetch_dma != DMA_NONE:
            s4pf_ticks = (
                S4PF_BOTH_TICKS
                if slot.s4_prefetch_dma == DMA_BOTH
                else S4PF_SINGLE_TICKS
            )
            if timeline.dma3_end + s4pf_ticks > timeline.task_end:
                raise AssertionError(f"E{expert_id} S4PF exceeds its task")

    expected_s2pf_eids = {0, 1, 3, *range(5, 22)}
    actual_s2pf_eids = {
        slot.expert_id
        for slot in all_slots
        if slot.s2_prefetch_dma != DMA_NONE
    }
    if actual_s2pf_eids != expected_s2pf_eids:
        raise AssertionError("DYNAMIC_DESC S2PF choices changed")
    expected_s4pf = {
        1: (DMA_XDMA, 2),
        3: (DMA_BOTH, 4),
    }
    actual_s4pf = {
        slot.expert_id: (
            slot.s4_prefetch_dma,
            slot.s4_prefetch_target_eid,
        )
        for slot in all_slots
        if slot.s4_prefetch_dma != DMA_NONE
    }
    if actual_s4pf != expected_s4pf:
        raise AssertionError("DYNAMIC_DESC S4PF choices changed")
    if {slot.expert_id for slot in all_slots if slot.skip_s1} != {2, 4}:
        raise AssertionError("DYNAMIC_DESC S4PF cache hits changed")
    if (
        slots_by_eid[22].s1_shape,
        slots_by_eid[22].s3_shape,
        slots_by_eid[22].s1_dma,
        slots_by_eid[22].s3_dma,
    ) != (SHAPE_C, SHAPE_C, DMA_BOTH, DMA_BOTH):
        raise AssertionError("DYNAMIC_DESC E22 must use C/C with BOTH")

    for cluster_name, slots in queues.items():
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} DYNAMIC_DESC stream has a gap")

    intervals = [
        (start, end, dma, slot.cluster_name, slot.expert_id, stage)
        for slot in all_slots
        for start, end, dma, stage in _dma_intervals(slot)
    ]
    if len(intervals) != 86:
        raise AssertionError(f"DYNAMIC_DESC must contain 86 DMA ops, got {len(intervals)}")
    for index, left in enumerate(intervals):
        for right in intervals[index + 1:]:
            if left[2] & right[2] and max(left[0], right[0]) < min(left[1], right[1]):
                raise AssertionError(
                    f"DMA overlap {left[3]}/E{left[4]}:{left[5]} {left[:2]} "
                    f"and {right[3]}/E{right[4]}:{right[5]} {right[:2]}"
                )

    release_edges = cross_cluster_dma_release_edges(queues)
    if len(release_edges) != 21:
        raise AssertionError(
            f"DYNAMIC_DESC must contain 21 DMA release edges, got {len(release_edges)}"
        )
    required_edges = {
        (("c1", 0, 1, "S4PF"), ("c0", 1, 3, "S1")),
        (("c0", 1, 3, "S4PF"), ("c1", 1, 2, "S3")),
        (("c1", 1, 2, "S3"), ("c0", 2, 4, "S3")),
        (("c0", 11, 21, "S2PF"), ("c1", 10, 22, "S1")),
        (("c1", 10, 22, "S3"), ("c0", 12, 23, "S1")),
    }
    if not required_edges.issubset(set(release_edges)):
        raise AssertionError("DYNAMIC_DESC key DMA release edges changed")

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 159, "c1": 159}:
        raise AssertionError(f"DYNAMIC_DESC queue endpoints changed: {queue_ticks}")
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if structural_quarters != DYNAMIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS:
        raise AssertionError("DYNAMIC_DESC structural lower bound must be 175.5 ticks")
    return {
        "distribution": HIGH_TO_LOW_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(HIGH_TO_LOW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_high_to_low_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(HIGH_TO_LOW_COUNTS) != EXPERT_COUNT:
        raise AssertionError("high-to-low distribution must contain 64 experts")
    if sum(HIGH_TO_LOW_COUNTS) != 140:
        raise AssertionError("high-to-low distribution must contain 140 routes")

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 43 or set(slots_by_eid) != set(range(43)):
        raise AssertionError("descending history must cover E0 through E42 once")
    if tuple(slot.expert_id for slot in queues["c0"]) != (
        0, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
        24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
    ):
        raise AssertionError("C0 descending assignment diverged from the showcase")
    if tuple(slot.expert_id for slot in queues["c1"]) != (
        1, 2, 5, 7, 9, 11, 13, 15, 17, 19, 21,
        23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
    ):
        raise AssertionError("C1 descending assignment diverged from the showcase")

    for expert_id, ntokens in enumerate(HIGH_TO_LOW_COUNTS[:43]):
        if slots_by_eid[expert_id].ntokens != ntokens:
            raise AssertionError(f"E{expert_id} token count mismatch")

    expected_s2pf = {
        0: DMA_IDMA,
        1: DMA_XDMA,
        2: DMA_XDMA,
    }
    actual_s2pf = {
        slot.expert_id: slot.s2_prefetch_dma
        for slot in all_slots
        if slot.s2_prefetch_dma != DMA_NONE
    }
    if actual_s2pf != expected_s2pf:
        raise AssertionError("descending S2 prefetch choices diverged")
    if (slots_by_eid[3].s1_shape, slots_by_eid[3].s3_shape) != (
        SHAPE_A,
        SHAPE_B,
    ):
        raise AssertionError("E3 must use the special A/B profile")
    if (slots_by_eid[22].s1_shape, slots_by_eid[22].s3_shape) != (
        SHAPE_B,
        SHAPE_B,
    ):
        raise AssertionError("E22 must remain B/B before the BOTH-DMA tail")
    if HIGH_TO_LOW_DMA_SERIAL_EIDS != tuple(range(22, 43)):
        raise AssertionError("descending DMA release chain must cover E22-E42")

    for slot in all_slots:
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{slot.expert_id} timeline ends at {timeline.task_end}, "
                f"expected {slot.reference_end_tick}"
            )
        if slot.s4_prefetch_dma != DMA_NONE or slot.skip_s1:
            raise AssertionError("descending showcase does not use S4 prefetch")

    for cluster_name, slots in queues.items():
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick < previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} local tasks overlap")

    intervals = [
        (start, end, dma, slot.expert_id, stage)
        for slot in all_slots
        for start, end, dma, stage in _dma_intervals(slot)
    ]
    for index, left in enumerate(intervals):
        for right in intervals[index + 1:]:
            if left[2] & right[2] and max(left[0], right[0]) < min(left[1], right[1]):
                raise AssertionError(
                    f"DMA overlap E{left[3]}:{left[4]} {left[:2]} and "
                    f"E{right[3]}:{right[4]} {right[:2]}"
                )

    if slots_by_eid[21].reference_end_tick != 102:
        raise AssertionError("E21 must release C1 at tick 102")
    if slots_by_eid[22].reference_end_tick != 103:
        raise AssertionError("E22 must hold iDMA through tick 103")
    if slots_by_eid[23].reference_start_tick != 103:
        raise AssertionError("E23 must contain the explicit 102-to-103 DMA stall")

    queue_ticks = {
        name: max(slot.reference_end_tick for slot in slots)
        for name, slots in queues.items()
    }
    makespan = max(queue_ticks.values())
    if makespan != HIGH_TO_LOW_EXPECTED_MAKESPAN_TICKS:
        raise AssertionError(
            f"expected {HIGH_TO_LOW_EXPECTED_MAKESPAN_TICKS} ticks, got {makespan}"
        )
    return {
        "distribution": HIGH_TO_LOW_COUNTS,
        "active_experts": len(all_slots),
        "routed_tokens": sum(HIGH_TO_LOW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": makespan,
        "dma_serial_eids": HIGH_TO_LOW_DMA_SERIAL_EIDS,
    }


def audit_low_to_high_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(HIGH_TO_LOW_COUNTS) != EXPERT_COUNT or sum(HIGH_TO_LOW_COUNTS) != 140:
        raise AssertionError("low-to-high distribution contract changed")
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (22, 22):
        raise AssertionError("ascending replay must contain 22 local slots per cluster")

    all_slots = queues["c0"] + queues["c1"]
    if len(all_slots) != 44:
        raise AssertionError("ascending replay must contain 44 tasks")
    if {slot.expert_id for slot in all_slots} != set(range(43)):
        raise AssertionError("ascending replay must cover E0 through E42")
    expected_c0 = tuple(range(41, 2, -2)) + (1, 0)
    expected_c1 = tuple(range(42, 3, -2)) + (2, 0)
    if tuple(slot.expert_id for slot in queues["c0"]) != expected_c0:
        raise AssertionError("C0 ascending assignment diverged from the handoff")
    if tuple(slot.expert_id for slot in queues["c1"]) != expected_c1:
        raise AssertionError("C1 ascending assignment diverged from the handoff")

    routed_by_eid = [0] * EXPERT_COUNT
    for slot in all_slots:
        routed_by_eid[slot.expert_id] += slot.ntokens
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{slot.expert_id} slice@{slot.token_start_rank} ends at "
                f"{timeline.task_end}, expected {slot.reference_end_tick}"
            )
        if slot.s4_prefetch_dma != DMA_NONE or slot.skip_s1:
            raise AssertionError("ascending showcase does not use S4PF or S1 cache hits")
    if tuple(routed_by_eid) != HIGH_TO_LOW_COUNTS:
        raise AssertionError("ascending replay token coverage diverged")

    split = sorted(
        (slot.token_start_rank, slot.ntokens, slot.cluster_name)
        for slot in all_slots
        if slot.expert_id == 0
    )
    if split != [(0, 11, "c0"), (11, 11, "c1")]:
        raise AssertionError("E0 must be split into disjoint 11-token slices")

    expected_s2pf = {
        ("c0", 1, 0): DMA_BOTH,
        ("c0", 0, 0): DMA_IDMA,
        ("c1", 0, 11): DMA_XDMA,
    }
    actual_s2pf = {
        (slot.cluster_name, slot.expert_id, slot.token_start_rank): (
            slot.s2_prefetch_dma
        )
        for slot in all_slots
        if slot.s2_prefetch_dma != DMA_NONE
    }
    if actual_s2pf != expected_s2pf:
        raise AssertionError("ascending S2 prefetch choices diverged")

    e2 = next(slot for slot in queues["c1"] if slot.expert_id == 2)
    if (e2.s1_shape, e2.s3_shape, e2.s1_dma, e2.s3_dma) != (
        SHAPE_C,
        SHAPE_C,
        DMA_XDMA,
        DMA_IDMA,
    ):
        raise AssertionError("ascending E2 must preserve its explicit C/C DMA lanes")

    for cluster_name, slots in queues.items():
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick < previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} local tasks overlap")

    intervals = [
        (start, end, dma, slot.expert_id, stage, slot.cluster_name)
        for slot in all_slots
        for start, end, dma, stage in _dma_intervals(slot)
    ]
    for index, left in enumerate(intervals):
        for right in intervals[index + 1:]:
            if left[2] & right[2] and max(left[0], right[0]) < min(left[1], right[1]):
                raise AssertionError(
                    f"DMA overlap {left[5]}/E{left[3]}:{left[4]} {left[:2]} "
                    f"and {right[5]}/E{right[3]}:{right[4]} {right[:2]}"
                )

    queue_ticks = {
        name: max(slot.reference_end_tick for slot in slots)
        for name, slots in queues.items()
    }
    makespan = max(queue_ticks.values())
    if queue_ticks != {"c0": 165, "c1": 165}:
        raise AssertionError(f"ascending queue endpoints changed: {queue_ticks}")
    if makespan != LOW_TO_HIGH_EXPECTED_MAKESPAN_TICKS:
        raise AssertionError("ascending fixed-order makespan changed")
    if LOW_TO_HIGH_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS != 726:
        raise AssertionError("low-to-high structural lower bound must be 181.5 ticks")
    return {
        "distribution": HIGH_TO_LOW_COUNTS,
        "active_experts": 43,
        "task_count": len(all_slots),
        "routed_tokens": sum(HIGH_TO_LOW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": makespan,
        "critical_local_slots": 22,
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_lower_bound_quarter_ticks": (
            LOW_TO_HIGH_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
        ),
        "dma_serial_eids": (),
    }


def audit_ends_inward_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(HIGH_TO_LOW_COUNTS) != EXPERT_COUNT or sum(HIGH_TO_LOW_COUNTS) != 140:
        raise AssertionError("ends-inward distribution contract changed")
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (21, 22):
        raise AssertionError("ends-inward replay must contain C0=21 and C1=22 slots")

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 43 or set(slots_by_eid) != set(range(43)):
        raise AssertionError("ends-inward replay must cover E0 through E42 once")
    expected_c0 = (0, 41, 2) + tuple(range(4, 22))
    expected_c1 = (42, 1, 40, 3) + tuple(range(39, 21, -1))
    if tuple(slot.expert_id for slot in queues["c0"]) != expected_c0:
        raise AssertionError("C0 ends-inward assignment diverged from the handoff")
    if tuple(slot.expert_id for slot in queues["c1"]) != expected_c1:
        raise AssertionError("C1 ends-inward assignment diverged from the handoff")

    for expert_id, ntokens in enumerate(HIGH_TO_LOW_COUNTS[:43]):
        slot = slots_by_eid[expert_id]
        if slot.ntokens != ntokens or slot.token_start_rank != 0:
            raise AssertionError(f"E{expert_id} ends-inward token coverage changed")
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{expert_id} timeline ends at {timeline.task_end}, "
                f"expected {slot.reference_end_tick}"
            )
        if slot.s4_prefetch_dma != DMA_NONE or slot.skip_s1:
            raise AssertionError("ends-inward does not use S4PF or S1 cache hits")

    expected_special = {
        0: (SHAPE_A, SHAPE_B, DMA_IDMA, DMA_NONE, DMA_BOTH),
        41: (SHAPE_C, SHAPE_B, DMA_BOTH, DMA_NONE, DMA_BOTH),
        2: (SHAPE_A, SHAPE_B, DMA_IDMA, DMA_NONE, DMA_BOTH),
        42: (SHAPE_A, SHAPE_B, DMA_XDMA, DMA_BOTH, DMA_NONE),
        1: (SHAPE_A, SHAPE_B, DMA_BOTH, DMA_NONE, DMA_BOTH),
        40: (SHAPE_A, SHAPE_B, DMA_XDMA, DMA_BOTH, DMA_NONE),
        3: (SHAPE_A, SHAPE_C, DMA_BOTH, DMA_IDMA, DMA_NONE),
    }
    actual_special = {
        expert_id: (
            slots_by_eid[expert_id].s1_shape,
            slots_by_eid[expert_id].s3_shape,
            slots_by_eid[expert_id].s1_dma,
            slots_by_eid[expert_id].s3_dma,
            slots_by_eid[expert_id].s2_prefetch_dma,
        )
        for expert_id in expected_special
    }
    if actual_special != expected_special:
        raise AssertionError("ends-inward special shape/DMA profiles changed")

    for cluster_name, slots in queues.items():
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick < previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} local tasks overlap")

    expected_release_edges = {
        (("c1", 0, 42, "S1"), ("c0", 0, 0, "S2PF")),
        (("c0", 0, 0, "S2PF"), ("c1", 0, 42, "S3")),
        (("c1", 1, 1, "S2PF"), ("c0", 1, 41, "S1")),
        (("c0", 1, 41, "S2PF"), ("c1", 2, 40, "S1")),
        (("c1", 2, 40, "S1"), ("c0", 2, 2, "S2PF")),
        (("c0", 2, 2, "S2PF"), ("c1", 2, 40, "S3")),
        (("c1", 3, 3, "S3"), ("c0", 3, 4, "S1")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError("ends-inward cross-cluster DMA release edges changed")

    queue_ticks = {
        name: max(slot.reference_end_tick for slot in slots)
        for name, slots in queues.items()
    }
    if queue_ticks != {"c0": 166, "c1": 166}:
        raise AssertionError(f"ends-inward queue endpoints changed: {queue_ticks}")
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if structural_quarters != ENDS_INWARD_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS:
        raise AssertionError("ends-inward structural lower bound must be 182.5 ticks")
    return {
        "distribution": HIGH_TO_LOW_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(HIGH_TO_LOW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def format_schedule_manifest(
    queues: dict[str, tuple[SlotSpec, ...]], profile: str = HIGH_TO_LOW_PROFILE
) -> str:
    if profile == DYNAMIC_DESC_PROFILE:
        audit = audit_dynamic_desc_schedule(queues)
        title = "DYNAMIC_DESC high-to-low schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  dynamic_physical=shape,DMA,S2PF,S4PF; fixed_order=descending\n"
            f"  dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == STATIC_DESC_PROFILE:
        audit = audit_static_desc_schedule(queues)
        title = "STATIC_DESC fixed high-to-low schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  fixed_physical=S1:B,S3:B,C2:IDMA,C3:XDMA,S2PF:off,S4PF:off\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == LOW_TO_HIGH_PROFILE:
        audit = audit_low_to_high_schedule(queues)
        title = "low_to_high ascending schedule:"
        detail = (
            "  structural_lower_bound="
            f"{audit['makespan_ticks']} + {audit['critical_local_slots']}*"
            f"{audit['api_quarter_ticks_per_slot']}/4 = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.1f} ticks"
        )
    elif profile == ENDS_INWARD_PROFILE:
        audit = audit_ends_inward_schedule(queues)
        title = "ends_inward schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  structural_lower_bound=max("
            f"C0={cluster_bounds['c0'] / 4:.2f}, "
            f"C1={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.1f} ticks"
        )
    else:
        audit = audit_high_to_low_schedule(queues)
        title = "high_to_low descending schedule:"
        detail = "  explicit_dma_stall=E22->E23@tick103"
    lines = [
        title,
        f"  active_experts={audit['active_experts']} routed_tokens={audit['routed_tokens']}",
        f"  queue_ticks={audit['queue_ticks']} makespan={audit['makespan_ticks']}",
        detail,
    ]
    for cluster_name in ("c0", "c1"):
        encoded = ", ".join(
            f"E{slot.expert_id}[{slot.token_start_rank}:"
            f"{slot.token_start_rank + slot.ntokens}]:{slot.profile}"
            f"@[{slot.reference_start_tick},{slot.reference_end_tick})"
            for slot in queues[cluster_name]
        )
        lines.append(f"  {cluster_name.upper()} [{encoded}]")
    return "\n".join(lines)
