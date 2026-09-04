"""Static schedule profiles for the focused multi-cluster MoE workload."""

from dataclasses import dataclass, replace
import pathlib
import sys


PRODUCTION_WORKLOAD_DIR = pathlib.Path(__file__).resolve().parent.parent / "multi_cluster_MoE"
if str(PRODUCTION_WORKLOAD_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_WORKLOAD_DIR))
from moe_routing_cases import EXPERT_ORDER, get_routing_case  # noqa: E402


BASELINE_PROFILE = "baseline"
S2PF_BOTH_PROFILE = "s2pf_both"
STATIC_DESC_PROFILE = "static_desc"
DYNAMIC_DESC_PROFILE = "dynamic_desc"
DYNAMIC_TWO_ENDED_PROFILE = "dynamic_two_ended"
FULL_SCHEDULER_PROFILE = "full_scheduler"
M70_THREE_HOT_STATIC_DESC_PROFILE = "m70_three_hot_static_desc"
M70_THREE_HOT_DYNAMIC_DESC_PROFILE = "m70_three_hot_dynamic_desc"
M70_THREE_HOT_DYNAMIC_TWO_ENDED_PROFILE = "m70_three_hot_dynamic_two_ended"
M70_THREE_HOT_FULL_SCHEDULER_PROFILE = "m70_three_hot_full_scheduler"
M92_PARAMETER_ORDER_STATIC_DESC_PROFILE = "m92_parameter_order_static_desc"
M92_PARAMETER_ORDER_DYNAMIC_DESC_PROFILE = "m92_parameter_order_dynamic_desc"
M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_PROFILE = (
    "m92_parameter_order_dynamic_two_ended"
)
M92_PARAMETER_ORDER_FULL_SCHEDULER_PROFILE = (
    "m92_parameter_order_full_scheduler"
)
M60_HIGH_SKEW_STATIC_DESC_PROFILE = "m60_high_skew_static_desc"
M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE = "m60_high_skew_dynamic_desc"
M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE = (
    "m60_high_skew_dynamic_two_ended"
)
M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE = "m60_high_skew_full_scheduler"
M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE = (
    "m70_three_hot_dynamic_desc_skip_elided"
)
HIGH_TO_LOW_PROFILE = "high_to_low"
LOW_TO_HIGH_PROFILE = "low_to_high"
ENDS_INWARD_PROFILE = "ends_inward"
C_TAIL_SMOKE_PROFILE = "c_tail_smoke"
S1_STAGE_SMOKE_PROFILE = "s1_stage_smoke"
M8_COMPARISON_PROFILE = "m8_4_2_2_compare"
M8_FIXED_A_PROFILE = "m8_4_2_2_fixed_a"
M8_FIXED_B_PROFILE = "m8_4_2_2_fixed_b"
M8_FIXED_C_PROFILE = "m8_4_2_2_fixed_c"
M8_DISTILLED_PROFILE = "m8_4_2_2_distilled"
M8_COMPARISON_RUN_PROFILES = (
    M8_FIXED_A_PROFILE,
    M8_FIXED_B_PROFILE,
    M8_FIXED_C_PROFILE,
    M8_DISTILLED_PROFILE,
)
M32_COMPARISON_PROFILE = "m32_moe_style_compare"
M32_FIXED_A_PROFILE = "m32_moe_style_fixed_a"
M32_FIXED_B_PROFILE = "m32_moe_style_fixed_b"
M32_FIXED_C_PROFILE = "m32_moe_style_fixed_c"
M32_DISTILLED_PROFILE = "m32_moe_style_distilled"
M32_COMPARISON_RUN_PROFILES = (
    M32_FIXED_A_PROFILE,
    M32_FIXED_B_PROFILE,
    M32_FIXED_C_PROFILE,
    M32_DISTILLED_PROFILE,
)
COMPARISON_PROFILES = (M8_COMPARISON_PROFILE, M32_COMPARISON_PROFILE)
COMPARISON_RUN_PROFILES = (
    *M8_COMPARISON_RUN_PROFILES,
    *M32_COMPARISON_RUN_PROFILES,
)
SCHEDULE_PROFILES = (
    BASELINE_PROFILE,
    S2PF_BOTH_PROFILE,
    STATIC_DESC_PROFILE,
    DYNAMIC_DESC_PROFILE,
    DYNAMIC_TWO_ENDED_PROFILE,
    FULL_SCHEDULER_PROFILE,
    M70_THREE_HOT_STATIC_DESC_PROFILE,
    M70_THREE_HOT_DYNAMIC_DESC_PROFILE,
    M70_THREE_HOT_DYNAMIC_TWO_ENDED_PROFILE,
    M70_THREE_HOT_FULL_SCHEDULER_PROFILE,
    M92_PARAMETER_ORDER_STATIC_DESC_PROFILE,
    M92_PARAMETER_ORDER_DYNAMIC_DESC_PROFILE,
    M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_PROFILE,
    M92_PARAMETER_ORDER_FULL_SCHEDULER_PROFILE,
    M60_HIGH_SKEW_STATIC_DESC_PROFILE,
    M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE,
    M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE,
    M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE,
    M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE,
    HIGH_TO_LOW_PROFILE,
    LOW_TO_HIGH_PROFILE,
    ENDS_INWARD_PROFILE,
    C_TAIL_SMOKE_PROFILE,
    S1_STAGE_SMOKE_PROFILE,
    M8_COMPARISON_PROFILE,
    *M32_COMPARISON_RUN_PROFILES,
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
S2PF_EARLY_CTRL_BIT = 20

EXPERT_COUNT = 64
STATIC_DESC_EXPECTED_MAKESPAN_TICKS = 162
DYNAMIC_DESC_EXPECTED_MAKESPAN_TICKS = 159
DYNAMIC_TWO_ENDED_EXPECTED_MAKESPAN_TICKS = 137
FULL_SCHEDULER_EXPECTED_MAKESPAN_TICKS = 129
M70_THREE_HOT_STATIC_DESC_EXPECTED_MAKESPAN_TICKS = 132
M70_THREE_HOT_DYNAMIC_DESC_EXPECTED_MAKESPAN_TICKS = 126
M70_THREE_HOT_DYNAMIC_TWO_ENDED_EXPECTED_MAKESPAN_TICKS = 127
M70_THREE_HOT_FULL_SCHEDULER_EXPECTED_MAKESPAN_TICKS = 105
M92_PARAMETER_ORDER_STATIC_DESC_EXPECTED_MAKESPAN_TICKS = 198
M92_PARAMETER_ORDER_DYNAMIC_DESC_EXPECTED_MAKESPAN_TICKS = 168
M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_EXPECTED_MAKESPAN_TICKS = 172
M92_PARAMETER_ORDER_FULL_SCHEDULER_EXPECTED_MAKESPAN_TICKS = 144
M60_HIGH_SKEW_STATIC_DESC_EXPECTED_MAKESPAN_TICKS = 138
M60_HIGH_SKEW_DYNAMIC_DESC_EXPECTED_MAKESPAN_TICKS = 133
M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_EXPECTED_MAKESPAN_TICKS = 111
M60_HIGH_SKEW_FULL_SCHEDULER_EXPECTED_MAKESPAN_TICKS = 99
HIGH_TO_LOW_EXPECTED_MAKESPAN_TICKS = 163
LOW_TO_HIGH_EXPECTED_MAKESPAN_TICKS = 165
ENDS_INWARD_EXPECTED_MAKESPAN_TICKS = 166
M8_FIXED_A_EXPECTED_MAKESPAN_TICKS = 24
M8_FIXED_B_EXPECTED_MAKESPAN_TICKS = 15
M8_FIXED_C_EXPECTED_MAKESPAN_TICKS = 21
M8_DISTILLED_EXPECTED_MAKESPAN_TICKS = 15
M32_FIXED_A_EXPECTED_MAKESPAN_TICKS = 180
M32_FIXED_B_EXPECTED_MAKESPAN_TICKS = 99
M32_FIXED_C_EXPECTED_MAKESPAN_TICKS = 108
M32_DISTILLED_EXPECTED_MAKESPAN_TICKS = 87
# Same mandatory DFG/API allowance used for the accepted high-to-low bound:
# three quarter-ticks per cluster-local slot.
STRUCTURAL_API_QUARTER_TICKS_PER_SLOT = 3
STATIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 711
DYNAMIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 702
DYNAMIC_TWO_ENDED_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 629
FULL_SCHEDULER_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 612
M70_THREE_HOT_STATIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 558
M70_THREE_HOT_DYNAMIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 546
M70_THREE_HOT_DYNAMIC_TWO_ENDED_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 517
M70_THREE_HOT_FULL_SCHEDULER_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 468
M92_PARAMETER_ORDER_STATIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 837
M92_PARAMETER_ORDER_DYNAMIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 756
M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 799
M92_PARAMETER_ORDER_FULL_SCHEDULER_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 660
M60_HIGH_SKEW_STATIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 600
M60_HIGH_SKEW_DYNAMIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 574
M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 457
M60_HIGH_SKEW_FULL_SCHEDULER_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS = 456
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


def s2pf_s1_overlap_steps(
    *, skip_s1: bool, s1_shape: int, s1_dma: int, s2_prefetch_dma: int
) -> int:
    if skip_s1 or s2_prefetch_dma == DMA_NONE:
        return 0
    s1_dma_ticks = 2 if s1_dma == DMA_BOTH else 4
    return 2 if S1_COMPUTE_TICKS[s1_shape] > s1_dma_ticks else 0


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

M70_THREE_HOT_COUNTS = (
    (28,) * 3
    + (6,) * 4
    + (2,) * 16
    + (0,) * 41
)

M92_PARAMETER_ORDER_COUNTS = (
    (76, 40)
    + (2,) * 32
    + (1,) * 4
    + (0,) * 26
)

M60_HIGH_SKEW_COUNTS = (
    (36, 22, 13, 6)
    + (2,) * 17
    + (1,) * 9
    + (0,) * 34
)

# M=8 deterministic Top-2 routing shared with multi_cluster_MoE/m8_4_2_2:
#   T0..T3 -> (E0, E63)
#   T4..T5 -> (E0, E1)
#   T6..T7 -> (E63, E1)
M8_4_2_2_COUNTS = (6, 4) + (0,) * 61 + (6,)
M8_4_2_2_TOP2 = (
    (0, 63), (0, 63), (0, 63), (0, 63),
    (0, 1), (0, 1),
    (63, 1), (63, 1),
)
M8_4_2_2_TOKEN_IDS_BY_EXPERT = tuple(
    tuple(
        token_id
        for token_id, pair in enumerate(M8_4_2_2_TOP2)
        if expert_id in pair
    )
    for expert_id in range(EXPERT_COUNT)
)

# Import the exact M=32 Top-2 case used by the full workload.  Keeping this as
# one source of truth prevents the static comparison from silently testing a
# different marginal distribution or token-to-expert mapping.
M32_ROUTING_CASE = get_routing_case(32, EXPERT_COUNT)
M32_SCALED_SKEW_COUNTS = M32_ROUTING_CASE.expected_counts
M32_SCALED_SKEW_TOP2 = M32_ROUTING_CASE.token_major_top2
M32_SCALED_SKEW_TOKEN_IDS_BY_EXPERT = tuple(
    tuple(
        token_id
        for token_id, pair in enumerate(M32_SCALED_SKEW_TOP2)
        if expert_id in pair
    )
    for expert_id in range(EXPERT_COUNT)
)
M32_ACTIVE_EIDS_HIGH_TO_LOW = tuple(
    eid for eid in EXPERT_ORDER if M32_SCALED_SKEW_COUNTS[eid] > 0
)
M32_FIXED_CLUSTER_EIDS = {
    "c0": M32_ACTIVE_EIDS_HIGH_TO_LOW[0::2],
    "c1": M32_ACTIVE_EIDS_HIGH_TO_LOW[1::2],
}

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

# Exact Top-2 routing exported for synthetic_three_hot_medium_cold_m70.
# Entries E23-E63 are inactive.
M70_THREE_HOT_TOKEN_IDS_BY_EXPERT = (
    (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25,
     27, 28, 30, 31, 33, 36, 40, 43, 47, 58),
    (0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26,
     27, 29, 30, 32, 33, 37, 40, 44, 47, 59),
    (1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26,
     28, 29, 31, 32, 34, 37, 41, 44, 48, 59),
    (34, 38, 41, 45, 48, 60),
    (35, 38, 42, 45, 49, 60),
    (35, 39, 42, 46, 49, 61),
    (36, 39, 43, 46, 50, 61),
    (50, 62), (51, 62), (51, 63), (52, 63),
    (52, 64), (53, 64), (53, 65), (54, 65),
    (54, 66), (55, 66), (55, 67), (56, 67),
    (56, 68), (57, 68), (57, 69), (58, 69),
    *(() for _ in range(41)),
)

# Exact Top-2 routing exported for synthetic_parameter_and_order_stress_m92.
# Entries E38-E63 are inactive.
M92_PARAMETER_ORDER_TOKEN_IDS_BY_EXPERT = (
    tuple(range(76)),
    (*range(39), 71),
    (39, 72), (40, 73), (41, 74), (42, 75), (43, 76),
    (44, 76), (45, 77), (46, 77), (47, 78), (48, 78),
    (49, 79), (50, 79), (51, 80), (52, 80), (53, 81),
    (54, 81), (55, 82), (56, 82), (57, 83), (58, 83),
    (59, 84), (60, 84), (61, 85), (62, 85), (63, 86),
    (64, 86), (65, 87), (66, 87), (67, 88), (68, 88),
    (69, 89), (70, 89), (90,), (90,), (91,), (91,),
    *(() for _ in range(26)),
)

# Exact Top-2 routing exported for synthetic_high_skew_olmoe_style_m60.
# Entries E30-E63 are inactive.
M60_HIGH_SKEW_TOKEN_IDS_BY_EXPERT = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
     34, 45),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21, 23, 26,
     29, 32, 35, 45),
    (10, 12, 14, 16, 18, 20, 22, 24, 27, 30, 33, 35, 46),
    (25, 28, 31, 34, 36, 46),
    (36, 47), (37, 47), (37, 48), (38, 48), (38, 49),
    (39, 49), (39, 50), (40, 50), (40, 51), (41, 51),
    (41, 52), (42, 52), (42, 53), (43, 53), (43, 54),
    (44, 54), (44, 55),
    (55,), (56,), (56,), (57,), (57,), (58,), (58,), (59,), (59,),
    *(() for _ in range(34)),
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

M70_THREE_HOT_STATIC_DESC_CLUSTER_EIDS = {
    "c0": (0, 2, 8, 10, 12, 14, 16, 18, 20, 22),
    "c1": (1, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19, 21),
}

M92_PARAMETER_ORDER_STATIC_DESC_CLUSTER_EIDS = {
    "c0": (0, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37),
    "c1": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22,
           24, 26, 28, 30, 32, 34, 36),
}

M60_HIGH_SKEW_STATIC_DESC_CLUSTER_EIDS = {
    "c0": (0, 3, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
    "c1": (1, 2, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29),
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
    skip_s3: bool = False


def _case0_fixed_order_task(
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


def _m70_fixed_order_task(
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
        M70_THREE_HOT_COUNTS[expert_id],
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


def _m92_fixed_order_task(
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
        M92_PARAMETER_ORDER_COUNTS[expert_id],
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


def _m60_fixed_order_task(
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
        M60_HIGH_SKEW_COUNTS[expert_id],
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
    _case0_fixed_order_task("c0", 0, 0, 33, SHAPE_A, SHAPE_B, DMA_IDMA, DMA_IDMA, DMA_NONE),
    _case0_fixed_order_task("c0", 3, 33, 39, SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE, DMA_BOTH, 4),
    _case0_fixed_order_task("c0", 4, 39, 45, SHAPE_A, SHAPE_B, DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True),
    *(
        _case0_fixed_order_task(
            "c0", expert_id, 45 + slot * 6, 51 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        )
        for slot, expert_id in enumerate(range(5, 22, 2))
    ),
    *(
        _case0_fixed_order_task(
            "c0", expert_id, 99 + slot * 6, 105 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA,
        )
        for slot, expert_id in enumerate(range(23, 43, 2))
    ),
    _case0_fixed_order_task("c1", 1, 0, 27, SHAPE_A, SHAPE_B, DMA_XDMA, DMA_XDMA, DMA_NONE, DMA_XDMA, 2),
    _case0_fixed_order_task("c1", 2, 27, 48, SHAPE_A, SHAPE_B, DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 48 + slot * 6, 54 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        )
        for slot, expert_id in enumerate(range(6, 22, 2))
    ),
    _case0_fixed_order_task("c1", 22, 96, 99, SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 99 + slot * 6, 105 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA,
        )
        for slot, expert_id in enumerate(range(24, 43, 2))
    ),
)


# Literal cluster-local replay of case 0 / DYNAMIC_TWO_ENDED. C2 takes the
# hottest remaining expert and C3 the coldest; each local stream refills as
# soon as its own task and the required shared-DMA release edges permit.
DYNAMIC_TWO_ENDED_HISTORY = (
    _case0_fixed_order_task(
        "c0", 0, 0, 33, SHAPE_A, SHAPE_B,
        DMA_NONE, DMA_IDMA, DMA_BOTH,
    ),
    _case0_fixed_order_task(
        "c0", 1, 34, 61, SHAPE_A, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE,
    ),
    _case0_fixed_order_task(
        "c0", 2, 61, 82, SHAPE_A, SHAPE_B,
        DMA_NONE, DMA_IDMA, DMA_BOTH,
    ),
    *(
        _case0_fixed_order_task(
            "c0", expert_id, 83 + slot * 6, 89 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        )
        for slot, expert_id in enumerate(range(3, 12))
    ),
    _case0_fixed_order_task(
        "c1", 42, 0, 5, SHAPE_B, SHAPE_C,
        DMA_NONE, DMA_XDMA, DMA_BOTH,
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 5 + slot * 3, 8 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(41, 36, -1))
    ),
    _case0_fixed_order_task(
        "c1", 36, 20, 25, SHAPE_B, SHAPE_C,
        DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 25 + slot * 3, 28 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(35, 32, -1))
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 37 + slot * 3, 40 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(32, 29, -1))
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 46 + slot * 3, 49 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(29, 24, -1))
    ),
    _case0_fixed_order_task(
        "c1", 24, 61, 66, SHAPE_B, SHAPE_C,
        DMA_NONE, DMA_XDMA, DMA_BOTH,
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 66 + slot * 3, 69 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(23, 21, -1))
    ),
    _case0_fixed_order_task(
        "c1", 21, 72, 78, SHAPE_B, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE, DMA_BOTH, 20,
    ),
    _case0_fixed_order_task(
        "c1", 20, 78, 84, SHAPE_A, SHAPE_B,
        DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True,
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 86 + slot * 6, 92 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        )
        for slot, expert_id in enumerate(range(19, 11, -1))
    ),
)


# Literal cluster-local replay of case 0 / FULL_SCHEDULER. The scheduler first
# pairs each hot expert with useful C3 work, drains the C/C tail while C2 is
# busy, and then finishes the B/B experts in synchronized pairs.
FULL_SCHEDULER_HISTORY = (
    _case0_fixed_order_task(
        "c0", 0, 0, 33, SHAPE_A, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    _case0_fixed_order_task(
        "c0", 1, 33, 60, SHAPE_A, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    _case0_fixed_order_task(
        "c0", 2, 60, 81, SHAPE_B, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    *(
        _case0_fixed_order_task(
            "c0", expert_id, 81 + slot * 6, 87 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA,
        )
        for slot, expert_id in enumerate((5, 8, 10, 12, 14, 16, 18, 20))
    ),
    _case0_fixed_order_task(
        "c1", 4, 0, 6, SHAPE_B, SHAPE_B,
        DMA_NONE, DMA_XDMA, DMA_XDMA,
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 6 + slot * 3, 9 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(42, 33, -1))
    ),
    _case0_fixed_order_task(
        "c1", 6, 33, 39, SHAPE_B, SHAPE_B,
        DMA_NONE, DMA_XDMA, DMA_XDMA,
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 39 + slot * 3, 42 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(33, 26, -1))
    ),
    _case0_fixed_order_task(
        "c1", 3, 60, 66, SHAPE_B, SHAPE_B,
        DMA_NONE, DMA_XDMA, DMA_XDMA,
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 66 + slot * 3, 69 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(26, 21, -1))
    ),
    *(
        _case0_fixed_order_task(
            "c1", expert_id, 81 + slot * 6, 87 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA,
        )
        for slot, expert_id in enumerate((7, 9, 11, 13, 15, 17, 19, 21))
    ),
)


# Literal cluster-local replay of case 1 / DYNAMIC_DESC. The descending order
# is fixed while shape, DMA, S2PF, S4PF, and cache-hit fields remain dynamic.
M70_THREE_HOT_DYNAMIC_DESC_HISTORY = (
    _m70_fixed_order_task(
        "c0", 0, 0, 42, SHAPE_A, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE, DMA_IDMA, 2,
    ),
    _m70_fixed_order_task(
        "c0", 2, 42, 84, SHAPE_A, SHAPE_B,
        DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True,
    ),
    *(
        _m70_fixed_order_task(
            "c0", expert_id, 84 + slot * 6, 90 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA,
        )
        for slot, expert_id in enumerate(range(9, 22, 2))
    ),
    _m70_fixed_order_task(
        "c1", 1, 0, 42, SHAPE_A, SHAPE_B,
        DMA_XDMA, DMA_XDMA, DMA_NONE, DMA_XDMA, 3,
    ),
    *(
        _m70_fixed_order_task(
            "c1", expert_id, 42 + slot * 9, 51 + slot * 9,
            SHAPE_A, SHAPE_B, DMA_NONE, DMA_NONE, DMA_BOTH,
            DMA_BOTH, expert_id + 1, skip_s1=True,
        )
        for slot, expert_id in enumerate(range(3, 7))
    ),
    _m70_fixed_order_task(
        "c1", 7, 78, 81, SHAPE_A, SHAPE_C,
        DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True,
    ),
    _m70_fixed_order_task(
        "c1", 8, 81, 84, SHAPE_C, SHAPE_C,
        DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    *(
        _m70_fixed_order_task(
            "c1", expert_id, 84 + slot * 6, 90 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA,
        )
        for slot, expert_id in enumerate(range(10, 23, 2))
    ),
)


# Literal cluster-local replay of case 2 / DYNAMIC_DESC. Issue order remains
# descending while physical parameters and the E1-to-E2 cache handoff are
# selected dynamically by the exported policy.
M92_PARAMETER_ORDER_DYNAMIC_DESC_HISTORY = (
    _m92_fixed_order_task(
        "c0", 0, 0, 114, SHAPE_A, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    *(
        _m92_fixed_order_task(
            "c0", expert_id, 114 + slot * 6, 120 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA,
        )
        for slot, expert_id in enumerate(range(20, 38, 2))
    ),
    _m92_fixed_order_task(
        "c1", 1, 0, 60, SHAPE_A, SHAPE_B,
        DMA_XDMA, DMA_XDMA, DMA_NONE, DMA_XDMA, 2,
    ),
    _m92_fixed_order_task(
        "c1", 2, 60, 63, SHAPE_A, SHAPE_C,
        DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True,
    ),
    *(
        _m92_fixed_order_task(
            "c1", expert_id, 63 + slot * 3, 66 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(3, 20))
    ),
    *(
        _m92_fixed_order_task(
            "c1", expert_id, 114 + slot * 6, 120 + slot * 6,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA,
        )
        for slot, expert_id in enumerate(range(21, 38, 2))
    ),
)


# Literal cluster-local replay of case 2 / DYNAMIC_TWO_ENDED. C2 holds the
# hottest expert while C3 walks from the cold end back to E1.
M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_HISTORY = (
    _m92_fixed_order_task(
        "c0", 0, 0, 114, SHAPE_A, SHAPE_B,
        DMA_NONE, DMA_IDMA, DMA_BOTH,
    ),
    _m92_fixed_order_task(
        "c1", 37, 0, 5, SHAPE_B, SHAPE_C,
        DMA_NONE, DMA_XDMA, DMA_BOTH,
    ),
    *(
        _m92_fixed_order_task(
            "c1", expert_id, 5 + slot * 3, 8 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(36, 13, -1))
    ),
    _m92_fixed_order_task(
        "c1", 13, 74, 79, SHAPE_B, SHAPE_C,
        DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    *(
        _m92_fixed_order_task(
            "c1", expert_id, 79 + slot * 3, 82 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(12, 1, -1))
    ),
    _m92_fixed_order_task(
        "c1", 1, 112, 172, SHAPE_A, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE,
    ),
)


# Literal cluster-local replay of case 2 / FULL_SCHEDULER. C2 keeps the hot E0
# task resident while C3 drains the cold tail, then both clusters finish their
# remaining streams without S4PF or cache-hit tasks.
M92_PARAMETER_ORDER_FULL_SCHEDULER_HISTORY = (
    _m92_fixed_order_task(
        "c0", 0, 0, 114, SHAPE_A, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    *(
        _m92_fixed_order_task(
            "c0", expert_id, 114 + slot * 3, 117 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(10, 1, -1))
    ),
    _m92_fixed_order_task(
        "c1", 37, 0, 6, SHAPE_B, SHAPE_B,
        DMA_NONE, DMA_XDMA, DMA_XDMA,
    ),
    *(
        _m92_fixed_order_task(
            "c1", expert_id, 6 + slot * 3, 9 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(36, 10, -1))
    ),
    _m92_fixed_order_task(
        "c1", 1, 84, 144, SHAPE_B, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE,
    ),
)


# Literal cluster-local replay of case 3 / DYNAMIC_DESC. The descending order
# is fixed while shape, DMA, S2PF, S4PF, and cache-hit fields remain dynamic.
M60_HIGH_SKEW_DYNAMIC_DESC_HISTORY = (
    _m60_fixed_order_task(
        "c0", 0, 0, 54, SHAPE_A, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE, DMA_IDMA, 3,
    ),
    _m60_fixed_order_task(
        "c0", 3, 54, 63, SHAPE_A, SHAPE_B,
        DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True,
    ),
    *(
        _m60_fixed_order_task(
            "c0", expert_id, 64 + slot * 6, 67 + slot * 6,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(7, 30, 2))
    ),
    _m60_fixed_order_task(
        "c1", 1, 0, 33, SHAPE_A, SHAPE_B,
        DMA_XDMA, DMA_XDMA, DMA_NONE, DMA_XDMA, 2,
    ),
    _m60_fixed_order_task(
        "c1", 2, 33, 54, SHAPE_A, SHAPE_B,
        DMA_NONE, DMA_NONE, DMA_BOTH, DMA_XDMA, 4, skip_s1=True,
    ),
    _m60_fixed_order_task(
        "c1", 4, 54, 57, SHAPE_B, SHAPE_C,
        DMA_NONE, DMA_NONE, DMA_BOTH, skip_s1=True,
    ),
    _m60_fixed_order_task(
        "c1", 5, 57, 60, SHAPE_C, SHAPE_C,
        DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    *(
        _m60_fixed_order_task(
            "c1", expert_id, 61 + slot * 6, 64 + slot * 6,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(6, 29, 2))
    ),
)


# Literal cluster-local replay of case 3 / DYNAMIC_TWO_ENDED. C2 consumes the
# hot end while C3 walks from the cold end back to E3.
M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_HISTORY = (
    _m60_fixed_order_task(
        "c0", 0, 0, 54, SHAPE_A, SHAPE_B,
        DMA_NONE, DMA_IDMA, DMA_BOTH,
    ),
    _m60_fixed_order_task(
        "c0", 1, 55, 88, SHAPE_A, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE,
    ),
    _m60_fixed_order_task(
        "c0", 2, 90, 111, SHAPE_A, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE,
    ),
    _m60_fixed_order_task(
        "c1", 29, 0, 5, SHAPE_B, SHAPE_C,
        DMA_NONE, DMA_XDMA, DMA_BOTH,
    ),
    *(
        _m60_fixed_order_task(
            "c1", expert_id, 5 + slot * 3, 8 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(28, 18, -1))
    ),
    *(
        _m60_fixed_order_task(
            "c1", expert_id, 37 + slot * 3, 40 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(18, 12, -1))
    ),
    *(
        _m60_fixed_order_task(
            "c1", expert_id, 58 + slot * 3, 61 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(12, 3, -1))
    ),
    _m60_fixed_order_task(
        "c1", 3, 85, 94, SHAPE_B, SHAPE_B,
        DMA_BOTH, DMA_IDMA, DMA_NONE,
    ),
)


# Literal cluster-local replay of case 3 / FULL_SCHEDULER. C2 executes E0,
# drains E13 through E6, and finishes E2. C3 drains E29 through E14, executes
# E1, and finishes E5 through E3. The exported policy has three S2PF events
# and no S4PF/cache-hit tasks.
M60_HIGH_SKEW_FULL_SCHEDULER_HISTORY = (
    _m60_fixed_order_task(
        "c0", 0, 0, 54, SHAPE_A, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    *(
        _m60_fixed_order_task(
            "c0", expert_id, 54 + slot * 3, 57 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(13, 5, -1))
    ),
    _m60_fixed_order_task(
        "c0", 2, 78, 99, SHAPE_B, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE,
    ),
    _m60_fixed_order_task(
        "c1", 29, 0, 6, SHAPE_B, SHAPE_B,
        DMA_NONE, DMA_XDMA, DMA_XDMA,
    ),
    *(
        _m60_fixed_order_task(
            "c1", expert_id, 6 + slot * 3, 9 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(28, 13, -1))
    ),
    _m60_fixed_order_task(
        "c1", 1, 51, 84, SHAPE_B, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE,
    ),
    *(
        _m60_fixed_order_task(
            "c1", expert_id, 84 + slot * 3, 87 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate((5, 4))
    ),
    _m60_fixed_order_task(
        "c1", 3, 90, 99, SHAPE_C, SHAPE_C,
        DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
)


# Literal cluster-local replay of case 1 / DYNAMIC_TWO_ENDED. C2 consumes the
# three hot experts while C3 walks inward from the cold end. The exported
# policy has six early S2PF events and no S4PF/cache-hit tasks.
M70_THREE_HOT_DYNAMIC_TWO_ENDED_HISTORY = (
    _m70_fixed_order_task(
        "c0", 0, 0, 42, SHAPE_A, SHAPE_B,
        DMA_NONE, DMA_IDMA, DMA_BOTH,
    ),
    _m70_fixed_order_task(
        "c0", 1, 43, 85, SHAPE_A, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE,
    ),
    _m70_fixed_order_task(
        "c0", 2, 85, 127, SHAPE_A, SHAPE_B,
        DMA_BOTH, DMA_IDMA, DMA_NONE,
    ),
    _m70_fixed_order_task(
        "c1", 22, 0, 5, SHAPE_B, SHAPE_C,
        DMA_NONE, DMA_XDMA, DMA_BOTH,
    ),
    *(
        _m70_fixed_order_task(
            "c1", expert_id, 5 + slot * 3, 8 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(21, 14, -1))
    ),
    _m70_fixed_order_task(
        "c1", 14, 26, 31, SHAPE_B, SHAPE_C,
        DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    *(
        _m70_fixed_order_task(
            "c1", expert_id, 31 + slot * 3, 34 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(13, 9, -1))
    ),
    *(
        _m70_fixed_order_task(
            "c1", expert_id, 46 + slot * 3, 49 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(9, 6, -1))
    ),
    *(
        _m70_fixed_order_task(
            "c1", expert_id, 55 + slot * 9, 64 + slot * 9,
            SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        )
        for slot, expert_id in enumerate(range(6, 2, -1))
    ),
)


# Literal cluster-local replay of case 1 / FULL_SCHEDULER. The scheduler owns
# task order, cluster assignment, shape, DMA binding, and S2PF selection.
M70_THREE_HOT_FULL_SCHEDULER_HISTORY = (
    _m70_fixed_order_task(
        "c0", 2, 0, 42, SHAPE_B, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    _m70_fixed_order_task(
        "c0", 4, 42, 51, SHAPE_B, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    _m70_fixed_order_task(
        "c0", 1, 51, 93, SHAPE_B, SHAPE_B,
        DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    *(
        _m70_fixed_order_task(
            "c0", expert_id, 93 + slot * 3, 96 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(10, 6, -1))
    ),
    _m70_fixed_order_task(
        "c1", 3, 0, 9, SHAPE_B, SHAPE_B,
        DMA_XDMA, DMA_XDMA, DMA_NONE,
    ),
    *(
        _m70_fixed_order_task(
            "c1", expert_id, 9 + slot * 3, 12 + slot * 3,
            SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
        )
        for slot, expert_id in enumerate(range(22, 11, -1))
    ),
    _m70_fixed_order_task(
        "c1", 5, 42, 51, SHAPE_B, SHAPE_B,
        DMA_XDMA, DMA_XDMA, DMA_NONE,
    ),
    _m70_fixed_order_task(
        "c1", 6, 51, 60, SHAPE_B, SHAPE_B,
        DMA_XDMA, DMA_XDMA, DMA_NONE,
    ),
    _m70_fixed_order_task(
        "c1", 11, 60, 63, SHAPE_C, SHAPE_C,
        DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    _m70_fixed_order_task(
        "c1", 0, 63, 105, SHAPE_B, SHAPE_B,
        DMA_BOTH, DMA_BOTH, DMA_NONE,
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
    skip_s3_override: bool = False

    @property
    def single_dma(self) -> int:
        return DMA_IDMA if self.cluster_index == 0 else DMA_XDMA

    @property
    def skip_s3(self) -> bool:
        return self.skip_s3_override or self.s2_prefetch_dma != DMA_NONE

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
    def s2pf_s1_overlap_steps(self) -> int:
        return s2pf_s1_overlap_steps(
            skip_s1=self.skip_s1,
            s1_shape=self.s1_shape,
            s1_dma=self.s1_dma,
            s2_prefetch_dma=self.s2_prefetch_dma,
        )

    @property
    def s2pf_starts_after_s1_dma(self) -> bool:
        return (
            self.s2_prefetch_dma == DMA_NONE
            or self.skip_s1
            or self.s2pf_s1_overlap_steps != 0
        )

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


def build_m70_three_hot_static_desc_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    queues = {"c0": [], "c1": []}
    for cluster_name in ("c0", "c1"):
        cluster_index = 0 if cluster_name == "c0" else 1
        dma = DMA_IDMA if cluster_index == 0 else DMA_XDMA
        start_tick = 0
        for expert_id in M70_THREE_HOT_STATIC_DESC_CLUSTER_EIDS[cluster_name]:
            slot = SlotSpec(
                cluster_name=cluster_name,
                cluster_index=cluster_index,
                local_slot=len(queues[cluster_name]),
                expert_id=expert_id,
                ntokens=M70_THREE_HOT_COUNTS[expert_id],
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
            queues[cluster_name].append(replace(slot, reference_end_tick=end_tick))
            start_tick = end_tick
    frozen = {name: tuple(slots) for name, slots in queues.items()}
    audit_m70_three_hot_static_desc_schedule(frozen)
    return frozen


def build_m92_parameter_order_static_desc_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    queues = {"c0": [], "c1": []}
    for cluster_name in ("c0", "c1"):
        cluster_index = 0 if cluster_name == "c0" else 1
        dma = DMA_IDMA if cluster_index == 0 else DMA_XDMA
        start_tick = 0
        for expert_id in M92_PARAMETER_ORDER_STATIC_DESC_CLUSTER_EIDS[cluster_name]:
            slot = SlotSpec(
                cluster_name=cluster_name,
                cluster_index=cluster_index,
                local_slot=len(queues[cluster_name]),
                expert_id=expert_id,
                ntokens=M92_PARAMETER_ORDER_COUNTS[expert_id],
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
            queues[cluster_name].append(replace(slot, reference_end_tick=end_tick))
            start_tick = end_tick
    frozen = {name: tuple(slots) for name, slots in queues.items()}
    audit_m92_parameter_order_static_desc_schedule(frozen)
    return frozen


def build_m60_high_skew_static_desc_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    queues = {"c0": [], "c1": []}
    for cluster_name in ("c0", "c1"):
        cluster_index = 0 if cluster_name == "c0" else 1
        dma = DMA_IDMA if cluster_index == 0 else DMA_XDMA
        start_tick = 0
        for expert_id in M60_HIGH_SKEW_STATIC_DESC_CLUSTER_EIDS[cluster_name]:
            slot = SlotSpec(
                cluster_name=cluster_name,
                cluster_index=cluster_index,
                local_slot=len(queues[cluster_name]),
                expert_id=expert_id,
                ntokens=M60_HIGH_SKEW_COUNTS[expert_id],
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
            queues[cluster_name].append(replace(slot, reference_end_tick=end_tick))
            start_tick = end_tick
    frozen = {name: tuple(slots) for name, slots in queues.items()}
    audit_m60_high_skew_static_desc_schedule(frozen)
    return frozen


def _build_fixed_order_schedule(
    history: tuple[FixedOrderTask, ...],
) -> dict[str, tuple[SlotSpec, ...]]:
    queues = {"c0": [], "c1": []}
    for task in history:
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
                skip_s3_override=task.skip_s3,
            )
        )
    return {name: tuple(slots) for name, slots in queues.items()}


def build_dynamic_desc_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(DYNAMIC_DESC_HISTORY)
    audit_dynamic_desc_schedule(frozen)
    return frozen


def build_dynamic_two_ended_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(DYNAMIC_TWO_ENDED_HISTORY)
    audit_dynamic_two_ended_schedule(frozen)
    return frozen


def build_full_scheduler_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(FULL_SCHEDULER_HISTORY)
    audit_full_scheduler_schedule(frozen)
    return frozen


def build_m70_three_hot_dynamic_desc_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(M70_THREE_HOT_DYNAMIC_DESC_HISTORY)
    audit_m70_three_hot_dynamic_desc_schedule(frozen)
    return frozen


def build_m92_parameter_order_dynamic_desc_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(M92_PARAMETER_ORDER_DYNAMIC_DESC_HISTORY)
    audit_m92_parameter_order_dynamic_desc_schedule(frozen)
    return frozen


def build_m60_high_skew_dynamic_desc_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(M60_HIGH_SKEW_DYNAMIC_DESC_HISTORY)
    audit_m60_high_skew_dynamic_desc_schedule(frozen)
    return frozen


def build_m60_high_skew_dynamic_two_ended_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(
        M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_HISTORY
    )
    audit_m60_high_skew_dynamic_two_ended_schedule(frozen)
    return frozen


def build_m60_high_skew_full_scheduler_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(M60_HIGH_SKEW_FULL_SCHEDULER_HISTORY)
    audit_m60_high_skew_full_scheduler_schedule(frozen)
    return frozen


def build_m92_parameter_order_dynamic_two_ended_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(
        M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_HISTORY
    )
    audit_m92_parameter_order_dynamic_two_ended_schedule(frozen)
    return frozen


def build_m92_parameter_order_full_scheduler_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(
        M92_PARAMETER_ORDER_FULL_SCHEDULER_HISTORY
    )
    audit_m92_parameter_order_full_scheduler_schedule(frozen)
    return frozen


def build_m70_three_hot_dynamic_two_ended_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(M70_THREE_HOT_DYNAMIC_TWO_ENDED_HISTORY)
    audit_m70_three_hot_dynamic_two_ended_schedule(frozen)
    return frozen


def build_m70_three_hot_full_scheduler_schedule(
) -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(M70_THREE_HOT_FULL_SCHEDULER_HISTORY)
    audit_m70_three_hot_full_scheduler_schedule(frozen)
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


def build_m8_fixed_shape_schedule(
    shape: int,
) -> dict[str, tuple[SlotSpec, ...]]:
    """Descending E0/E63/E1 issue order with one literal S1/S3 shape.

    Every cluster starts with an empty expert-weight cache. All fixed-order
    tasks therefore load S1 and S3 weights through the cluster-local DMA lane;
    prefetch is disabled so the only changed physical parameter is shape.
    """
    if shape not in (SHAPE_A, SHAPE_B, SHAPE_C):
        raise ValueError(f"invalid M8 fixed shape {shape}")
    profile = f"{SHAPE_NAMES[shape]}/{SHAPE_NAMES[shape]} fixed-high-to-low"
    queues = {"c0": [], "c1": []}

    for cluster_name, cluster_index, expert_id in (
        ("c0", 0, 0),
        ("c1", 1, 63),
    ):
        first = SlotSpec(
            cluster_name=cluster_name,
            cluster_index=cluster_index,
            local_slot=0,
            expert_id=expert_id,
            ntokens=M8_4_2_2_COUNTS[expert_id],
            profile=profile,
            s1_shape=shape,
            s3_shape=shape,
            s2_prefetch_dma=DMA_NONE,
            reference_start_tick=0,
            reference_end_tick=-1,
            s1_dma_override=DMA_IDMA if cluster_index == 0 else DMA_XDMA,
            s3_dma_override=DMA_IDMA if cluster_index == 0 else DMA_XDMA,
        )
        queues[cluster_name].append(
            replace(first, reference_end_tick=_task_timeline(first, 0).task_end)
        )

    e1_start = queues["c0"][0].reference_end_tick
    e1 = SlotSpec(
        cluster_name="c0",
        cluster_index=0,
        local_slot=1,
        expert_id=1,
        ntokens=M8_4_2_2_COUNTS[1],
        profile=profile,
        s1_shape=shape,
        s3_shape=shape,
        s2_prefetch_dma=DMA_NONE,
        reference_start_tick=e1_start,
        reference_end_tick=-1,
        s1_dma_override=DMA_IDMA,
        s3_dma_override=DMA_IDMA,
    )
    e1 = replace(
        e1,
        reference_end_tick=_task_timeline(e1, e1_start).task_end,
    )
    queues["c0"].append(e1)
    frozen = {name: tuple(slots) for name, slots in queues.items()}
    audit_m8_comparison_schedule(frozen, {
        SHAPE_A: M8_FIXED_A_PROFILE,
        SHAPE_B: M8_FIXED_B_PROFILE,
        SHAPE_C: M8_FIXED_C_PROFILE,
    }[shape])
    return frozen


# Literal lowering of scheduler_rtl_distilled_policy.schedule(
#     {0: 6, 1: 4, 63: 6}, initial_cache_c2=-1, initial_cache_c3=-1
# ).  The deployed C mirror is checked against the same normative model.
M8_DISTILLED_HISTORY = (
    FixedOrderTask(
        "c0", 0, 0, 6, 0, 9,
        SHAPE_B, SHAPE_B, DMA_IDMA, DMA_IDMA, DMA_NONE,
    ),
    FixedOrderTask(
        "c0", 1, 0, 4, 9, 15,
        SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    FixedOrderTask(
        "c1", 63, 0, 6, 0, 9,
        SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA,
    ),
)


def build_m8_distilled_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(M8_DISTILLED_HISTORY)
    audit_m8_comparison_schedule(frozen, M8_DISTILLED_PROFILE)
    return frozen


def build_m8_comparison_schedules(
) -> dict[str, dict[str, tuple[SlotSpec, ...]]]:
    """Return the four sequential runs emitted by the M8 comparison profile."""
    return {
        M8_FIXED_A_PROFILE: build_m8_fixed_shape_schedule(SHAPE_A),
        M8_FIXED_B_PROFILE: build_m8_fixed_shape_schedule(SHAPE_B),
        M8_FIXED_C_PROFILE: build_m8_fixed_shape_schedule(SHAPE_C),
        M8_DISTILLED_PROFILE: build_m8_distilled_schedule(),
    }


def build_m32_fixed_shape_schedule(
    shape: int,
) -> dict[str, tuple[SlotSpec, ...]]:
    """Issue all active M=32 experts high-to-low with one fixed shape.

    The globally descending stream is distributed round-robin onto C2/C3.
    Both clusters start empty, every task loads S1 and S3 through its local
    DMA lane, and no prefetch is enabled.  Consequently the three runs differ
    only in the selected A/A, B/B or C/C physical shape.
    """
    if shape not in (SHAPE_A, SHAPE_B, SHAPE_C):
        raise ValueError(f"invalid M32 fixed shape {shape}")
    profile = f"{SHAPE_NAMES[shape]}/{SHAPE_NAMES[shape]} fixed-high-to-low"
    queues = {"c0": [], "c1": []}
    for cluster_name in ("c0", "c1"):
        cluster_index = 0 if cluster_name == "c0" else 1
        dma = DMA_IDMA if cluster_index == 0 else DMA_XDMA
        start_tick = 0
        for expert_id in M32_FIXED_CLUSTER_EIDS[cluster_name]:
            slot = SlotSpec(
                cluster_name=cluster_name,
                cluster_index=cluster_index,
                local_slot=len(queues[cluster_name]),
                expert_id=expert_id,
                ntokens=M32_SCALED_SKEW_COUNTS[expert_id],
                profile=profile,
                s1_shape=shape,
                s3_shape=shape,
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
    audit_m32_comparison_schedule(frozen, {
        SHAPE_A: M32_FIXED_A_PROFILE,
        SHAPE_B: M32_FIXED_B_PROFILE,
        SHAPE_C: M32_FIXED_C_PROFILE,
    }[shape])
    return frozen


# Frozen lowering of scheduler_rtl_distilled_policy.schedule() for the exact
# production M=32 routing distribution and empty C2/C3 caches.  Keep every
# irregular leading action explicit; the regular B/B tail is compacted below.
M32_DISTILLED_HISTORY = (
    FixedOrderTask(
        "c0", 0, 0, 8, 0, 12,
        SHAPE_A, SHAPE_B, DMA_IDMA, DMA_IDMA, DMA_NONE,
        skip_s3=True,
    ),
    FixedOrderTask(
        "c1", 3, 0, 3, 0, 6,
        SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA,
    ),
    FixedOrderTask(
        "c1", 27, 0, 1, 6, 9,
        SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    FixedOrderTask(
        "c1", 63, 0, 6, 9, 18,
        SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        skip_s3=True,
    ),
    FixedOrderTask(
        "c0", 26, 0, 1, 12, 15,
        SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    FixedOrderTask(
        "c0", 1, 0, 5, 15, 24,
        SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        skip_s3=True,
    ),
    FixedOrderTask(
        "c1", 25, 0, 1, 18, 21,
        SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
    FixedOrderTask(
        "c1", 2, 0, 4, 21, 27,
        SHAPE_B, SHAPE_B, DMA_BOTH, DMA_BOTH, DMA_NONE,
        skip_s3=True,
    ),
    FixedOrderTask(
        "c0", 4, 0, 2, 24, 27,
        SHAPE_C, SHAPE_C, DMA_NONE, DMA_BOTH, DMA_BOTH,
    ),
) + tuple(
    task
    for pair_index, (c0_eid, c1_eid) in enumerate(
        zip(range(5, 24, 2), range(6, 25, 2))
    )
    for task in (
        FixedOrderTask(
            "c0", c0_eid, 0, M32_SCALED_SKEW_COUNTS[c0_eid],
            27 + 6 * pair_index, 33 + 6 * pair_index,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_IDMA, DMA_IDMA,
        ),
        FixedOrderTask(
            "c1", c1_eid, 0, M32_SCALED_SKEW_COUNTS[c1_eid],
            27 + 6 * pair_index, 33 + 6 * pair_index,
            SHAPE_B, SHAPE_B, DMA_NONE, DMA_XDMA, DMA_XDMA,
        ),
    )
)


def build_m32_distilled_schedule() -> dict[str, tuple[SlotSpec, ...]]:
    frozen = _build_fixed_order_schedule(M32_DISTILLED_HISTORY)
    audit_m32_comparison_schedule(frozen, M32_DISTILLED_PROFILE)
    return frozen


def build_m32_comparison_schedules(
) -> dict[str, dict[str, tuple[SlotSpec, ...]]]:
    """Return the four sequential runs emitted by the M=32 profile."""
    return {
        M32_FIXED_A_PROFILE: build_m32_fixed_shape_schedule(SHAPE_A),
        M32_FIXED_B_PROFILE: build_m32_fixed_shape_schedule(SHAPE_B),
        M32_FIXED_C_PROFILE: build_m32_fixed_shape_schedule(SHAPE_C),
        M32_DISTILLED_PROFILE: build_m32_distilled_schedule(),
    }


def build_schedule_profile(profile: str) -> dict[str, tuple[SlotSpec, ...]]:
    if profile == M8_COMPARISON_PROFILE:
        # One representative queue is sufficient for generic layout sizing;
        # main_bingo/datagen explicitly consume all four runs.
        return build_m8_distilled_schedule()
    if profile == M32_COMPARISON_PROFILE:
        return build_m32_distilled_schedule()
    if profile == M32_FIXED_A_PROFILE:
        return build_m32_fixed_shape_schedule(SHAPE_A)
    if profile == M32_FIXED_B_PROFILE:
        return build_m32_fixed_shape_schedule(SHAPE_B)
    if profile == M32_FIXED_C_PROFILE:
        return build_m32_fixed_shape_schedule(SHAPE_C)
    if profile == M32_DISTILLED_PROFILE:
        return build_m32_distilled_schedule()
    if profile in (
        M70_THREE_HOT_DYNAMIC_DESC_PROFILE,
        M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE,
    ):
        return build_m70_three_hot_dynamic_desc_schedule()
    if profile == M70_THREE_HOT_STATIC_DESC_PROFILE:
        return build_m70_three_hot_static_desc_schedule()
    if profile == M70_THREE_HOT_DYNAMIC_TWO_ENDED_PROFILE:
        return build_m70_three_hot_dynamic_two_ended_schedule()
    if profile == M70_THREE_HOT_FULL_SCHEDULER_PROFILE:
        return build_m70_three_hot_full_scheduler_schedule()
    if profile == M92_PARAMETER_ORDER_STATIC_DESC_PROFILE:
        return build_m92_parameter_order_static_desc_schedule()
    if profile == M92_PARAMETER_ORDER_DYNAMIC_DESC_PROFILE:
        return build_m92_parameter_order_dynamic_desc_schedule()
    if profile == M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_PROFILE:
        return build_m92_parameter_order_dynamic_two_ended_schedule()
    if profile == M92_PARAMETER_ORDER_FULL_SCHEDULER_PROFILE:
        return build_m92_parameter_order_full_scheduler_schedule()
    if profile == M60_HIGH_SKEW_STATIC_DESC_PROFILE:
        return build_m60_high_skew_static_desc_schedule()
    if profile == M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE:
        return build_m60_high_skew_dynamic_desc_schedule()
    if profile == M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE:
        return build_m60_high_skew_dynamic_two_ended_schedule()
    if profile == M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE:
        return build_m60_high_skew_full_scheduler_schedule()
    if profile == STATIC_DESC_PROFILE:
        return build_static_desc_schedule()
    if profile == DYNAMIC_DESC_PROFILE:
        return build_dynamic_desc_schedule()
    if profile == DYNAMIC_TWO_ENDED_PROFILE:
        return build_dynamic_two_ended_schedule()
    if profile == FULL_SCHEDULER_PROFILE:
        return build_full_scheduler_schedule()
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


def audit_m8_comparison_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
    profile: str,
) -> dict[str, object]:
    """Audit one of the four M=8 comparison streams.

    The distribution describes Top-2 routes, so eight source tokens must
    contribute exactly sixteen routed token instances.  The four streams must
    differ only in fixed shape or in the distilled scheduler decisions.
    """
    if len(M8_4_2_2_COUNTS) != EXPERT_COUNT:
        raise AssertionError("M8 must define all 64 conceptual experts")
    if sum(M8_4_2_2_COUNTS) != 16:
        raise AssertionError("M8 Top-2 routing must contain 16 routes")
    if tuple(
        len(token_ids) for token_ids in M8_4_2_2_TOKEN_IDS_BY_EXPERT
    ) != M8_4_2_2_COUNTS:
        raise AssertionError("M8 routing does not match expert loads")

    token_owners = [[] for _ in range(8)]
    for expert_id, token_ids in enumerate(M8_4_2_2_TOKEN_IDS_BY_EXPERT):
        for token_id in token_ids:
            if not 0 <= token_id < len(token_owners):
                raise AssertionError(f"E{expert_id} has invalid token {token_id}")
            token_owners[token_id].append(expert_id)
    if any(len(owners) != 2 for owners in token_owners):
        raise AssertionError("M8 must route every source token exactly twice")

    if tuple(len(queues[name]) for name in ("c0", "c1")) != (2, 1):
        raise AssertionError("M8 comparison must contain two C2 and one C3 task")
    all_slots = queues["c0"] + queues["c1"]
    if {slot.expert_id for slot in all_slots} != {0, 1, 63}:
        raise AssertionError("M8 comparison must cover E0, E1 and E63 once")
    for slot in all_slots:
        if slot.ntokens != M8_4_2_2_COUNTS[slot.expert_id]:
            raise AssertionError(f"E{slot.expert_id} token count mismatch")
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{slot.expert_id} timeline ends at {timeline.task_end}, "
                f"expected {slot.reference_end_tick}"
            )
        if slot.s4_prefetch_dma != DMA_NONE:
            raise AssertionError("M8 comparison does not use S4 prefetch")

    for cluster_name, slots in queues.items():
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick < previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} local tasks overlap")

    fixed_profiles = {
        M8_FIXED_A_PROFILE: (SHAPE_A, M8_FIXED_A_EXPECTED_MAKESPAN_TICKS),
        M8_FIXED_B_PROFILE: (SHAPE_B, M8_FIXED_B_EXPECTED_MAKESPAN_TICKS),
        M8_FIXED_C_PROFILE: (SHAPE_C, M8_FIXED_C_EXPECTED_MAKESPAN_TICKS),
    }
    if profile in fixed_profiles:
        expected_shape, expected_makespan = fixed_profiles[profile]
        if tuple(slot.expert_id for slot in queues["c0"]) != (0, 1):
            raise AssertionError("fixed high-to-low C2 stream must be E0 then E1")
        if tuple(slot.expert_id for slot in queues["c1"]) != (63,):
            raise AssertionError("fixed high-to-low C3 stream must be E63")
        if any(
            slot.s1_shape != expected_shape or slot.s3_shape != expected_shape
            for slot in all_slots
        ):
            raise AssertionError("fixed run contains a non-fixed shape")
        for slot in all_slots:
            expected_dma = DMA_IDMA if slot.cluster_name == "c0" else DMA_XDMA
            if (
                slot.skip_s1
                or slot.skip_s3
                or slot.s1_dma != expected_dma
                or slot.s3_dma != expected_dma
                or slot.s2_prefetch_dma != DMA_NONE
            ):
                raise AssertionError(
                    "empty-cache fixed tasks must use local DMA without prefetch"
                )
    elif profile == M8_DISTILLED_PROFILE:
        expected_makespan = M8_DISTILLED_EXPECTED_MAKESPAN_TICKS
        if tuple(slot.expert_id for slot in queues["c0"]) != (0, 1):
            raise AssertionError("distilled C2 stream must be E0 then E1")
        if tuple(slot.expert_id for slot in queues["c1"]) != (63,):
            raise AssertionError("distilled C3 stream must be E63")
        expected = {
            0: (SHAPE_B, SHAPE_B, DMA_IDMA, DMA_NONE, DMA_IDMA, False, True),
            1: (SHAPE_C, SHAPE_C, DMA_BOTH, DMA_BOTH, DMA_NONE, False, False),
            63: (SHAPE_B, SHAPE_B, DMA_XDMA, DMA_XDMA, DMA_NONE, False, False),
        }
        for slot in all_slots:
            actual = (
                slot.s1_shape,
                slot.s3_shape,
                slot.s1_dma,
                slot.s3_dma,
                slot.s2_prefetch_dma,
                slot.skip_s1,
                slot.skip_s3,
            )
            if actual != expected[slot.expert_id]:
                raise AssertionError(
                    f"distilled E{slot.expert_id} decision changed: {actual}"
                )
    else:
        raise ValueError(f"unsupported M8 run profile {profile!r}")

    queue_ticks = {
        name: max(slot.reference_end_tick for slot in queues[name])
        for name in ("c0", "c1")
    }
    makespan = max(queue_ticks.values())
    if makespan != expected_makespan:
        raise AssertionError(
            f"{profile} expected {expected_makespan} ticks, got {makespan}"
        )
    return {
        "distribution": M8_4_2_2_COUNTS,
        "active_experts": 3,
        "source_tokens": 8,
        "routed_tokens": 16,
        "queue_ticks": queue_ticks,
        "makespan_ticks": makespan,
        "dma_release_edges": cross_cluster_dma_release_edges(queues),
    }


def audit_m32_comparison_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
    profile: str,
) -> dict[str, object]:
    """Audit one M=32 fixed-shape or distilled comparison stream."""
    counts = M32_SCALED_SKEW_COUNTS
    token_ids_by_expert = M32_SCALED_SKEW_TOKEN_IDS_BY_EXPERT
    if len(counts) != EXPERT_COUNT or sum(counts) != 64:
        raise AssertionError("M32 must define 64 experts and 64 Top-2 routes")
    if tuple(len(ids) for ids in token_ids_by_expert) != counts:
        raise AssertionError("M32 routing does not match expert loads")
    token_owners = [[] for _ in range(32)]
    for expert_id, token_ids in enumerate(token_ids_by_expert):
        for token_id in token_ids:
            if not 0 <= token_id < len(token_owners):
                raise AssertionError(f"E{expert_id} has invalid token {token_id}")
            token_owners[token_id].append(expert_id)
    if any(len(owners) != 2 for owners in token_owners):
        raise AssertionError("M32 must route every source token exactly twice")

    all_slots = queues["c0"] + queues["c1"]
    active_eids = {eid for eid, count in enumerate(counts) if count}
    if len(all_slots) != len(active_eids):
        raise AssertionError("M32 comparison must execute every active expert once")
    if {slot.expert_id for slot in all_slots} != active_eids:
        raise AssertionError("M32 comparison active-expert set changed")
    if len({slot.expert_id for slot in all_slots}) != len(all_slots):
        raise AssertionError("M32 comparison executes an expert more than once")
    for slot in all_slots:
        if slot.ntokens != counts[slot.expert_id]:
            raise AssertionError(f"E{slot.expert_id} token count mismatch")
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{slot.expert_id} timeline ends at {timeline.task_end}, "
                f"expected {slot.reference_end_tick}"
            )
        if slot.s4_prefetch_dma != DMA_NONE:
            raise AssertionError("M32 comparison does not use S4 prefetch")
    for cluster_name, slots in queues.items():
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick < previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} local tasks overlap")

    fixed_profiles = {
        M32_FIXED_A_PROFILE: (SHAPE_A, M32_FIXED_A_EXPECTED_MAKESPAN_TICKS),
        M32_FIXED_B_PROFILE: (SHAPE_B, M32_FIXED_B_EXPECTED_MAKESPAN_TICKS),
        M32_FIXED_C_PROFILE: (SHAPE_C, M32_FIXED_C_EXPECTED_MAKESPAN_TICKS),
    }
    if profile in fixed_profiles:
        expected_shape, expected_makespan = fixed_profiles[profile]
        for cluster_name in ("c0", "c1"):
            if tuple(slot.expert_id for slot in queues[cluster_name]) != (
                M32_FIXED_CLUSTER_EIDS[cluster_name]
            ):
                raise AssertionError(
                    f"fixed high-to-low {cluster_name} stream changed"
                )
        for slot in all_slots:
            expected_dma = DMA_IDMA if slot.cluster_name == "c0" else DMA_XDMA
            if (
                slot.s1_shape != expected_shape
                or slot.s3_shape != expected_shape
                or slot.skip_s1
                or slot.skip_s3
                or slot.s1_dma != expected_dma
                or slot.s3_dma != expected_dma
                or slot.s2_prefetch_dma != DMA_NONE
            ):
                raise AssertionError(
                    "fixed M32 tasks must use one shape and empty-cache local DMA"
                )
    elif profile == M32_DISTILLED_PROFILE:
        expected_makespan = M32_DISTILLED_EXPECTED_MAKESPAN_TICKS
        expected = _build_fixed_order_schedule(M32_DISTILLED_HISTORY)
        for cluster_name in ("c0", "c1"):
            actual_signature = tuple(
                (
                    slot.expert_id,
                    slot.ntokens,
                    slot.reference_start_tick,
                    slot.reference_end_tick,
                    slot.s1_shape,
                    slot.s3_shape,
                    slot.s1_dma,
                    slot.s3_dma,
                    slot.s2_prefetch_dma,
                    slot.skip_s1,
                    slot.skip_s3,
                )
                for slot in queues[cluster_name]
            )
            expected_signature = tuple(
                (
                    slot.expert_id,
                    slot.ntokens,
                    slot.reference_start_tick,
                    slot.reference_end_tick,
                    slot.s1_shape,
                    slot.s3_shape,
                    slot.s1_dma,
                    slot.s3_dma,
                    slot.s2_prefetch_dma,
                    slot.skip_s1,
                    slot.skip_s3,
                )
                for slot in expected[cluster_name]
            )
            if actual_signature != expected_signature:
                raise AssertionError(
                    f"distilled M32 {cluster_name} action stream changed"
                )
        if any(slot.skip_s1 for slot in all_slots):
            raise AssertionError("empty initial cache cannot skip any M32 S1 load")
        prefetched_s3 = {
            slot.expert_id for slot in all_slots if slot.s2_prefetch_dma != DMA_NONE
        }
        if prefetched_s3 != {0, 1, 2, 63}:
            raise AssertionError(
                "distilled M32 S2-prefetch set must be E0/E1/E2/E63"
            )
    else:
        raise ValueError(f"unsupported M32 run profile {profile!r}")

    queue_ticks = {
        name: max(slot.reference_end_tick for slot in queues[name])
        for name in ("c0", "c1")
    }
    makespan = max(queue_ticks.values())
    if makespan != expected_makespan:
        raise AssertionError(
            f"{profile} expected {expected_makespan} ticks, got {makespan}"
        )
    return {
        "distribution": counts,
        "active_experts": len(active_eids),
        "source_tokens": 32,
        "routed_tokens": 64,
        "queue_ticks": queue_ticks,
        "makespan_ticks": makespan,
        "dma_release_edges": cross_cluster_dma_release_edges(queues),
    }


def _audit_m70_three_hot_distribution() -> None:
    if len(M70_THREE_HOT_COUNTS) != EXPERT_COUNT or sum(M70_THREE_HOT_COUNTS) != 140:
        raise AssertionError("M70 distribution contract changed")
    routed_counts = tuple(
        len(M70_THREE_HOT_TOKEN_IDS_BY_EXPERT[eid])
        for eid in range(EXPERT_COUNT)
    )
    if routed_counts != M70_THREE_HOT_COUNTS:
        raise AssertionError("M70 routing does not match expert loads")

    token_owners = [[] for _ in range(70)]
    for expert_id, token_ids in enumerate(M70_THREE_HOT_TOKEN_IDS_BY_EXPERT):
        for token_id in token_ids:
            if not 0 <= token_id < len(token_owners):
                raise AssertionError(f"E{expert_id} has invalid token {token_id}")
            token_owners[token_id].append(expert_id)
    if any(len(owners) != 2 for owners in token_owners):
        raise AssertionError("M70 must route every input token twice")


def _audit_m60_high_skew_distribution() -> None:
    if len(M60_HIGH_SKEW_COUNTS) != EXPERT_COUNT:
        raise AssertionError("M60 must define 64 conceptual experts")
    if sum(M60_HIGH_SKEW_COUNTS) != 120:
        raise AssertionError("M60 assignment total must be 120")
    if tuple(
        len(token_ids) for token_ids in M60_HIGH_SKEW_TOKEN_IDS_BY_EXPERT
    ) != M60_HIGH_SKEW_COUNTS:
        raise AssertionError("M60 routing does not match expert loads")

    token_owners = [[] for _ in range(60)]
    for expert_id, token_ids in enumerate(M60_HIGH_SKEW_TOKEN_IDS_BY_EXPERT):
        for token_id in token_ids:
            if not 0 <= token_id < len(token_owners):
                raise AssertionError(f"E{expert_id} has invalid token {token_id}")
            token_owners[token_id].append(expert_id)
    if any(len(owners) != 2 for owners in token_owners):
        raise AssertionError("M60 must route every input token twice")


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


def audit_m70_three_hot_static_desc_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    _audit_m70_three_hot_distribution()

    if tuple(len(queues[name]) for name in ("c0", "c1")) != (10, 13):
        raise AssertionError("M70 STATIC_DESC must contain C2=10 and C3=13 slots")
    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 23 or set(slots_by_eid) != set(range(23)):
        raise AssertionError("M70 STATIC_DESC must cover E0 through E22 once")

    for cluster_name, expected_eids in M70_THREE_HOT_STATIC_DESC_CLUSTER_EIDS.items():
        if tuple(slot.expert_id for slot in queues[cluster_name]) != expected_eids:
            raise AssertionError(f"{cluster_name} M70 STATIC_DESC stream changed")
        if queues[cluster_name][0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(queues[cluster_name], queues[cluster_name][1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} M70 STATIC_DESC stream has a gap")

    for expert_id, slot in slots_by_eid.items():
        expected_dma = DMA_IDMA if slot.cluster_name == "c0" else DMA_XDMA
        if slot.ntokens != M70_THREE_HOT_COUNTS[expert_id]:
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
        if _task_timeline(slot, slot.reference_start_tick).task_end != slot.reference_end_tick:
            raise AssertionError(f"E{expert_id} M70 STATIC_DESC timeline changed")

    if cross_cluster_dma_release_edges(queues):
        raise AssertionError("M70 STATIC_DESC dedicated DMA lanes need no release edges")
    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 132, "c1": 126}:
        raise AssertionError(f"M70 STATIC_DESC queue endpoints changed: {queue_ticks}")
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if structural_quarters != M70_THREE_HOT_STATIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS:
        raise AssertionError("M70 STATIC_DESC structural lower bound must be 139.5 ticks")
    return {
        "distribution": M70_THREE_HOT_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M70_THREE_HOT_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": (),
    }


def audit_m92_parameter_order_static_desc_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(M92_PARAMETER_ORDER_COUNTS) != EXPERT_COUNT:
        raise AssertionError("M92 STATIC_DESC must define 64 conceptual experts")
    if sum(M92_PARAMETER_ORDER_COUNTS) != 184:
        raise AssertionError("M92 STATIC_DESC assignment total must be 184")
    if tuple(
        len(token_ids) for token_ids in M92_PARAMETER_ORDER_TOKEN_IDS_BY_EXPERT
    ) != M92_PARAMETER_ORDER_COUNTS:
        raise AssertionError("M92 STATIC_DESC routing does not match expert loads")

    token_owners = [[] for _ in range(92)]
    for expert_id, token_ids in enumerate(M92_PARAMETER_ORDER_TOKEN_IDS_BY_EXPERT):
        for token_id in token_ids:
            if not 0 <= token_id < len(token_owners):
                raise AssertionError(f"E{expert_id} has invalid token {token_id}")
            token_owners[token_id].append(expert_id)
    if any(len(owners) != 2 for owners in token_owners):
        raise AssertionError("M92 STATIC_DESC must route every input token twice")

    if tuple(len(queues[name]) for name in ("c0", "c1")) != (15, 23):
        raise AssertionError(
            "M92 STATIC_DESC must contain C2=15 and C3=23 slots"
        )
    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 38 or set(slots_by_eid) != set(range(38)):
        raise AssertionError("M92 STATIC_DESC must cover active E0 through E37 once")

    for cluster_name, expected_eids in M92_PARAMETER_ORDER_STATIC_DESC_CLUSTER_EIDS.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(f"{cluster_name} M92 STATIC_DESC stream changed")
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} M92 STATIC_DESC stream has a gap")

    for expert_id, slot in slots_by_eid.items():
        expected_dma = DMA_IDMA if slot.cluster_name == "c0" else DMA_XDMA
        if slot.ntokens != M92_PARAMETER_ORDER_COUNTS[expert_id]:
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
        if _task_timeline(slot, slot.reference_start_tick).task_end != slot.reference_end_tick:
            raise AssertionError(f"E{expert_id} M92 STATIC_DESC timeline changed")

    if cross_cluster_dma_release_edges(queues):
        raise AssertionError("M92 STATIC_DESC dedicated DMA lanes need no release edges")
    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 198, "c1": 192}:
        raise AssertionError(
            f"M92 STATIC_DESC queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M92_PARAMETER_ORDER_STATIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError(
            "M92 STATIC_DESC structural lower bound must be 209.25 ticks"
        )
    return {
        "distribution": M92_PARAMETER_ORDER_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M92_PARAMETER_ORDER_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": (),
    }


def audit_m60_high_skew_static_desc_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    _audit_m60_high_skew_distribution()

    if tuple(len(queues[name]) for name in ("c0", "c1")) != (14, 16):
        raise AssertionError("M60 STATIC_DESC must contain C2=14 and C3=16 slots")
    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 30 or set(slots_by_eid) != set(range(30)):
        raise AssertionError("M60 STATIC_DESC must cover active E0 through E29 once")

    for cluster_name, expected_eids in M60_HIGH_SKEW_STATIC_DESC_CLUSTER_EIDS.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(f"{cluster_name} M60 STATIC_DESC stream changed")
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} M60 STATIC_DESC stream has a gap")

    for expert_id, slot in slots_by_eid.items():
        expected_dma = DMA_IDMA if slot.cluster_name == "c0" else DMA_XDMA
        if slot.ntokens != M60_HIGH_SKEW_COUNTS[expert_id]:
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
        if (
            _task_timeline(slot, slot.reference_start_tick).task_end
            != slot.reference_end_tick
        ):
            raise AssertionError(f"E{expert_id} M60 STATIC_DESC timeline changed")

    if cross_cluster_dma_release_edges(queues):
        raise AssertionError("M60 STATIC_DESC dedicated DMA lanes need no release edges")
    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 135, "c1": 138}:
        raise AssertionError(f"M60 STATIC_DESC queue endpoints changed: {queue_ticks}")
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M60_HIGH_SKEW_STATIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError("M60 STATIC_DESC structural lower bound must be 150 ticks")
    return {
        "distribution": M60_HIGH_SKEW_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M60_HIGH_SKEW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": (),
    }


def audit_m60_high_skew_dynamic_desc_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    _audit_m60_high_skew_distribution()
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (14, 16):
        raise AssertionError("M60 DYNAMIC_DESC must contain C2=14 and C3=16 slots")

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 30 or set(slots_by_eid) != set(range(30)):
        raise AssertionError("M60 DYNAMIC_DESC must cover active E0 through E29 once")
    expected_streams = {
        "c0": (0, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29),
        "c1": (1, 2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
    }
    for cluster_name, expected_eids in expected_streams.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(f"{cluster_name} M60 DYNAMIC_DESC stream changed")
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")

    history_by_eid = {
        task.expert_id: task for task in M60_HIGH_SKEW_DYNAMIC_DESC_HISTORY
    }
    for expert_id, slot in slots_by_eid.items():
        task = history_by_eid[expert_id]
        if (
            slot.ntokens,
            slot.s1_shape,
            slot.s3_shape,
            slot.s2_prefetch_dma,
            slot.s1_dma,
            slot.s3_dma,
            slot.s4_prefetch_dma,
            slot.s4_prefetch_target_eid,
            slot.skip_s1,
            slot.reference_start_tick,
            slot.reference_end_tick,
        ) != (
            M60_HIGH_SKEW_COUNTS[expert_id],
            task.s1_shape,
            task.s3_shape,
            task.s2_prefetch_dma,
            task.s1_dma,
            task.s3_dma,
            task.s4_prefetch_dma,
            task.s4_prefetch_target_eid,
            task.skip_s1,
            task.start_tick,
            task.end_tick,
        ):
            raise AssertionError(f"E{expert_id} M60 DYNAMIC_DESC fields changed")
        if (
            _task_timeline(slot, slot.reference_start_tick).task_end
            != slot.reference_end_tick
        ):
            raise AssertionError(f"E{expert_id} M60 DYNAMIC_DESC timeline changed")

    if {
        eid for eid, slot in slots_by_eid.items()
        if slot.s2_prefetch_dma != DMA_NONE
    } != {0, 1}:
        raise AssertionError("M60 DYNAMIC_DESC S2PF choices changed")
    if {eid: slots_by_eid[eid].s2pf_s1_overlap_steps for eid in (0, 1)} != {
        0: 2,
        1: 2,
    }:
        raise AssertionError("M60 DYNAMIC_DESC early S2PF events changed")
    expected_s4pf = {
        0: (DMA_IDMA, 3),
        1: (DMA_XDMA, 2),
        2: (DMA_XDMA, 4),
    }
    actual_s4pf = {
        eid: (slot.s4_prefetch_dma, slot.s4_prefetch_target_eid)
        for eid, slot in slots_by_eid.items()
        if slot.s4_prefetch_dma != DMA_NONE
    }
    if actual_s4pf != expected_s4pf:
        raise AssertionError("M60 DYNAMIC_DESC S4PF choices changed")
    if {eid for eid, slot in slots_by_eid.items() if slot.skip_s1} != {2, 3, 4}:
        raise AssertionError("M60 DYNAMIC_DESC cache-hit tasks changed")

    expected_release_edges = {
        (("c0", 0, 0, "S4PF"), ("c1", 1, 2, "S3")),
        (("c0", 1, 3, "S3"), ("c1", 4, 6, "S1")),
        (("c1", 3, 5, "S3"), ("c0", 1, 3, "S3")),
        *(
            (("c0", slot, 2 * slot + 3, "S3"),
             ("c1", slot + 3, 2 * slot + 4, "S1"))
            for slot in range(2, 13)
        ),
        *(
            (("c1", slot, 2 * slot - 2, "S3"),
             ("c0", slot - 2, 2 * slot - 1, "S1"))
            for slot in range(4, 16)
        ),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError("M60 DYNAMIC_DESC DMA release edges changed")

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 133, "c1": 130}:
        raise AssertionError(f"M60 DYNAMIC_DESC queue endpoints changed: {queue_ticks}")
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M60_HIGH_SKEW_DYNAMIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError("M60 DYNAMIC_DESC structural lower bound must be 143.5 ticks")
    return {
        "distribution": M60_HIGH_SKEW_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M60_HIGH_SKEW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_m60_high_skew_dynamic_two_ended_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    _audit_m60_high_skew_distribution()
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (3, 27):
        raise AssertionError(
            "M60 DYNAMIC_TWO_ENDED must contain C2=3 and C3=27 slots"
        )

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 30 or set(slots_by_eid) != set(range(30)):
        raise AssertionError(
            "M60 DYNAMIC_TWO_ENDED must cover active E0 through E29 once"
        )
    expected_streams = {
        "c0": (0, 1, 2),
        "c1": tuple(range(29, 2, -1)),
    }
    for cluster_name, expected_eids in expected_streams.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(
                f"{cluster_name} M60 DYNAMIC_TWO_ENDED stream changed"
            )
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")

    history_by_eid = {
        task.expert_id: task
        for task in M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_HISTORY
    }
    for expert_id, slot in slots_by_eid.items():
        task = history_by_eid[expert_id]
        if (
            slot.ntokens,
            slot.s1_shape,
            slot.s3_shape,
            slot.s2_prefetch_dma,
            slot.s1_dma,
            slot.s3_dma,
            slot.s4_prefetch_dma,
            slot.s4_prefetch_target_eid,
            slot.skip_s1,
            slot.reference_start_tick,
            slot.reference_end_tick,
        ) != (
            M60_HIGH_SKEW_COUNTS[expert_id],
            task.s1_shape,
            task.s3_shape,
            task.s2_prefetch_dma,
            task.s1_dma,
            task.s3_dma,
            task.s4_prefetch_dma,
            task.s4_prefetch_target_eid,
            task.skip_s1,
            task.start_tick,
            task.end_tick,
        ):
            raise AssertionError(
                f"E{expert_id} M60 DYNAMIC_TWO_ENDED fields changed"
            )
        if (
            _task_timeline(slot, slot.reference_start_tick).task_end
            != slot.reference_end_tick
        ):
            raise AssertionError(
                f"E{expert_id} M60 DYNAMIC_TWO_ENDED timeline changed"
            )

    if {
        eid for eid, slot in slots_by_eid.items()
        if slot.s2_prefetch_dma != DMA_NONE
    } != {1, 2, 3}:
        raise AssertionError("M60 DYNAMIC_TWO_ENDED S2PF choices changed")
    if {
        eid: slots_by_eid[eid].s2pf_s1_overlap_steps for eid in (1, 2, 3)
    } != {1: 2, 2: 2, 3: 0}:
        raise AssertionError("M60 DYNAMIC_TWO_ENDED S2PF timing changed")
    if any(slot.s4_prefetch_dma != DMA_NONE for slot in all_slots):
        raise AssertionError("M60 DYNAMIC_TWO_ENDED must not use S4PF")
    if any(slot.skip_s1 for slot in all_slots):
        raise AssertionError("M60 DYNAMIC_TWO_ENDED must not skip S1")

    expected_release_edges = {
        (("c0", 0, 0, "S1"), ("c1", 0, 29, "S3")),
        (("c0", 0, 0, "S3"), ("c1", 11, 18, "S1")),
        (("c0", 1, 1, "S2PF"), ("c1", 17, 12, "S1")),
        (("c1", 10, 19, "S3"), ("c0", 0, 0, "S3")),
        (("c1", 16, 13, "S3"), ("c0", 1, 1, "S1")),
        (("c1", 26, 3, "S2PF"), ("c0", 2, 2, "S1")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError("M60 DYNAMIC_TWO_ENDED DMA release edges changed")

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 111, "c1": 94}:
        raise AssertionError(
            f"M60 DYNAMIC_TWO_ENDED queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError(
            "M60 DYNAMIC_TWO_ENDED structural lower bound must be 114.25 ticks"
        )
    return {
        "distribution": M60_HIGH_SKEW_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M60_HIGH_SKEW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_m60_high_skew_full_scheduler_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    _audit_m60_high_skew_distribution()
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (10, 20):
        raise AssertionError(
            "M60 FULL_SCHEDULER must contain C2=10 and C3=20 slots"
        )

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 30 or set(slots_by_eid) != set(range(30)):
        raise AssertionError(
            "M60 FULL_SCHEDULER must cover active E0 through E29 once"
        )
    expected_streams = {
        "c0": (0, *range(13, 5, -1), 2),
        "c1": (29, *range(28, 13, -1), 1, 5, 4, 3),
    }
    for cluster_name, expected_eids in expected_streams.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(
                f"{cluster_name} M60 FULL_SCHEDULER stream changed"
            )
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(
                    f"{cluster_name} M60 FULL_SCHEDULER stream has a gap"
                )

    history_by_eid = {
        task.expert_id: task for task in M60_HIGH_SKEW_FULL_SCHEDULER_HISTORY
    }
    for expert_id, slot in slots_by_eid.items():
        task = history_by_eid[expert_id]
        if (
            slot.ntokens,
            slot.token_start_rank,
            slot.s1_shape,
            slot.s3_shape,
            slot.s2_prefetch_dma,
            slot.s1_dma,
            slot.s3_dma,
            slot.s4_prefetch_dma,
            slot.s4_prefetch_target_eid,
            slot.skip_s1,
            slot.reference_start_tick,
            slot.reference_end_tick,
        ) != (
            M60_HIGH_SKEW_COUNTS[expert_id],
            0,
            task.s1_shape,
            task.s3_shape,
            task.s2_prefetch_dma,
            task.s1_dma,
            task.s3_dma,
            task.s4_prefetch_dma,
            task.s4_prefetch_target_eid,
            task.skip_s1,
            task.start_tick,
            task.end_tick,
        ):
            raise AssertionError(f"E{expert_id} M60 FULL_SCHEDULER fields changed")
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{expert_id} M60 FULL_SCHEDULER timeline ends at "
                f"{timeline.task_end}, expected {slot.reference_end_tick}"
            )

    expected_s2pf = {0: DMA_IDMA, 1: DMA_BOTH, 2: DMA_BOTH}
    actual_s2pf = {
        slot.expert_id: slot.s2_prefetch_dma
        for slot in all_slots
        if slot.s2_prefetch_dma != DMA_NONE
    }
    if actual_s2pf != expected_s2pf:
        raise AssertionError("M60 FULL_SCHEDULER S2PF choices changed")
    if any(
        slots_by_eid[eid].s2pf_s1_overlap_steps != 2
        for eid in expected_s2pf
    ):
        raise AssertionError("M60 FULL_SCHEDULER early S2PF events changed")
    if any(slot.s4_prefetch_dma != DMA_NONE for slot in all_slots):
        raise AssertionError("M60 FULL_SCHEDULER must not use S4PF")
    if any(slot.skip_s1 for slot in all_slots):
        raise AssertionError("M60 FULL_SCHEDULER must not skip S1")

    intervals = [
        interval
        for slot in all_slots
        for interval in _dma_intervals(slot)
    ]
    if len(intervals) != 60:
        raise AssertionError(
            f"M60 FULL_SCHEDULER must contain 60 DMA ops, got {len(intervals)}"
        )
    expected_release_edges = {
        (("c0", 0, 0, "S2PF"), ("c1", 1, 28, "S1")),
        (("c1", 16, 1, "S2PF"), ("c0", 1, 13, "S1")),
        (("c0", 9, 2, "S2PF"), ("c1", 17, 5, "S1")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError(
            "M60 FULL_SCHEDULER cross-cluster DMA release edges changed"
        )

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 99, "c1": 99}:
        raise AssertionError(
            f"M60 FULL_SCHEDULER queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M60_HIGH_SKEW_FULL_SCHEDULER_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError(
            "M60 FULL_SCHEDULER structural lower bound must be 114 ticks"
        )
    return {
        "distribution": M60_HIGH_SKEW_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M60_HIGH_SKEW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {
            name: len(queues[name]) for name in ("c0", "c1")
        },
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_m70_three_hot_dynamic_desc_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    _audit_m70_three_hot_distribution()
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (9, 14):
        raise AssertionError("M70 DYNAMIC_DESC must contain C2=9 and C3=14 slots")

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 23 or set(slots_by_eid) != set(range(23)):
        raise AssertionError("M70 DYNAMIC_DESC must cover E0 through E22 once")
    expected_streams = {
        "c0": (0, 2, 9, 11, 13, 15, 17, 19, 21),
        "c1": (1, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22),
    }
    for cluster_name, expected_eids in expected_streams.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(f"{cluster_name} M70 DYNAMIC_DESC stream changed")
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(f"{cluster_name} M70 DYNAMIC_DESC stream has a gap")

    history_by_eid = {
        task.expert_id: task for task in M70_THREE_HOT_DYNAMIC_DESC_HISTORY
    }
    for expert_id, slot in slots_by_eid.items():
        task = history_by_eid[expert_id]
        if (
            slot.ntokens,
            slot.s1_shape,
            slot.s3_shape,
            slot.s2_prefetch_dma,
            slot.s1_dma,
            slot.s3_dma,
            slot.s4_prefetch_dma,
            slot.s4_prefetch_target_eid,
            slot.skip_s1,
            slot.reference_start_tick,
            slot.reference_end_tick,
        ) != (
            M70_THREE_HOT_COUNTS[expert_id],
            task.s1_shape,
            task.s3_shape,
            task.s2_prefetch_dma,
            task.s1_dma,
            task.s3_dma,
            task.s4_prefetch_dma,
            task.s4_prefetch_target_eid,
            task.skip_s1,
            task.start_tick,
            task.end_tick,
        ):
            raise AssertionError(f"E{expert_id} M70 DYNAMIC_DESC fields changed")
        if _task_timeline(slot, slot.reference_start_tick).task_end != slot.reference_end_tick:
            raise AssertionError(f"E{expert_id} M70 DYNAMIC_DESC timeline changed")

    if {
        eid for eid, slot in slots_by_eid.items()
        if slot.s2_prefetch_dma != DMA_NONE
    } != {0, 1}:
        raise AssertionError("M70 DYNAMIC_DESC S2PF choices changed")
    if {eid: slots_by_eid[eid].s2pf_s1_overlap_steps for eid in (0, 1)} != {
        0: 2, 1: 2,
    }:
        raise AssertionError("M70 DYNAMIC_DESC early S2PF events changed")
    expected_s4pf = {
        0: (DMA_IDMA, 2),
        1: (DMA_XDMA, 3),
        3: (DMA_BOTH, 4),
        4: (DMA_BOTH, 5),
        5: (DMA_BOTH, 6),
        6: (DMA_BOTH, 7),
    }
    actual_s4pf = {
        eid: (slot.s4_prefetch_dma, slot.s4_prefetch_target_eid)
        for eid, slot in slots_by_eid.items()
        if slot.s4_prefetch_dma != DMA_NONE
    }
    if actual_s4pf != expected_s4pf:
        raise AssertionError("M70 DYNAMIC_DESC S4PF choices changed")
    if {eid for eid, slot in slots_by_eid.items() if slot.skip_s1} != {
        2, 3, 4, 5, 6, 7,
    }:
        raise AssertionError("M70 DYNAMIC_DESC cache-hit tasks changed")

    expected_release_edges = {
        (("c0", 0, 0, "S4PF"), ("c1", 1, 3, "S3")),
        (("c0", 1, 2, "S3"), ("c1", 4, 6, "S3")),
        (("c1", 3, 5, "S4PF"), ("c0", 1, 2, "S3")),
        (("c1", 6, 8, "S3"), ("c0", 2, 9, "S1")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError("M70 DYNAMIC_DESC DMA release edges changed")

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 126, "c1": 126}:
        raise AssertionError(f"M70 DYNAMIC_DESC queue endpoints changed: {queue_ticks}")
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if structural_quarters != M70_THREE_HOT_DYNAMIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS:
        raise AssertionError("M70 DYNAMIC_DESC structural lower bound must be 136.5 ticks")
    return {
        "distribution": M70_THREE_HOT_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M70_THREE_HOT_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_m92_parameter_order_dynamic_desc_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(M92_PARAMETER_ORDER_COUNTS) != EXPERT_COUNT:
        raise AssertionError("M92 DYNAMIC_DESC must define 64 conceptual experts")
    if sum(M92_PARAMETER_ORDER_COUNTS) != 184:
        raise AssertionError("M92 DYNAMIC_DESC assignment total must be 184")
    if tuple(
        len(token_ids) for token_ids in M92_PARAMETER_ORDER_TOKEN_IDS_BY_EXPERT
    ) != M92_PARAMETER_ORDER_COUNTS:
        raise AssertionError("M92 DYNAMIC_DESC routing does not match expert loads")

    token_owners = [[] for _ in range(92)]
    for expert_id, token_ids in enumerate(M92_PARAMETER_ORDER_TOKEN_IDS_BY_EXPERT):
        for token_id in token_ids:
            if not 0 <= token_id < len(token_owners):
                raise AssertionError(f"E{expert_id} has invalid token {token_id}")
            token_owners[token_id].append(expert_id)
    if any(len(owners) != 2 for owners in token_owners):
        raise AssertionError("M92 DYNAMIC_DESC must route every input token twice")

    if tuple(len(queues[name]) for name in ("c0", "c1")) != (10, 28):
        raise AssertionError(
            "M92 DYNAMIC_DESC must contain C2=10 and C3=28 slots"
        )
    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 38 or set(slots_by_eid) != set(range(38)):
        raise AssertionError(
            "M92 DYNAMIC_DESC must cover active E0 through E37 once"
        )

    expected_streams = {
        "c0": (0, 20, 22, 24, 26, 28, 30, 32, 34, 36),
        "c1": (
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37,
        ),
    }
    for cluster_name, expected_eids in expected_streams.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(f"{cluster_name} M92 DYNAMIC_DESC stream changed")
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(
                    f"{cluster_name} M92 DYNAMIC_DESC stream has a gap"
                )

    history_by_eid = {
        task.expert_id: task for task in M92_PARAMETER_ORDER_DYNAMIC_DESC_HISTORY
    }
    for expert_id, slot in slots_by_eid.items():
        task = history_by_eid[expert_id]
        if (
            slot.ntokens,
            slot.s1_shape,
            slot.s3_shape,
            slot.s2_prefetch_dma,
            slot.s1_dma,
            slot.s3_dma,
            slot.s4_prefetch_dma,
            slot.s4_prefetch_target_eid,
            slot.skip_s1,
            slot.reference_start_tick,
            slot.reference_end_tick,
        ) != (
            M92_PARAMETER_ORDER_COUNTS[expert_id],
            task.s1_shape,
            task.s3_shape,
            task.s2_prefetch_dma,
            task.s1_dma,
            task.s3_dma,
            task.s4_prefetch_dma,
            task.s4_prefetch_target_eid,
            task.skip_s1,
            task.start_tick,
            task.end_tick,
        ):
            raise AssertionError(f"E{expert_id} M92 DYNAMIC_DESC fields changed")
        if (
            _task_timeline(slot, slot.reference_start_tick).task_end
            != slot.reference_end_tick
        ):
            raise AssertionError(f"E{expert_id} M92 DYNAMIC_DESC timeline changed")

    if {
        eid for eid, slot in slots_by_eid.items()
        if slot.s2_prefetch_dma != DMA_NONE
    } != {0, 1}:
        raise AssertionError("M92 DYNAMIC_DESC S2PF choices changed")
    if {eid: slots_by_eid[eid].s2pf_s1_overlap_steps for eid in (0, 1)} != {
        0: 2, 1: 2,
    }:
        raise AssertionError("M92 DYNAMIC_DESC early S2PF events changed")
    actual_s4pf = {
        eid: (slot.s4_prefetch_dma, slot.s4_prefetch_target_eid)
        for eid, slot in slots_by_eid.items()
        if slot.s4_prefetch_dma != DMA_NONE
    }
    if actual_s4pf != {1: (DMA_XDMA, 2)}:
        raise AssertionError("M92 DYNAMIC_DESC S4PF choices changed")
    if {eid for eid, slot in slots_by_eid.items() if slot.skip_s1} != {2}:
        raise AssertionError("M92 DYNAMIC_DESC cache-hit tasks changed")

    expected_release_edges = {
        (("c0", 0, 0, "S2PF"), ("c1", 1, 2, "S3")),
        (("c1", 18, 19, "S3"), ("c0", 1, 20, "S1")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError("M92 DYNAMIC_DESC DMA release edges changed")

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 168, "c1": 168}:
        raise AssertionError(
            f"M92 DYNAMIC_DESC queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M92_PARAMETER_ORDER_DYNAMIC_DESC_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError(
            "M92 DYNAMIC_DESC structural lower bound must be 189 ticks"
        )
    return {
        "distribution": M92_PARAMETER_ORDER_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M92_PARAMETER_ORDER_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {
            name: len(queues[name]) for name in ("c0", "c1")
        },
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_m92_parameter_order_dynamic_two_ended_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(M92_PARAMETER_ORDER_COUNTS) != EXPERT_COUNT:
        raise AssertionError(
            "M92 DYNAMIC_TWO_ENDED must define 64 conceptual experts"
        )
    if sum(M92_PARAMETER_ORDER_COUNTS) != 184:
        raise AssertionError("M92 DYNAMIC_TWO_ENDED assignment total must be 184")
    if tuple(
        len(token_ids) for token_ids in M92_PARAMETER_ORDER_TOKEN_IDS_BY_EXPERT
    ) != M92_PARAMETER_ORDER_COUNTS:
        raise AssertionError(
            "M92 DYNAMIC_TWO_ENDED routing does not match expert loads"
        )

    token_owners = [[] for _ in range(92)]
    for expert_id, token_ids in enumerate(M92_PARAMETER_ORDER_TOKEN_IDS_BY_EXPERT):
        for token_id in token_ids:
            if not 0 <= token_id < len(token_owners):
                raise AssertionError(f"E{expert_id} has invalid token {token_id}")
            token_owners[token_id].append(expert_id)
    if any(len(owners) != 2 for owners in token_owners):
        raise AssertionError(
            "M92 DYNAMIC_TWO_ENDED must route every input token twice"
        )

    if tuple(len(queues[name]) for name in ("c0", "c1")) != (1, 37):
        raise AssertionError(
            "M92 DYNAMIC_TWO_ENDED must contain C2=1 and C3=37 slots"
        )
    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 38 or set(slots_by_eid) != set(range(38)):
        raise AssertionError(
            "M92 DYNAMIC_TWO_ENDED must cover active E0 through E37 once"
        )

    expected_streams = {
        "c0": (0,),
        "c1": tuple(range(37, 0, -1)),
    }
    for cluster_name, expected_eids in expected_streams.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(
                f"{cluster_name} M92 DYNAMIC_TWO_ENDED stream changed"
            )
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(
                    f"{cluster_name} M92 DYNAMIC_TWO_ENDED stream has a gap"
                )

    history_by_eid = {
        task.expert_id: task
        for task in M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_HISTORY
    }
    for expert_id, slot in slots_by_eid.items():
        task = history_by_eid[expert_id]
        if (
            slot.ntokens,
            slot.s1_shape,
            slot.s3_shape,
            slot.s2_prefetch_dma,
            slot.s1_dma,
            slot.s3_dma,
            slot.s4_prefetch_dma,
            slot.s4_prefetch_target_eid,
            slot.skip_s1,
            slot.reference_start_tick,
            slot.reference_end_tick,
        ) != (
            M92_PARAMETER_ORDER_COUNTS[expert_id],
            task.s1_shape,
            task.s3_shape,
            task.s2_prefetch_dma,
            task.s1_dma,
            task.s3_dma,
            task.s4_prefetch_dma,
            task.s4_prefetch_target_eid,
            task.skip_s1,
            task.start_tick,
            task.end_tick,
        ):
            raise AssertionError(
                f"E{expert_id} M92 DYNAMIC_TWO_ENDED fields changed"
            )
        if (
            _task_timeline(slot, slot.reference_start_tick).task_end
            != slot.reference_end_tick
        ):
            raise AssertionError(
                f"E{expert_id} M92 DYNAMIC_TWO_ENDED timeline changed"
            )

    if {
        eid for eid, slot in slots_by_eid.items()
        if slot.s2_prefetch_dma != DMA_NONE
    } != {1}:
        raise AssertionError("M92 DYNAMIC_TWO_ENDED S2PF choices changed")
    if slots_by_eid[1].s2pf_s1_overlap_steps != 2:
        raise AssertionError("M92 DYNAMIC_TWO_ENDED early S2PF event changed")
    if any(slot.s4_prefetch_dma != DMA_NONE for slot in all_slots):
        raise AssertionError("M92 DYNAMIC_TWO_ENDED must not use S4PF")
    if any(slot.skip_s1 for slot in all_slots):
        raise AssertionError("M92 DYNAMIC_TWO_ENDED must not use cache hits")

    expected_release_edges = {
        (("c0", 0, 0, "S1"), ("c1", 0, 37, "S3")),
        (("c0", 0, 0, "S3"), ("c1", 24, 13, "S3")),
        (("c1", 24, 13, "S1"), ("c0", 0, 0, "S3")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError("M92 DYNAMIC_TWO_ENDED DMA release edges changed")

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 114, "c1": 172}:
        raise AssertionError(
            f"M92 DYNAMIC_TWO_ENDED queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError(
            "M92 DYNAMIC_TWO_ENDED structural lower bound must be 199.75 ticks"
        )
    return {
        "distribution": M92_PARAMETER_ORDER_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M92_PARAMETER_ORDER_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {
            name: len(queues[name]) for name in ("c0", "c1")
        },
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_m92_parameter_order_full_scheduler_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(M92_PARAMETER_ORDER_COUNTS) != EXPERT_COUNT:
        raise AssertionError("M92 FULL_SCHEDULER expert count changed")
    if sum(M92_PARAMETER_ORDER_COUNTS) != 184:
        raise AssertionError("M92 FULL_SCHEDULER routed-token count changed")
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (10, 28):
        raise AssertionError(
            "M92 FULL_SCHEDULER must contain C2=10 and C3=28 slots"
        )

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 38 or set(slots_by_eid) != set(range(38)):
        raise AssertionError(
            "M92 FULL_SCHEDULER must cover E0 through E37 once"
        )
    expected_streams = {
        "c0": (0, *range(10, 1, -1)),
        "c1": (37, *range(36, 10, -1), 1),
    }
    for cluster_name, expected_eids in expected_streams.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(
                f"{cluster_name} M92 FULL_SCHEDULER stream changed"
            )
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(
                    f"{cluster_name} M92 FULL_SCHEDULER stream has a gap"
                )

    for expert_id, slot in slots_by_eid.items():
        if slot.ntokens != M92_PARAMETER_ORDER_COUNTS[expert_id]:
            raise AssertionError(f"M92 E{expert_id} token count changed")
        if slot.token_start_rank != 0 or slot.skip_s1:
            raise AssertionError(f"M92 E{expert_id} must execute one complete task")
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(
                f"M92 E{expert_id} timeline ends at {timeline.task_end}, "
                f"expected {slot.reference_end_tick}"
            )

    expected_s2pf = {0: DMA_IDMA, 1: DMA_BOTH}
    actual_s2pf = {
        slot.expert_id: slot.s2_prefetch_dma
        for slot in all_slots
        if slot.s2_prefetch_dma != DMA_NONE
    }
    if actual_s2pf != expected_s2pf:
        raise AssertionError("M92 FULL_SCHEDULER S2PF choices changed")
    if any(slots_by_eid[eid].s2pf_s1_overlap_steps != 2 for eid in expected_s2pf):
        raise AssertionError("M92 FULL_SCHEDULER early S2PF events changed")
    if any(slot.s4_prefetch_dma != DMA_NONE for slot in all_slots):
        raise AssertionError("M92 FULL_SCHEDULER must not contain S4PF")

    if (
        slots_by_eid[0].s1_shape,
        slots_by_eid[0].s3_shape,
        slots_by_eid[0].s1_dma,
        slots_by_eid[0].s3_dma,
    ) != (SHAPE_A, SHAPE_B, DMA_IDMA, DMA_NONE):
        raise AssertionError("M92 E0 physical profile changed")
    if (
        slots_by_eid[1].s1_shape,
        slots_by_eid[1].s3_shape,
        slots_by_eid[1].s1_dma,
        slots_by_eid[1].s3_dma,
    ) != (SHAPE_B, SHAPE_B, DMA_BOTH, DMA_NONE):
        raise AssertionError("M92 E1 physical profile changed")
    if (
        slots_by_eid[37].s1_shape,
        slots_by_eid[37].s3_shape,
        slots_by_eid[37].s1_dma,
        slots_by_eid[37].s3_dma,
    ) != (SHAPE_B, SHAPE_B, DMA_XDMA, DMA_XDMA):
        raise AssertionError("M92 E37 physical profile changed")
    for expert_id in range(2, 37):
        slot = slots_by_eid[expert_id]
        if (
            slot.s1_shape,
            slot.s3_shape,
            slot.s1_dma,
            slot.s3_dma,
        ) != (SHAPE_C, SHAPE_C, DMA_BOTH, DMA_BOTH):
            raise AssertionError(f"M92 E{expert_id} C/C profile changed")

    intervals = [
        interval
        for slot in all_slots
        for interval in _dma_intervals(slot)
    ]
    if len(intervals) != 76:
        raise AssertionError(
            f"M92 FULL_SCHEDULER must contain 76 DMA ops, got {len(intervals)}"
        )
    expected_release_edges = {
        (("c0", 0, 0, "S2PF"), ("c1", 1, 36, "S1")),
        (("c1", 27, 1, "S2PF"), ("c0", 1, 10, "S1")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError(
            "M92 FULL_SCHEDULER cross-cluster DMA release edges changed"
        )

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 141, "c1": 144}:
        raise AssertionError(
            f"M92 FULL_SCHEDULER queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M92_PARAMETER_ORDER_FULL_SCHEDULER_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError(
            "M92 FULL_SCHEDULER structural lower bound must be 165 ticks"
        )
    return {
        "distribution": M92_PARAMETER_ORDER_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M92_PARAMETER_ORDER_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {
            name: len(queues[name]) for name in ("c0", "c1")
        },
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_m70_three_hot_dynamic_two_ended_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    _audit_m70_three_hot_distribution()
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (3, 20):
        raise AssertionError(
            "M70 DYNAMIC_TWO_ENDED must contain C2=3 and C3=20 slots"
        )

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 23 or set(slots_by_eid) != set(range(23)):
        raise AssertionError(
            "M70 DYNAMIC_TWO_ENDED must cover E0 through E22 once"
        )
    expected_streams = {
        "c0": (0, 1, 2),
        "c1": tuple(range(22, 2, -1)),
    }
    for cluster_name, expected_eids in expected_streams.items():
        slots = queues[cluster_name]
        if tuple(slot.expert_id for slot in slots) != expected_eids:
            raise AssertionError(
                f"{cluster_name} M70 DYNAMIC_TWO_ENDED stream changed"
            )

    history_by_eid = {
        task.expert_id: task
        for task in M70_THREE_HOT_DYNAMIC_TWO_ENDED_HISTORY
    }
    for expert_id, slot in slots_by_eid.items():
        task = history_by_eid[expert_id]
        if (
            slot.ntokens,
            slot.s1_shape,
            slot.s3_shape,
            slot.s2_prefetch_dma,
            slot.s1_dma,
            slot.s3_dma,
            slot.s4_prefetch_dma,
            slot.skip_s1,
            slot.reference_start_tick,
            slot.reference_end_tick,
        ) != (
            M70_THREE_HOT_COUNTS[expert_id],
            task.s1_shape,
            task.s3_shape,
            task.s2_prefetch_dma,
            task.s1_dma,
            task.s3_dma,
            DMA_NONE,
            False,
            task.start_tick,
            task.end_tick,
        ):
            raise AssertionError(
                f"E{expert_id} M70 DYNAMIC_TWO_ENDED fields changed"
            )
        if _task_timeline(slot, slot.reference_start_tick).task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{expert_id} M70 DYNAMIC_TWO_ENDED timeline changed"
            )

    expected_s2pf = {1, 2, 3, 4, 5, 6}
    if {
        eid for eid, slot in slots_by_eid.items()
        if slot.s2_prefetch_dma != DMA_NONE
    } != expected_s2pf:
        raise AssertionError("M70 DYNAMIC_TWO_ENDED S2PF choices changed")
    if {
        eid: slots_by_eid[eid].s2pf_s1_overlap_steps for eid in expected_s2pf
    } != {eid: 2 for eid in expected_s2pf}:
        raise AssertionError(
            "M70 DYNAMIC_TWO_ENDED early S2PF events changed"
        )

    expected_release_edges = {
        (("c0", 0, 0, "S1"), ("c1", 0, 22, "S3")),
        (("c0", 0, 0, "S3"), ("c1", 8, 14, "S3")),
        (("c0", 1, 1, "S2PF"), ("c1", 13, 9, "S1")),
        (("c1", 8, 14, "S1"), ("c0", 0, 0, "S3")),
        (("c1", 12, 10, "S3"), ("c0", 1, 1, "S1")),
        (("c1", 19, 3, "S2PF"), ("c0", 2, 2, "S1")),
        (("c1", 19, 3, "S2PF"), ("c0", 2, 2, "S2PF")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError(
            "M70 DYNAMIC_TWO_ENDED DMA release edges changed"
        )

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 127, "c1": 91}:
        raise AssertionError(
            f"M70 DYNAMIC_TWO_ENDED queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M70_THREE_HOT_DYNAMIC_TWO_ENDED_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError(
            "M70 DYNAMIC_TWO_ENDED structural lower bound must be 129.25 ticks"
        )
    return {
        "distribution": M70_THREE_HOT_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M70_THREE_HOT_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_m70_three_hot_full_scheduler_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    _audit_m70_three_hot_distribution()
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (7, 16):
        raise AssertionError(
            "M70 FULL_SCHEDULER must contain C2=7 and C3=16 slots"
        )

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 23 or set(slots_by_eid) != set(range(23)):
        raise AssertionError(
            "M70 FULL_SCHEDULER must cover E0 through E22 once"
        )
    expected_streams = {
        "c0": (2, 4, 1, 10, 9, 8, 7),
        "c1": (3, *range(22, 11, -1), 5, 6, 11, 0),
    }
    for cluster_name, expected_eids in expected_streams.items():
        if tuple(slot.expert_id for slot in queues[cluster_name]) != expected_eids:
            raise AssertionError(
                f"{cluster_name} M70 FULL_SCHEDULER stream changed"
            )

    history_by_eid = {
        task.expert_id: task for task in M70_THREE_HOT_FULL_SCHEDULER_HISTORY
    }
    for expert_id, slot in slots_by_eid.items():
        task = history_by_eid[expert_id]
        if (
            slot.ntokens,
            slot.s1_shape,
            slot.s3_shape,
            slot.s2_prefetch_dma,
            slot.s1_dma,
            slot.s3_dma,
            slot.s4_prefetch_dma,
            slot.skip_s1,
            slot.reference_start_tick,
            slot.reference_end_tick,
        ) != (
            M70_THREE_HOT_COUNTS[expert_id],
            task.s1_shape,
            task.s3_shape,
            task.s2_prefetch_dma,
            task.s1_dma,
            task.s3_dma,
            DMA_NONE,
            False,
            task.start_tick,
            task.end_tick,
        ):
            raise AssertionError(
                f"E{expert_id} M70 FULL_SCHEDULER fields changed"
            )
        if _task_timeline(slot, slot.reference_start_tick).task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{expert_id} M70 FULL_SCHEDULER timeline changed"
            )

    expected_s2pf = {
        0: DMA_BOTH,
        1: DMA_IDMA,
        2: DMA_IDMA,
        3: DMA_XDMA,
        4: DMA_IDMA,
        5: DMA_XDMA,
        6: DMA_XDMA,
    }
    actual_s2pf = {
        eid: slot.s2_prefetch_dma for eid, slot in slots_by_eid.items()
        if slot.s2_prefetch_dma != DMA_NONE
    }
    if actual_s2pf != expected_s2pf:
        raise AssertionError("M70 FULL_SCHEDULER S2PF choices changed")
    if {
        eid: slots_by_eid[eid].s2pf_s1_overlap_steps for eid in expected_s2pf
    } != {0: 2, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}:
        raise AssertionError("M70 FULL_SCHEDULER S2PF start events changed")

    expected_release_edges = {
        (("c0", 0, 2, "S2PF"), ("c1", 1, 22, "S1")),
        (("c0", 2, 1, "S2PF"), ("c1", 14, 11, "S1")),
        (("c1", 11, 12, "S3"), ("c0", 1, 4, "S1")),
        (("c1", 15, 0, "S2PF"), ("c0", 3, 10, "S1")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError("M70 FULL_SCHEDULER DMA release edges changed")

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 105, "c1": 105}:
        raise AssertionError(
            f"M70 FULL_SCHEDULER queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != M70_THREE_HOT_FULL_SCHEDULER_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError(
            "M70 FULL_SCHEDULER structural lower bound must be 117 ticks"
        )
    return {
        "distribution": M70_THREE_HOT_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(M70_THREE_HOT_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {name: len(queues[name]) for name in ("c0", "c1")},
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
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


def audit_dynamic_two_ended_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(HIGH_TO_LOW_COUNTS) != EXPERT_COUNT or sum(HIGH_TO_LOW_COUNTS) != 140:
        raise AssertionError("DYNAMIC_TWO_ENDED distribution contract changed")
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (12, 31):
        raise AssertionError(
            "DYNAMIC_TWO_ENDED must contain C2=12 and C3=31 slots"
        )

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 43 or set(slots_by_eid) != set(range(43)):
        raise AssertionError(
            "DYNAMIC_TWO_ENDED must cover E0 through E42 once"
        )
    if tuple(slot.expert_id for slot in queues["c0"]) != tuple(range(12)):
        raise AssertionError("C2 must consume the hot end in ascending EID order")
    if tuple(slot.expert_id for slot in queues["c1"]) != tuple(
        range(42, 11, -1)
    ):
        raise AssertionError("C3 must consume the cold end in descending EID order")

    for expert_id, slot in slots_by_eid.items():
        if slot.ntokens != HIGH_TO_LOW_COUNTS[expert_id]:
            raise AssertionError(f"E{expert_id} token count changed")
        if slot.token_start_rank != 0:
            raise AssertionError(f"E{expert_id} must cover its complete token slice")
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

    expected_s2pf_eids = {1, *range(3, 20), 21}
    actual_s2pf_eids = {
        slot.expert_id
        for slot in all_slots
        if slot.s2_prefetch_dma != DMA_NONE
    }
    if actual_s2pf_eids != expected_s2pf_eids:
        raise AssertionError("DYNAMIC_TWO_ENDED S2PF choices changed")
    if any(
        slot.s2pf_s1_overlap_steps != 2
        for slot in all_slots
        if slot.s2_prefetch_dma != DMA_NONE
    ):
        raise AssertionError("DYNAMIC_TWO_ENDED S2PF must use the early path")

    expected_s4pf = {21: (DMA_BOTH, 20)}
    actual_s4pf = {
        slot.expert_id: (
            slot.s4_prefetch_dma,
            slot.s4_prefetch_target_eid,
        )
        for slot in all_slots
        if slot.s4_prefetch_dma != DMA_NONE
    }
    if actual_s4pf != expected_s4pf:
        raise AssertionError("DYNAMIC_TWO_ENDED S4PF choice changed")
    if {slot.expert_id for slot in all_slots if slot.skip_s1} != {20}:
        raise AssertionError("DYNAMIC_TWO_ENDED S4PF cache hit changed")

    expected_special = {
        0: (SHAPE_A, SHAPE_B, DMA_IDMA, DMA_BOTH, DMA_NONE),
        1: (SHAPE_A, SHAPE_B, DMA_BOTH, DMA_NONE, DMA_BOTH),
        2: (SHAPE_A, SHAPE_B, DMA_IDMA, DMA_BOTH, DMA_NONE),
        42: (SHAPE_B, SHAPE_C, DMA_XDMA, DMA_BOTH, DMA_NONE),
        36: (SHAPE_B, SHAPE_C, DMA_BOTH, DMA_BOTH, DMA_NONE),
        24: (SHAPE_B, SHAPE_C, DMA_XDMA, DMA_BOTH, DMA_NONE),
        21: (SHAPE_B, SHAPE_B, DMA_BOTH, DMA_NONE, DMA_BOTH),
        20: (SHAPE_A, SHAPE_B, DMA_NONE, DMA_BOTH, DMA_NONE),
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
        raise AssertionError(
            "DYNAMIC_TWO_ENDED special shape/DMA profiles changed"
        )

    for cluster_name, slots in queues.items():
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick < previous.reference_end_tick:
                raise AssertionError(
                    f"{cluster_name} DYNAMIC_TWO_ENDED local tasks overlap"
                )

    intervals = [
        (start, end, dma, slot.cluster_name, slot.expert_id, stage)
        for slot in all_slots
        for start, end, dma, stage in _dma_intervals(slot)
    ]
    if len(intervals) != 86:
        raise AssertionError(
            f"DYNAMIC_TWO_ENDED must contain 86 DMA ops, got {len(intervals)}"
        )
    for index, left in enumerate(intervals):
        for right in intervals[index + 1:]:
            if left[2] & right[2] and max(left[0], right[0]) < min(
                left[1], right[1]
            ):
                raise AssertionError(
                    f"DMA overlap {left[3]}/E{left[4]}:{left[5]} {left[:2]} "
                    f"and {right[3]}/E{right[4]}:{right[5]} {right[:2]}"
                )

    expected_release_edges = {
        (("c0", 0, 0, "S1"), ("c1", 0, 42, "S3")),
        (("c0", 0, 0, "S3"), ("c1", 6, 36, "S3")),
        (("c0", 1, 1, "S2PF"), ("c1", 10, 32, "S1")),
        (("c0", 2, 2, "S1"), ("c1", 18, 24, "S3")),
        (("c0", 2, 2, "S3"), ("c1", 21, 21, "S4PF")),
        (("c1", 6, 36, "S1"), ("c0", 0, 0, "S3")),
        (("c1", 9, 33, "S3"), ("c0", 1, 1, "S1")),
        (("c1", 17, 25, "S3"), ("c0", 2, 2, "S1")),
        (("c1", 21, 21, "S2PF"), ("c0", 2, 2, "S3")),
        (("c1", 22, 20, "S3"), ("c0", 3, 3, "S1")),
        *{
            (
                ("c0", expert_id, expert_id, "S2PF"),
                ("c1", expert_id + 20, 22 - expert_id, "S1"),
            )
            for expert_id in range(3, 11)
        },
        *{
            (
                ("c1", source_slot, source_eid, "S2PF"),
                ("c0", target_eid, target_eid, "S1"),
            )
            for source_slot, source_eid, target_eid in zip(
                range(23, 31), range(19, 11, -1), range(4, 12)
            )
        },
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError(
            "DYNAMIC_TWO_ENDED cross-cluster DMA release edges changed"
        )

    queue_ticks = {
        name: max(slot.reference_end_tick for slot in slots)
        for name, slots in queues.items()
    }
    if queue_ticks != {"c0": 137, "c1": 134}:
        raise AssertionError(
            f"DYNAMIC_TWO_ENDED queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if (
        structural_quarters
        != DYNAMIC_TWO_ENDED_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS
    ):
        raise AssertionError(
            "DYNAMIC_TWO_ENDED structural lower bound must be 157.25 ticks"
        )
    return {
        "distribution": HIGH_TO_LOW_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(HIGH_TO_LOW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {
            name: len(queues[name]) for name in ("c0", "c1")
        },
        "api_quarter_ticks_per_slot": STRUCTURAL_API_QUARTER_TICKS_PER_SLOT,
        "structural_cluster_quarter_ticks": cluster_quarter_ticks,
        "structural_lower_bound_quarter_ticks": structural_quarters,
        "dma_release_edges": release_edges,
    }


def audit_full_scheduler_schedule(
    queues: dict[str, tuple[SlotSpec, ...]],
) -> dict[str, object]:
    if len(HIGH_TO_LOW_COUNTS) != EXPERT_COUNT or sum(HIGH_TO_LOW_COUNTS) != 140:
        raise AssertionError("FULL_SCHEDULER distribution contract changed")
    if tuple(len(queues[name]) for name in ("c0", "c1")) != (11, 32):
        raise AssertionError("FULL_SCHEDULER must contain C2=11 and C3=32 slots")

    all_slots = queues["c0"] + queues["c1"]
    slots_by_eid = {slot.expert_id: slot for slot in all_slots}
    if len(all_slots) != 43 or set(slots_by_eid) != set(range(43)):
        raise AssertionError("FULL_SCHEDULER must cover E0 through E42 once")
    expected_streams = {
        "c0": (0, 1, 2, 5, 8, 10, 12, 14, 16, 18, 20),
        "c1": (
            4, *range(42, 33, -1), 6, *range(33, 26, -1), 3,
            *range(26, 21, -1), 7, 9, 11, 13, 15, 17, 19, 21,
        ),
    }
    for cluster_name, expected in expected_streams.items():
        actual = tuple(slot.expert_id for slot in queues[cluster_name])
        if actual != expected:
            raise AssertionError(
                f"{cluster_name} FULL_SCHEDULER stream changed"
            )

    for expert_id, slot in slots_by_eid.items():
        if slot.ntokens != HIGH_TO_LOW_COUNTS[expert_id]:
            raise AssertionError(f"E{expert_id} token count changed")
        if slot.token_start_rank != 0 or slot.skip_s1:
            raise AssertionError(f"E{expert_id} must execute one complete task")
        timeline = _task_timeline(slot, slot.reference_start_tick)
        if timeline.task_end != slot.reference_end_tick:
            raise AssertionError(
                f"E{expert_id} timeline ends at {timeline.task_end}, "
                f"expected {slot.reference_end_tick}"
            )

    expected_s2pf = {0, 1, 2}
    actual_s2pf = {
        slot.expert_id for slot in all_slots
        if slot.s2_prefetch_dma != DMA_NONE
    }
    if actual_s2pf != expected_s2pf:
        raise AssertionError("FULL_SCHEDULER S2PF choices changed")
    if any(slots_by_eid[eid].s2_prefetch_dma != DMA_IDMA for eid in expected_s2pf):
        raise AssertionError("FULL_SCHEDULER hot S2PF must use IDMA")
    if {
        eid: slots_by_eid[eid].s2pf_s1_overlap_steps for eid in expected_s2pf
    } != {0: 2, 1: 2, 2: 0}:
        raise AssertionError("FULL_SCHEDULER S2PF start events changed")
    if any(slot.s4_prefetch_dma != DMA_NONE for slot in all_slots):
        raise AssertionError("FULL_SCHEDULER must not contain S4PF")

    for expert_id in (0, 1):
        slot = slots_by_eid[expert_id]
        if (
            slot.s1_shape, slot.s3_shape, slot.s1_dma, slot.s3_dma
        ) != (SHAPE_A, SHAPE_B, DMA_IDMA, DMA_NONE):
            raise AssertionError(f"E{expert_id} hot physical profile changed")
    e2 = slots_by_eid[2]
    if (e2.s1_shape, e2.s3_shape, e2.s1_dma, e2.s3_dma) != (
        SHAPE_B, SHAPE_B, DMA_IDMA, DMA_NONE,
    ):
        raise AssertionError("E2 hot physical profile changed")

    for expert_id in range(22, 43):
        slot = slots_by_eid[expert_id]
        if (
            slot.cluster_name,
            slot.s1_shape,
            slot.s3_shape,
            slot.s1_dma,
            slot.s3_dma,
        ) != ("c1", SHAPE_C, SHAPE_C, DMA_BOTH, DMA_BOTH):
            raise AssertionError(f"E{expert_id} C/C tail profile changed")

    for cluster_name, slots in queues.items():
        if slots[0].reference_start_tick != 0:
            raise AssertionError(f"{cluster_name} must start at tick 0")
        for previous, current in zip(slots, slots[1:]):
            if current.reference_start_tick != previous.reference_end_tick:
                raise AssertionError(
                    f"{cluster_name} FULL_SCHEDULER stream has a gap"
                )

    expected_release_edges = {
        (("c0", 0, 0, "S2PF"), ("c1", 1, 42, "S1")),
        (("c0", 1, 1, "S2PF"), ("c1", 11, 33, "S1")),
        (("c0", 2, 2, "S2PF"), ("c1", 19, 26, "S1")),
        (("c1", 9, 34, "S3"), ("c0", 1, 1, "S1")),
        (("c1", 17, 27, "S3"), ("c0", 2, 2, "S1")),
        (("c1", 23, 22, "S3"), ("c0", 3, 5, "S1")),
    }
    release_edges = cross_cluster_dma_release_edges(queues)
    if set(release_edges) != expected_release_edges:
        raise AssertionError("FULL_SCHEDULER DMA release edges changed")

    queue_ticks = {
        name: queues[name][-1].reference_end_tick for name in ("c0", "c1")
    }
    if queue_ticks != {"c0": 129, "c1": 129}:
        raise AssertionError(
            f"FULL_SCHEDULER queue endpoints changed: {queue_ticks}"
        )
    cluster_quarter_ticks = {
        name: 4 * queue_ticks[name]
        + len(queues[name]) * STRUCTURAL_API_QUARTER_TICKS_PER_SLOT
        for name in ("c0", "c1")
    }
    structural_quarters = max(cluster_quarter_ticks.values())
    if structural_quarters != FULL_SCHEDULER_STRUCTURAL_LOWER_BOUND_QUARTER_TICKS:
        raise AssertionError("FULL_SCHEDULER structural lower bound must be 153 ticks")
    return {
        "distribution": HIGH_TO_LOW_COUNTS,
        "active_experts": len(all_slots),
        "task_count": len(all_slots),
        "routed_tokens": sum(HIGH_TO_LOW_COUNTS),
        "queue_ticks": queue_ticks,
        "makespan_ticks": max(queue_ticks.values()),
        "cluster_local_slots": {
            name: len(queues[name]) for name in ("c0", "c1")
        },
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
    if profile in M8_COMPARISON_RUN_PROFILES:
        audit = audit_m8_comparison_schedule(queues, profile)
        title = f"{profile} schedule:"
        detail = (
            "  input_tokens=8 top_k=2 distribution=E0:6,E1:4,E63:6; "
            f"dma_release_edges={len(audit['dma_release_edges'])}"
        )
    elif profile in M32_COMPARISON_RUN_PROFILES:
        audit = audit_m32_comparison_schedule(queues, profile)
        title = f"{profile} schedule:"
        detail = (
            "  input_tokens=32 top_k=2 distribution="
            "8,6,5,4,3,2x14,1x10; empty_cache=1; "
            f"dma_release_edges={len(audit['dma_release_edges'])}"
        )
    elif profile == M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE:
        audit = audit_m60_high_skew_full_scheduler_schedule(queues)
        title = "M60 HIGH_SKEW FULL_SCHEDULER schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=36,22,13,6,2x17,1x9; input_tokens=60; "
            "scheduler=order,cluster,shape,DMA,S2PF; certified_model=99 ticks\n"
            f"  dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE:
        audit = audit_m60_high_skew_dynamic_two_ended_schedule(queues)
        title = "M60 HIGH_SKEW DYNAMIC_TWO_ENDED schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=36,22,13,6,2x17,1x9; input_tokens=60; "
            "C2=hot-end,C3=cold-end; dynamic_physical=shape,DMA,S2PF\n"
            "  certified_model=111 ticks; "
            f"dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE:
        audit = audit_m60_high_skew_dynamic_desc_schedule(queues)
        title = "M60 HIGH_SKEW DYNAMIC_DESC schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=36,22,13,6,2x17,1x9; input_tokens=60; "
            "dynamic_physical=shape,DMA,S2PF,S4PF,cache\n"
            "  certified_model=133 ticks; "
            f"dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M60_HIGH_SKEW_STATIC_DESC_PROFILE:
        audit = audit_m60_high_skew_static_desc_schedule(queues)
        title = "M60 HIGH_SKEW STATIC_DESC schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=36,22,13,6,2x17,1x9; input_tokens=60; "
            "fixed_physical=S1:B,S3:B,C2:IDMA,C3:XDMA,prefetch:off\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M92_PARAMETER_ORDER_FULL_SCHEDULER_PROFILE:
        audit = audit_m92_parameter_order_full_scheduler_schedule(queues)
        title = "M92 PARAMETER_ORDER FULL_SCHEDULER schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=76,40,2x32,1x4; input_tokens=92; "
            "scheduler=order,cluster,shape,DMA,S2PF; certified_model=144 ticks\n"
            f"  dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_PROFILE:
        audit = audit_m92_parameter_order_dynamic_two_ended_schedule(queues)
        title = "M92 PARAMETER_ORDER DYNAMIC_TWO_ENDED schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=76,40,2x32,1x4; input_tokens=92; "
            "C2=hot-end,C3=cold-end; dynamic_physical=shape,DMA,S2PF\n"
            "  certified_model=172 ticks; "
            f"dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M92_PARAMETER_ORDER_DYNAMIC_DESC_PROFILE:
        audit = audit_m92_parameter_order_dynamic_desc_schedule(queues)
        title = "M92 PARAMETER_ORDER DYNAMIC_DESC schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=76,40,2x32,1x4; input_tokens=92; "
            "dynamic_physical=shape,DMA,S2PF,S4PF,cache\n"
            "  certified_model=168 ticks; "
            f"dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M92_PARAMETER_ORDER_STATIC_DESC_PROFILE:
        audit = audit_m92_parameter_order_static_desc_schedule(queues)
        title = "M92 PARAMETER_ORDER STATIC_DESC schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=76,40,2x32,1x4; input_tokens=92; "
            "fixed_physical=S1:B,S3:B,C2:IDMA,C3:XDMA,prefetch:off\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M70_THREE_HOT_FULL_SCHEDULER_PROFILE:
        audit = audit_m70_three_hot_full_scheduler_schedule(queues)
        title = "M70 THREE_HOT FULL_SCHEDULER schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=28x3,6x4,2x16; "
            "scheduler=order,cluster,shape,DMA,S2PF; certified_model=105 ticks\n"
            f"  dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M70_THREE_HOT_DYNAMIC_TWO_ENDED_PROFILE:
        audit = audit_m70_three_hot_dynamic_two_ended_schedule(queues)
        title = "M70 THREE_HOT DYNAMIC_TWO_ENDED schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=28x3,6x4,2x16; "
            "dynamic_physical=shape,DMA,S2PF; C2=hot-end,C3=cold-end\n"
            f"  dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile in (
        M70_THREE_HOT_DYNAMIC_DESC_PROFILE,
        M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE,
    ):
        audit = audit_m70_three_hot_dynamic_desc_schedule(queues)
        title = (
            "M70 THREE_HOT DYNAMIC_DESC skip-elided schedule:"
            if profile == M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE
            else "M70 THREE_HOT DYNAMIC_DESC schedule:"
        )
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        rejection_detail = (
            "  bingo_rejects=empty_S1_stage_tasks\n"
            if profile == M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE
            else ""
        )
        detail = (
            "  distribution=28x3,6x4,2x16; "
            "dynamic_physical=shape,DMA,S2PF,S4PF\n"
            f"{rejection_detail}"
            f"  dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == M70_THREE_HOT_STATIC_DESC_PROFILE:
        audit = audit_m70_three_hot_static_desc_schedule(queues)
        title = "M70 THREE_HOT STATIC_DESC fixed schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  distribution=28x3,6x4,2x16; "
            "fixed_physical=S1:B,S3:B,C2:IDMA,C3:XDMA,prefetch:off\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == DYNAMIC_DESC_PROFILE:
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
    elif profile == DYNAMIC_TWO_ENDED_PROFILE:
        audit = audit_dynamic_two_ended_schedule(queues)
        title = "DYNAMIC_TWO_ENDED hot/cold schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  dynamic_physical=shape,DMA,S2PF,S4PF; "
            "C2=hot-end,C3=cold-end\n"
            f"  dma_release_edges={len(audit['dma_release_edges'])}\n"
            "  structural_lower_bound=max("
            f"C2={cluster_bounds['c0'] / 4:.2f}, "
            f"C3={cluster_bounds['c1'] / 4:.2f}) = "
            f"{audit['structural_lower_bound_quarter_ticks'] / 4:.2f} ticks"
        )
    elif profile == FULL_SCHEDULER_PROFILE:
        audit = audit_full_scheduler_schedule(queues)
        title = "FULL_SCHEDULER certified schedule:"
        cluster_bounds = audit["structural_cluster_quarter_ticks"]
        detail = (
            "  scheduler=order,cluster,shape,DMA,S2PF; certified_model=129 ticks\n"
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
