"""Dimensions and aliases for the production-layout MoE integration test."""

import pathlib
import sys


PRODUCTION_WORKLOAD_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "multi_cluster_MoE"
)
sys.path.insert(0, str(PRODUCTION_WORKLOAD_DIR))
from moe_layout import derive_bank_workload_params  # noqa: E402
from moe_test_schedule import (  # noqa: E402
    BASELINE_PROFILE,
    C_TAIL_SMOKE_PROFILE,
    DYNAMIC_DESC_PROFILE,
    DYNAMIC_TWO_ENDED_PROFILE,
    ENDS_INWARD_PROFILE,
    FULL_SCHEDULER_PROFILE,
    EXPERT_COUNT,
    HIGH_TO_LOW_PROFILE,
    HIGH_TO_LOW_COUNTS,
    LOW_TO_HIGH_PROFILE,
    M60_HIGH_SKEW_COUNTS,
    M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE,
    M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE,
    M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE,
    M60_HIGH_SKEW_STATIC_DESC_PROFILE,
    M70_THREE_HOT_COUNTS,
    M70_THREE_HOT_DYNAMIC_DESC_PROFILE,
    M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE,
    M70_THREE_HOT_DYNAMIC_TWO_ENDED_PROFILE,
    M70_THREE_HOT_FULL_SCHEDULER_PROFILE,
    M70_THREE_HOT_STATIC_DESC_PROFILE,
    M92_PARAMETER_ORDER_COUNTS,
    M92_PARAMETER_ORDER_DYNAMIC_DESC_PROFILE,
    M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_PROFILE,
    M92_PARAMETER_ORDER_FULL_SCHEDULER_PROFILE,
    M92_PARAMETER_ORDER_STATIC_DESC_PROFILE,
    S2PF_BOTH_PROFILE,
    S1_STAGE_SMOKE_PROFILE,
    SCHEDULE_PROFILES,
    STATIC_DESC_PROFILE,
    build_schedule_profile,
)


S1_SHAPE = 1
S2_SHAPE = 2
S3_SHAPE = 1
S4_SHAPE = 2
SLOT_COUNT = 2
SLOT0_TOKENS = 11
SLOT1_TOKENS = 6
GATHER_SOURCE_TOKENS = 32

SHAPE_ROWS = (8, 4, 2)
SHAPE_COLS = (4, 8, 16)


def derive_params(config: dict, schedule_profile: str = BASELINE_PROFILE) -> dict:
    """Reuse the large workload's bank layout and expose test-friendly names."""
    if schedule_profile not in SCHEDULE_PROFILES:
        raise ValueError(
            f"unknown schedule profile {schedule_profile!r}; "
            f"expected one of {SCHEDULE_PROFILES}"
        )
    p = derive_bank_workload_params(config)
    if p["hidden_size"] != 2048 or p["intermediate_size"] != 1024:
        raise ValueError("production path test requires the 2048x1024 MoE dimensions")
    if p["s1_weight_chunk_cols"] != 128 or p["s1_block_count"] != 8:
        raise ValueError("S4 test requires eight 128-column gate/up blocks")
    if p["s3_weight_chunk_cols"] != 256 or p["s3_block_count"] != 4:
        raise ValueError("S4 test requires four 256-column down blocks per VC")

    gate_block_output = p["intermediate_size"] // p["s1_block_count"]
    down_block_per_vc = (p["hidden_size"] // 2) // p["s3_block_count"]
    token_bytes = p["A_token_bytes"]

    clear_prod_output = schedule_profile != S2PF_BOTH_PROFILE
    if schedule_profile in (
        HIGH_TO_LOW_PROFILE,
        LOW_TO_HIGH_PROFILE,
        ENDS_INWARD_PROFILE,
        STATIC_DESC_PROFILE,
        DYNAMIC_DESC_PROFILE,
        DYNAMIC_TWO_ENDED_PROFILE,
        FULL_SCHEDULER_PROFILE,
        M70_THREE_HOT_STATIC_DESC_PROFILE,
        M70_THREE_HOT_DYNAMIC_DESC_PROFILE,
        M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE,
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
    ):
        queues = build_schedule_profile(schedule_profile)
        distribution = (
            M60_HIGH_SKEW_COUNTS
            if schedule_profile in (
                M60_HIGH_SKEW_STATIC_DESC_PROFILE,
                M60_HIGH_SKEW_DYNAMIC_DESC_PROFILE,
                M60_HIGH_SKEW_DYNAMIC_TWO_ENDED_PROFILE,
                M60_HIGH_SKEW_FULL_SCHEDULER_PROFILE,
            )
            else (
                M70_THREE_HOT_COUNTS
                if schedule_profile in (
                    M70_THREE_HOT_STATIC_DESC_PROFILE,
                    M70_THREE_HOT_DYNAMIC_DESC_PROFILE,
                    M70_THREE_HOT_DYNAMIC_DESC_SKIP_ELIDED_PROFILE,
                    M70_THREE_HOT_DYNAMIC_TWO_ENDED_PROFILE,
                    M70_THREE_HOT_FULL_SCHEDULER_PROFILE,
                )
                else (
                    M92_PARAMETER_ORDER_COUNTS
                    if schedule_profile in (
                        M92_PARAMETER_ORDER_STATIC_DESC_PROFILE,
                        M92_PARAMETER_ORDER_DYNAMIC_DESC_PROFILE,
                        M92_PARAMETER_ORDER_DYNAMIC_TWO_ENDED_PROFILE,
                        M92_PARAMETER_ORDER_FULL_SCHEDULER_PROFILE,
                    )
                    else HIGH_TO_LOW_COUNTS
                )
            )
        )
        gather_source_tokens = sum(distribution) // 2
        max_tokens_per_expert = max(distribution)
        slot_count = max(len(slots) for slots in queues.values())
        token_ref_count = EXPERT_COUNT * max_tokens_per_expert
        output_expert_stride = max_tokens_per_expert * token_bytes
        output_bytes = EXPERT_COUNT * output_expert_stride
    elif schedule_profile in (C_TAIL_SMOKE_PROFILE, S1_STAGE_SMOKE_PROFILE):
        queues = build_schedule_profile(schedule_profile)
        active_slots = tuple(slot for slots in queues.values() for slot in slots)
        gather_source_tokens = sum(slot.ntokens for slot in active_slots)
        max_tokens_per_expert = max(slot.ntokens for slot in active_slots)
        slot_count = max(len(slots) for slots in queues.values())
        max_expert_id = max(slot.expert_id for slot in active_slots)
        token_ref_count = (max_expert_id + 1) * max_tokens_per_expert
        output_expert_stride = max_tokens_per_expert * token_bytes
        # Both smoke profiles write every byte that their checks consume. Avoid
        # a Host-side clear of the sparse logical-expert output range so the
        # diagnostic starts directly with the two-slot device path.
        clear_prod_output = False
        if schedule_profile == S1_STAGE_SMOKE_PROFILE:
            # S1 writes only to L1. Keep one valid output placeholder.
            output_bytes = max_tokens_per_expert * token_bytes
        else:
            output_bytes = (max_expert_id + 1) * output_expert_stride
    else:
        gather_source_tokens = GATHER_SOURCE_TOKENS
        max_tokens_per_expert = SLOT0_TOKENS + SLOT1_TOKENS
        slot_count = SLOT_COUNT
        token_ref_count = SLOT0_TOKENS + SLOT1_TOKENS
        output_expert_stride = token_ref_count * token_bytes
        output_bytes = output_expert_stride

    p.update(
        {
            "app_name": "multi_cluster_MoE_test",
            "schedule_profile": schedule_profile,
            "gather_source_tokens": gather_source_tokens,
            "prod_slot_count": slot_count,
            "prod_slot_tokens": SLOT0_TOKENS,
            "prod_slot0_tokens": SLOT0_TOKENS,
            "prod_slot1_tokens": SLOT1_TOKENS,
            "prod_max_tokens_per_expert": max_tokens_per_expert,
            "prod_token_refs_bytes": token_ref_count * 2,
            "prod_output_expert_stride_bytes": output_expert_stride,
            "prod_output_bytes": output_bytes,
            "prod_clear_output": clear_prod_output,
            "token_payload_bytes": token_bytes,
            "token_row_stride": p["A_token_row_stride_bytes"],
            "token_buffer_bytes": p["total_tokens"] * token_bytes,
            "s1_shape": S1_SHAPE,
            "s2_shape": S2_SHAPE,
            "s3_shape": S3_SHAPE,
            "s4_shape": S4_SHAPE,
            "base_mesh_row": p["meshRow"],
            "base_mesh_col": p["meshCol"],
            "s1_rows": SHAPE_ROWS[S1_SHAPE],
            "s2_rows": SHAPE_ROWS[S2_SHAPE],
            "s3_rows": SHAPE_ROWS[S3_SHAPE],
            "s4_rows": SHAPE_ROWS[S4_SHAPE],
            "gate_K": p["indiv_K1"],
            "gate_N_s1": gate_block_output // SHAPE_COLS[S1_SHAPE],
            "gate_N_s2": p["intermediate_size"] // SHAPE_COLS[S2_SHAPE],
            "down_K": p["indiv_down_K1"],
            "down_N_s3_block": down_block_per_vc // SHAPE_COLS[S3_SHAPE],
            "down_N_s4": (p["hidden_size"] // 2) // SHAPE_COLS[S4_SHAPE],
            "indiv_N_per_block": gate_block_output,
            "indiv_down_N_per_block": down_block_per_vc,
            "gate_block_bytes": p["indiv_B_block_stride"],
            "gate_weight_bytes": p["indiv_B_expert_stride"],
            "down_block_bytes": p["indiv_down_B_block_stride"],
            "down_weight_bytes": p["indiv_down_B_expert_stride"],
        }
    )
    return p
