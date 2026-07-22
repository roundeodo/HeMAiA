"""Dimensions and aliases for the production-layout MoE integration test."""

import pathlib
import sys


PRODUCTION_WORKLOAD_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "multi_cluster_MoE"
)
sys.path.insert(0, str(PRODUCTION_WORKLOAD_DIR))
from moe_layout import derive_bank_workload_params  # noqa: E402


S1_SHAPE = 1
S2_SHAPE = 2
SLOT_COUNT = 2
SLOT_TOKENS = 7
GATHER_SOURCE_TOKENS = 32


def derive_params(config: dict) -> dict:
    """Reuse the large workload's bank layout and expose test-friendly names."""
    p = derive_bank_workload_params(config)
    if p["hidden_size"] != 2048 or p["intermediate_size"] != 1024:
        raise ValueError("production path test requires the 2048x1024 MoE dimensions")
    if p["weight_chunk_cols"] != 128:
        raise ValueError("production path test requires 128-column weight chunks")
    if p["s1_block_count"] != 8 or p["s3_block_count"] != 8:
        raise ValueError("production path test requires eight S1 and S3 blocks")

    gate_block_output = p["intermediate_size"] // p["s1_block_count"]
    down_block_per_vc = (p["hidden_size"] // 2) // p["s3_block_count"]
    token_bytes = p["A_token_bytes"]

    p.update(
        {
            "app_name": "multi_cluster_MoE_test",
            "gather_source_tokens": GATHER_SOURCE_TOKENS,
            "prod_slot_count": SLOT_COUNT,
            "prod_slot_tokens": SLOT_TOKENS,
            "prod_token_refs_bytes": SLOT_COUNT * SLOT_TOKENS * 2,
            "prod_output_bytes": SLOT_COUNT * SLOT_TOKENS * token_bytes,
            "token_payload_bytes": token_bytes,
            "token_row_stride": p["A_token_row_stride_bytes"],
            "token_buffer_bytes": p["total_tokens"] * token_bytes,
            "s1_shape": S1_SHAPE,
            "s2_shape": S2_SHAPE,
            "base_mesh_row": p["meshRow"],
            "base_mesh_col": p["meshCol"],
            "s1_rows": 4,
            "s1_M": 1,
            "s2_M": 2,
            "gate_K": p["indiv_K1"],
            "gate_N_s1": gate_block_output >> (S1_SHAPE + 2),
            "gate_N_s2": p["intermediate_size"] // 16,
            "down_K": p["indiv_down_K1"],
            "down_N_s3_full": (p["hidden_size"] // 2) // 8,
            "down_N_s3_block": down_block_per_vc >> (S1_SHAPE + 2),
            "down_N_s4": (p["hidden_size"] // 2) // 16,
            "indiv_N_per_block": gate_block_output,
            "indiv_down_N_per_block": down_block_per_vc,
            "block_count": p["s1_block_count"],
            "gate_block_bytes": p["indiv_B_block_stride"],
            "gate_weight_bytes": p["indiv_B_expert_stride"],
            "down_block_bytes": p["indiv_down_B_block_stride"],
            "down_weight_bytes": p["indiv_down_B_expert_stride"],
        }
    )
    return p
