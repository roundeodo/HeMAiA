"""Dimensions and byte layout for the single-slot static MoE pattern test."""

TOKEN_PADDING_BYTES = 32
S1_SHAPE = 1
S2_SHAPE = 2
BLOCK_COUNT = 2


def derive_params(config: dict) -> dict:
    total_tokens = int(config["total_tokens"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])

    core = config["snax_dual_versacore_int16x4_core_template"]
    shapes = core["snax_acc_cfg"][0]["snax_versacore_spatial_unrolling"][0]
    if tuple(shapes[S1_SHAPE]) != (4, 8, 8):
        raise ValueError("S1 hardware shape must be (4, 8, 8) per VersaCore")
    if tuple(shapes[S2_SHAPE]) != (2, 8, 16):
        raise ValueError("S2 hardware shape must be (2, 8, 16) per VersaCore")
    if total_tokens != 8:
        raise ValueError("multi_cluster_MoE_test intentionally fixes one slot to 8 tokens")
    if hidden_size % 16 != 0 or intermediate_size % 16 != 0:
        raise ValueError("hidden_size and intermediate_size must be divisible by 16")

    s1_rows = 4
    tail_rows = total_tokens - s1_rows
    if tail_rows <= 0 or tail_rows % 2 != 0:
        raise ValueError("S2/S4 tail must contain a positive even number of rows")

    gate_block_output = intermediate_size // BLOCK_COUNT
    down_block_output = hidden_size // BLOCK_COUNT
    down_block_per_vc = down_block_output // 2
    if gate_block_output % 8 != 0 or down_block_per_vc % 8 != 0:
        raise ValueError("block outputs must be divisible by S1 meshCol=8")

    token_payload_bytes = hidden_size * 2
    token_row_stride = token_payload_bytes + TOKEN_PADDING_BYTES
    gate_block_row_bytes = gate_block_output * 2
    gate_full_row_bytes = intermediate_size * 2
    down_block_vc_row_bytes = down_block_per_vc * 2
    down_vc_row_bytes = hidden_size

    gate_block_bytes = hidden_size * gate_block_output // 2
    down_block_bytes = intermediate_size * down_block_per_vc // 2
    dma_probe_bytes = BLOCK_COUNT * gate_block_bytes
    if dma_probe_bytes % 512 != 0:
        raise ValueError("DMA probe size must preserve a 512-byte bank phase")

    return {
        "app_name": "multi_cluster_MoE_test",
        "total_tokens": total_tokens,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "token_payload_bytes": token_payload_bytes,
        "token_row_stride": token_row_stride,
        "token_buffer_bytes": total_tokens * token_row_stride,
        "s1_shape": S1_SHAPE,
        "s2_shape": S2_SHAPE,
        "s1_rows": s1_rows,
        "tail_rows": tail_rows,
        "s1_M": 1,
        "s2_M": tail_rows // 2,
        "gate_K": hidden_size // 8,
        "gate_N_s1": gate_block_output // 8,
        "gate_N_s2": intermediate_size // 16,
        "down_K": intermediate_size // 8,
        "down_N_s3_full": (hidden_size // 2) // 8,
        "down_N_s4": (hidden_size // 2) // 16,
        "block_count": BLOCK_COUNT,
        "gate_block_bytes": gate_block_bytes,
        "gate_weight_bytes": BLOCK_COUNT * gate_block_bytes,
        "dma_probe_bytes": dma_probe_bytes,
        "down_block_bytes": down_block_bytes,
        "down_weight_bytes": 2 * BLOCK_COUNT * down_block_bytes,
        "gate_block_row_bytes": gate_block_row_bytes,
        "gate_full_row_bytes": gate_full_row_bytes,
        "gate_output_bytes": total_tokens * gate_full_row_bytes,
        "gate_scratch_bytes": tail_rows * gate_full_row_bytes,
        "down_block_vc_row_bytes": down_block_vc_row_bytes,
        "down_vc_row_bytes": down_vc_row_bytes,
        "output_bytes": total_tokens * token_row_stride,
    }
