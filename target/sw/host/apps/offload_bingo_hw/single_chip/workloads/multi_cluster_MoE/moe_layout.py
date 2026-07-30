"""Parameter and physical-layout derivation for the multi-cluster MoE workload.

The common section derives logical tensor dimensions from the selected SNAX
configuration. The legacy L1.5 and current bank-partition sections then add
their own physical addresses and strides without sharing placement state.
"""

L15_ALIGNMENT_BYTES = 1024
L15_BANK_WORD_BYTES = 8
L15_TOKEN_ROW_GAP_BYTES = 32
L15_B1_COLOR_BYTES = 272
L15_W2L_COLOR_BYTES = 128
L15_MODE1_D_COLOR_BYTES = 256
L15_CFG_WORDS = 91
MOE_DYNAMIC_ARG_SLOT_BYTES = 384
MOE_STATIC_ARG_SLOT_BYTES = 192
MOE_MAX_BLOCKS = 8
MOE_TOKEN_REF_MAX_TOKENS = 1 << 15
MOE_HW_TASK_MAX_TOKENS = 0x1FF

SHAPE_DIMS = (
    ("S0", 0, 8, 8, 4),
    ("S1", 1, 4, 8, 8),
    ("S2", 2, 2, 8, 16),
)


# ---------------------------------------------------------------------------
# Legacy L1.5 comparison layout
# ---------------------------------------------------------------------------


def align_up(value: int, alignment: int = L15_ALIGNMENT_BYTES) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def colored_offset(offset: int, color_bytes: int = 0) -> int:
    return align_up(offset) + int(color_bytes)


def token_row_stride(payload_bytes: int) -> int:
    stride = int(payload_bytes) + L15_TOKEN_ROW_GAP_BYTES
    if stride % L15_BANK_WORD_BYTES != 0:
        raise ValueError("L15 token row stride must be bank-word aligned")
    return stride


def build_l15_placement(
    gate_bytes: int,
    up_bytes: int,
    down_half_bytes: int,
    token_buffer_bytes: int,
    mode0_output_bytes: int,
    mode1_output_bytes: int,
):
    """Return the validated full-size L15 bank-colored tensor placement."""
    a = 0
    b0 = colored_offset(a + int(token_buffer_bytes))
    b1 = colored_offset(b0 + int(gate_bytes), L15_B1_COLOR_BYTES)
    d0 = colored_offset(b1 + int(up_bytes))
    w2l = colored_offset(d0 + int(mode0_output_bytes), L15_W2L_COLOR_BYTES)
    w2r = colored_offset(w2l + int(down_half_bytes))
    mode1_d0 = colored_offset(w2r + int(down_half_bytes), L15_MODE1_D_COLOR_BYTES)
    return {
        "delta_local_b0": b0,
        "delta_local_b1": b1,
        "delta_local_w2l": w2l,
        "delta_local_w2r": w2r,
        "delta_local_a": a,
        "delta_local_d0": d0,
        "delta_local_mode1_d0": mode1_d0,
        "tcdm_end": mode1_d0 + int(mode1_output_bytes),
    }


def validate_bank_rotation(payload_bytes: int) -> None:
    payload_words = int(payload_bytes) // L15_BANK_WORD_BYTES
    strided_words = token_row_stride(payload_bytes) // L15_BANK_WORD_BYTES
    if payload_words % 64 != 0 or strided_words % 64 == 0:
        raise ValueError("L15 row gap must rotate token rows across 64 TCDM banks")


# ---------------------------------------------------------------------------
# Layout-independent model and hardware-shape parameters
# ---------------------------------------------------------------------------


def derive_common_workload_params(config: dict) -> dict:
    """Derive layout-independent tensor dimensions and byte sizes.

    Both datagen and DFG generation call this function. The input dictionary is
    the explicit merge of params.hjson and the selected SNAX hardware cfg.
    """
    core = config["snax_dual_versacore_int16x4_core_template"]
    shape = int(config["array_shape"])
    mesh_row, tile_size, mesh_col = core["snax_acc_cfg"][0][
        "snax_versacore_spatial_unrolling"
    ][0][shape]
    expected = next(entry for entry in SHAPE_DIMS if entry[1] == shape)
    if (mesh_row, tile_size, mesh_col) != expected[2:]:
        raise ValueError("hardware shape does not match the MoE workload ABI")

    if shape != 0:
        raise ValueError("multi_cluster_MoE router/shared path requires S0")

    total_tokens = int(config["total_tokens"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    chunk_cols = int(config.get("weight_chunk_cols", 0))
    s1_block_count = int(
        config.get(
            "s1_block_count",
            intermediate_size // chunk_cols if chunk_cols else 0,
        )
    )
    s3_block_count = int(
        config.get(
            "s3_block_count",
            hidden_size // (2 * chunk_cols) if chunk_cols else 0,
        )
    )
    num_experts = int(config["num_indiv_experts"])
    num_weight_backings = int(config["num_indiv_weight_backings"])
    down_mesh_col = 2 * int(mesh_col)
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if total_tokens > MOE_TOKEN_REF_MAX_TOKENS:
        raise ValueError("packed token-reference ABI supports at most 32768 tokens")
    if total_tokens > MOE_HW_TASK_MAX_TOKENS:
        raise ValueError("HW scheduler task ABI supports at most 511 tokens")
    if hidden_size <= 0 or hidden_size % int(tile_size) != 0:
        raise ValueError("hidden_size must be divisible by S0 tileSize")
    if intermediate_size <= 0 or intermediate_size % int(tile_size) != 0:
        raise ValueError("intermediate_size must be divisible by S0 tileSize")
    if num_experts <= 0 or num_experts % (2 * int(mesh_col)) != 0:
        raise ValueError("num_indiv_experts must be divisible by 2*S0 meshCol")
    if num_weight_backings <= 0 or num_weight_backings > num_experts:
        raise ValueError(
            "num_indiv_weight_backings must be in [1, num_indiv_experts]"
        )
    if num_weight_backings & (num_weight_backings - 1):
        raise ValueError("num_indiv_weight_backings must be a power of two")
    if num_experts % num_weight_backings != 0:
        raise ValueError(
            "num_indiv_experts must be divisible by num_indiv_weight_backings"
        )
    if s1_block_count <= 0 or s3_block_count <= 0:
        raise ValueError("S1/S3 block counts must be positive")
    if s1_block_count > MOE_MAX_BLOCKS or s3_block_count > MOE_MAX_BLOCKS:
        raise ValueError(f"S1/S3 block counts must not exceed {MOE_MAX_BLOCKS}")
    if intermediate_size % (s1_block_count * int(mesh_col)) != 0:
        raise ValueError("intermediate_size must divide S1 blocks * S0 meshCol")
    if hidden_size % (s3_block_count * down_mesh_col) != 0:
        raise ValueError("hidden_size must divide S3 blocks * down_meshCol")

    # Router/shared hardware executes complete spatial rows.  A final partial
    # row is padded in the private Router input and ignored by post-processing.
    m1 = (total_tokens + int(mesh_row) - 1) // int(mesh_row)
    padded_tokens = m1 * int(mesh_row)
    input_k1 = hidden_size // int(tile_size)
    hidden_n1 = intermediate_size // (s1_block_count * int(mesh_col))
    down_k1 = intermediate_size // int(tile_size)
    down_n1 = hidden_size // (s3_block_count * down_mesh_col)
    router_n1 = num_experts // (2 * int(mesh_col))

    p = {
        "app_name": "multi_cluster_MoE",
        "array_shape": shape,
        "meshRow": int(mesh_row),
        "tileSize": int(tile_size),
        "meshCol": int(mesh_col),
        "A_meshRow": int(mesh_row),
        "A_tileSize": int(tile_size),
        "down_meshCol": down_mesh_col,
        "num_indiv_experts": num_experts,
        "num_indiv_weight_backings": num_weight_backings,
        "indiv_weight_backing_mask": num_weight_backings - 1,
        "num_shared_experts": int(config["num_shared_experts"]),
        "top_k": int(config["top_k"]),
        # A split candidate may emit one task to C2 and one task to C3 for the
        # same expert. The expert is removed after that commit, so each cluster
        # still receives at most one task per expert in one batch.
        "dynamic_slot_count": min(
            num_experts, total_tokens * int(config["top_k"])
        ),
        # CVA6 expands each RTL task into complete S1/S2/S3/S4 call records.
        # The 384B slot covers up to eight S1 and eight S3 blocks and is kept
        # 64B-aligned for the L3-to-L1 active-range transfer.
        "dynamic_arg_slot_bytes": MOE_DYNAMIC_ARG_SLOT_BYTES,
        "static_arg_slot_bytes": MOE_STATIC_ARG_SLOT_BYTES,
        "addNonZeroC": int(config["addNonZeroC"]),
        "addZeroC": int(config["addZeroC"]),
        "accumPrevC": int(config["accumPrevC"]),
        "total_tokens": total_tokens,
        "padded_tokens": padded_tokens,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "s1_block_count": s1_block_count,
        "s3_block_count": s3_block_count,
    }
    for prefix, k1, n1, n2 in (
        ("router", input_k1, router_n1, 2),
        ("indiv", input_k1, hidden_n1, s1_block_count),
        ("shared", input_k1, hidden_n1, s1_block_count),
        ("indiv_down", down_k1, down_n1, s3_block_count),
        ("shared_down", down_k1, down_n1, s3_block_count),
    ):
        p[f"{prefix}_M1"] = m1
        p[f"{prefix}_N1"] = n1
        p[f"{prefix}_K1"] = k1
        p[f"{prefix}_M2"] = 1
        p[f"{prefix}_N2"] = n2
        p[f"{prefix}_K2"] = 1

    if p["A_meshRow"] != p["meshRow"] or p["A_tileSize"] != p["tileSize"]:
        raise ValueError("this workload requires A tile shape to match hardware S0")
    if p["down_meshCol"] != 2 * p["meshCol"]:
        raise ValueError("dual-VC down output must be twice the per-VC meshCol")
    if p["top_k"] <= 0 or p["top_k"] > p["num_indiv_experts"]:
        raise ValueError("top_k must be in [1, num_indiv_experts]")
    if p["top_k"] > 2:
        raise ValueError("the direct HW Router path supports top_k=1 or top_k=2")
    if p["num_shared_experts"] != 2:
        raise ValueError("the four-cluster DFG requires exactly two shared experts")
    if p["num_indiv_experts"] > 64:
        raise ValueError("scheduler ABI supports at most 64 individual experts")
    p["M_total"] = total_tokens
    p["max_tokens_per_expert"] = p["M_total"]
    input_k = hidden_size
    if input_k % p["A_tileSize"] != 0:
        raise ValueError("input K must be divisible by A_tileSize")
    p["input_hidden"] = input_k
    p["A_token_bytes"] = input_k * 2
    p["A_token_row_stride_bytes"] = p["A_token_bytes"]
    p["A_total_row_span_bytes"] = p["M_total"] * p["A_token_row_stride_bytes"]
    p["router_A_tile_span_bytes"] = (
        p["router_M1"] * p["A_meshRow"] * p["A_token_row_stride_bytes"]
    )
    p["router_A_span_bytes"] = (
        p["router_M2"] * p["router_A_tile_span_bytes"]
    )
    p["router_output_width"] = (
        p["router_N2"] * p["router_N1"] * p["meshCol"]
    )
    p["indiv_hidden"] = p["indiv_N2"] * p["indiv_N1"] * p["meshCol"]
    p["shared_hidden"] = p["shared_N2"] * p["shared_N1"] * p["meshCol"]
    p["indiv_down_k"] = p["indiv_down_K2"] * p["indiv_down_K1"] * p["tileSize"]
    p["shared_down_k"] = p["shared_down_K2"] * p["shared_down_K1"] * p["tileSize"]
    p["indiv_output_width"] = (
        p["indiv_down_N2"] * p["indiv_down_N1"] * p["down_meshCol"]
    )
    p["shared_output_width"] = (
        p["shared_down_N2"] * p["shared_down_N1"] * p["down_meshCol"]
    )
    if p["router_output_width"] != p["num_indiv_experts"]:
        raise ValueError("router output width must equal num_indiv_experts")
    if p["indiv_down_k"] != p["indiv_hidden"]:
        raise ValueError("individual down K must equal SwiGLU hidden width")
    if p["shared_down_k"] != p["shared_hidden"]:
        raise ValueError("shared down K must equal SwiGLU hidden width")
    if p["indiv_output_width"] * 2 != p["A_token_bytes"]:
        raise ValueError("individual output width must equal input hidden width")
    if p["shared_output_width"] * 2 != p["A_token_bytes"]:
        raise ValueError("shared output width must equal input hidden width")

    p["router_B_tilesize"] = (
        p["router_K1"] * p["router_N1"] * p["tileSize"] * p["meshCol"] // 2
    )
    p["router_D_tilesize"] = (
        p["router_M1"] * p["router_N1"] * p["A_meshRow"] * p["meshCol"] * 2
    )
    p["router_B_total_bytes"] = p["router_N2"] * p["router_B_tilesize"]
    p["router_D_total_bytes"] = p["router_N2"] * p["router_D_tilesize"]
    router_n_groups = p["router_N2"] * p["router_N1"]
    if router_n_groups % 2 != 0:
        raise ValueError("dual-VC router requires an even number of N groups")
    p["router_vc_N"] = router_n_groups // 2
    p["router_B_vc_stride"] = (
        p["router_vc_N"] * p["router_K2"] * p["router_K1"]
        * p["tileSize"] * p["meshCol"] // 2
    )
    p["router_D_vc_stride"] = (
        p["router_vc_N"] * p["router_M1"] * p["A_meshRow"]
        * p["meshCol"] * 2
    )

    p["indiv_B_tilesize"] = (
        p["indiv_K1"] * p["indiv_N1"] * p["tileSize"] * p["meshCol"] // 2
    )
    p["indiv_D_tilesize"] = (
        p["indiv_M1"] * p["indiv_N1"] * p["A_meshRow"] * p["meshCol"] * 2
    )
    p["indiv_down_B_tilesize"] = (
        p["indiv_down_K1"] * p["indiv_down_N1"]
        * p["tileSize"] * p["meshCol"] // 2
    )
    p["indiv_down_D_tilesize"] = (
        p["indiv_down_M1"] * p["indiv_down_N1"]
        * p["A_meshRow"] * p["meshCol"] * 2
    )
    p["indiv_B_block_stride"] = p["indiv_B_tilesize"]
    p["indiv_down_B_block_stride"] = p["indiv_down_B_tilesize"]
    p["indiv_B_expert_stride"] = (
        p["indiv_N2"] * p["indiv_B_block_stride"]
    )
    p["indiv_down_B_expert_stride"] = (
        2 * p["indiv_down_N2"] * p["indiv_down_B_block_stride"]
    )
    p["shared_B_expert_stride"] = (
        p["shared_N2"] * p["shared_K2"] * p["shared_K1"]
        * p["shared_N1"] * p["tileSize"] * p["meshCol"] // 2
    )
    p["shared_down_B_expert_stride"] = (
        2 * p["shared_down_N2"] * p["shared_down_K2"] * p["shared_down_K1"]
        * p["shared_down_N1"] * p["tileSize"] * p["meshCol"] // 2
    )

    return p


def derive_l15_workload_params(config: dict) -> dict:
    """Add the legacy padded-row L1.5 placement for comparison experiments."""
    p = derive_common_workload_params(config)
    p["A_token_row_stride_bytes"] = token_row_stride(p["A_token_bytes"])
    p["A_total_row_span_bytes"] = p["M_total"] * p["A_token_row_stride_bytes"]
    p["router_A_tile_span_bytes"] = (
        p["router_M1"] * p["A_meshRow"] * p["A_token_row_stride_bytes"]
    )
    p["router_A_span_bytes"] = p["router_M2"] * p["router_A_tile_span_bytes"]
    validate_bank_rotation(p["A_token_bytes"])

    # The full-size L1.5 reference stores A first and uses S0 weight order.
    input_k = p["input_hidden"]
    n0_total = p["shared_hidden"]
    n1_total = p["shared_output_width"]
    if n1_total % 2 != 0:
        raise ValueError("dual-VC output width must split evenly")
    n1_per_vc = n1_total // 2
    p["l15_a_row_stride"] = p["A_token_row_stride_bytes"]
    p["l15_a_span_bytes"] = p["A_total_row_span_bytes"]
    p["l15_b_data_length"] = (input_k // 8) * (n0_total // 4) * 16
    p["l15_w2_data_length"] = (n0_total // 8) * (n1_per_vc // 4) * 16
    p["l15_mode0_output_bytes"] = p["M_total"] * n0_total * 2
    p["l15_mode0_row_bytes"] = n0_total * 2
    p["l15_mode1_payload_bytes_per_row"] = n1_total * 2
    p["l15_mode1_output_span_bytes"] = p["A_total_row_span_bytes"]
    if p["shared_B_expert_stride"] != p["l15_b_data_length"]:
        raise ValueError("shared gate/up weights do not match L15 order")
    if p["shared_down_B_expert_stride"] != 2 * p["l15_w2_data_length"]:
        raise ValueError("shared down weights do not match L15 order")
    if p["shared_down_M1"] * p["meshRow"] != p["M_total"]:
        raise ValueError("shared_down_M1 must cover all tokens")

    placement = build_l15_placement(
        gate_bytes=p["l15_b_data_length"],
        up_bytes=p["l15_b_data_length"],
        down_half_bytes=p["l15_w2_data_length"],
        token_buffer_bytes=p["l15_a_span_bytes"],
        mode0_output_bytes=p["l15_mode0_output_bytes"],
        mode1_output_bytes=p["l15_mode1_output_span_bytes"],
    )
    for field, value in placement.items():
        if field != "tcdm_end":
            p[f"l15_{field}"] = value
    p["l15_delta_local_mode1_d1"] = (
        p["l15_delta_local_mode1_d0"] + n1_per_vc * 2
    )
    p["l15_delta_cfg"] = placement["tcdm_end"]
    p["l15_cfg_bytes"] = L15_CFG_WORDS * 4
    p["l15_tcdm_size"] = p["l15_delta_cfg"] + p["l15_cfg_bytes"]
    return p


# ---------------------------------------------------------------------------
# Current conflict-free bank-partition layout
# ---------------------------------------------------------------------------


BANK_TCDM_ROW_BYTES = 512
BANK_TCDM_CAPACITY_BYTES = 8 * 1024 * 1024
BANK_WORD_BYTES = 8
BANK_WEIGHT_ROW_BYTES = 64
BANK_A_BASE = 0 * BANK_WORD_BYTES
BANK_B0_PING_BASE = 16 * BANK_WORD_BYTES
BANK_B1_PING_BASE = 24 * BANK_WORD_BYTES
BANK_B0_PONG_BASE = 32 * BANK_WORD_BYTES
BANK_B1_PONG_BASE = 40 * BANK_WORD_BYTES
BANK_MODE0_D_BASE = 48 * BANK_WORD_BYTES
BANK_MODE1_D0_BASE = 0 * BANK_WORD_BYTES
BANK_MODE1_D1_BASE = 8 * BANK_WORD_BYTES


def derive_bank_workload_params(config: dict) -> dict:
    """Derive the conflict-free 16-bank A/B0/B1/D workload layout.

    L3 tensors remain dense and canonical.  Only their L1 representation is
    expanded over 512-byte TCDM rows.  A logical DFG block is also one physical
    weight chunk; changing model dimensions therefore requires changing only
    params.hjson, while all DMA and streamer spans are recomputed here.
    """
    p = derive_common_workload_params(config)
    mode0_chunk_cols = p["intermediate_size"] // p["s1_block_count"]
    mode1_chunk_cols = (p["hidden_size"] // 2) // p["s3_block_count"]
    requested_mode0_chunk_cols = int(
        config.get("s1_weight_chunk_cols", config["weight_chunk_cols"])
    )
    requested_mode1_chunk_cols = int(
        config.get("s3_weight_chunk_cols", config["weight_chunk_cols"])
    )
    if (
        requested_mode0_chunk_cols <= 0
        or requested_mode0_chunk_cols % 16 != 0
        or requested_mode1_chunk_cols <= 0
        or requested_mode1_chunk_cols % 16 != 0
    ):
        raise ValueError("S1/S3 weight chunks must be positive multiples of 16")
    if mode0_chunk_cols != requested_mode0_chunk_cols:
        raise ValueError(
            "intermediate_size / s1_block_count must equal s1_weight_chunk_cols"
        )
    if mode1_chunk_cols != requested_mode1_chunk_cols:
        raise ValueError(
            "hidden_size / 2 / s3_block_count must equal s3_weight_chunk_cols"
        )

    # L3 token rows are dense.  The 2D gather maps each 16-byte K tile to the
    # token's two-bank pair and advances one complete 512-byte TCDM row.
    p["A_token_row_stride_bytes"] = p["A_token_bytes"]
    p["A_total_row_span_bytes"] = p["M_total"] * p["A_token_bytes"]
    p["router_A_tile_span_bytes"] = (
        p["router_M1"] * p["A_meshRow"] * p["A_token_bytes"]
    )
    p["router_A_span_bytes"] = p["router_M2"] * p["router_A_tile_span_bytes"]
    p["router_legacy_A_row_stride_bytes"] = (
        p["A_token_bytes"] + L15_TOKEN_ROW_GAP_BYTES
    )
    p["router_legacy_A_span_bytes"] = (
        p["padded_tokens"] * p["router_legacy_A_row_stride_bytes"]
    )

    mode0_panel_payload = p["hidden_size"] * 4 // 2
    mode1_panel_payload = p["intermediate_size"] * 4 // 2
    if mode0_panel_payload % BANK_WEIGHT_ROW_BYTES != 0:
        raise ValueError("Mode0 4-column panel must divide into 64-byte DMA rows")
    if mode1_panel_payload % BANK_WEIGHT_ROW_BYTES != 0:
        raise ValueError("Mode1 4-column panel must divide into 64-byte DMA rows")

    mode0_panel_span = (
        mode0_panel_payload // BANK_WEIGHT_ROW_BYTES * BANK_TCDM_ROW_BYTES
    )
    mode1_panel_span = (
        mode1_panel_payload // BANK_WEIGHT_ROW_BYTES * BANK_TCDM_ROW_BYTES
    )
    mode0_panels_per_chunk = mode0_chunk_cols // 4
    mode1_panels_per_chunk = mode1_chunk_cols // 4
    mode0_chunk_span = mode0_panels_per_chunk * mode0_panel_span
    mode1_chunk_span = mode1_panels_per_chunk * mode1_panel_span
    mode0_slots = (p["s1_block_count"] + 1) // 2
    mode1_slots = (p["s3_block_count"] + 1) // 2
    mode0_region_span = mode0_slots * mode0_chunk_span
    mode1_region_offset = mode0_region_span
    mode1_region_span = mode1_slots * mode1_chunk_span
    weight_region_end = mode1_region_offset + mode1_region_span

    # Eight local tokens occupy banks 0..15 in one row-depth page.  Additional
    # tokens reuse those banks in later pages; every task gathers its complete
    # token slice once and compute stages address pages by local-token index.
    token_lanes = 8
    token_pages = (p["M_total"] + token_lanes - 1) // token_lanes
    token_page_span = (
        p["A_token_bytes"] // (2 * token_lanes) * BANK_TCDM_ROW_BYTES
    )
    mode0_output_page_span = (
        p["s1_block_count"]
        * (mode0_chunk_cols // token_lanes)
        * BANK_TCDM_ROW_BYTES
    )
    mode1_output_page_span = (
        p["s3_block_count"] * (mode1_chunk_cols // 4) * BANK_TCDM_ROW_BYTES
    )
    if token_page_span != mode1_output_page_span:
        raise ValueError("A and Mode1 output page spans must match")

    token_region_end = token_pages * token_page_span
    mode0_output_region_end = token_pages * mode0_output_page_span
    resident_depth_end = max(
        weight_region_end,
        token_region_end,
        mode0_output_region_end,
    )
    # Every physical base selects banks within one 512-byte TCDM row. Slot
    # parity may place a full token/output page at the highest bank group, so
    # retain the complete final row instead of allocating only its depth span.
    resident_end = resident_depth_end + BANK_TCDM_ROW_BYTES
    if resident_end > BANK_TCDM_CAPACITY_BYTES:
        raise ValueError(
            f"bank-partition resident layout needs {resident_end} bytes, "
            f"capacity is {BANK_TCDM_CAPACITY_BYTES}"
        )

    p.update(
        {
            "weight_chunk_cols": mode0_chunk_cols,
            "s1_weight_chunk_cols": mode0_chunk_cols,
            "s3_weight_chunk_cols": mode1_chunk_cols,
            "bank_tcdm_row_bytes": BANK_TCDM_ROW_BYTES,
            "bank_tcdm_capacity_bytes": BANK_TCDM_CAPACITY_BYTES,
            "bank_weight_row_bytes": BANK_WEIGHT_ROW_BYTES,
            "bank_weight_dst_stride": BANK_TCDM_ROW_BYTES,
            "bank_weight_depth_scale": BANK_TCDM_ROW_BYTES
            // BANK_WEIGHT_ROW_BYTES,
            "bank_delta_local_a": BANK_A_BASE,
            "bank_delta_local_b0": BANK_B0_PING_BASE,
            "bank_delta_local_b1": BANK_B1_PING_BASE,
            "bank_delta_local_b0_pong": BANK_B0_PONG_BASE,
            "bank_delta_local_b1_pong": BANK_B1_PONG_BASE,
            "bank_delta_local_d0": BANK_MODE0_D_BASE,
            "bank_delta_local_mode1_d0": BANK_MODE1_D0_BASE,
            "bank_delta_local_mode1_d1": BANK_MODE1_D1_BASE,
            "bank_mode0_panel_span": mode0_panel_span,
            "bank_mode1_panel_span": mode1_panel_span,
            "bank_mode0_chunk_span": mode0_chunk_span,
            "bank_mode1_chunk_span": mode1_chunk_span,
            "bank_mode0_region_span": mode0_region_span,
            "bank_mode1_region_offset": mode1_region_offset,
            "bank_mode1_region_span": mode1_region_span,
            "bank_tcdm_size": resident_end,
            "bank_token_lanes": token_lanes,
            "bank_token_pages": token_pages,
            "bank_token_page_span": token_page_span,
            "bank_token_dma_row_bytes": 16,
            "bank_token_dma_repeats": p["hidden_size"] // 8,
            "bank_mode0_output_block_span": (mode0_chunk_cols // 8)
            * BANK_TCDM_ROW_BYTES,
            "bank_mode0_output_page_span": mode0_output_page_span,
            "bank_mode1_output_block_span": (mode1_chunk_cols // 4)
            * BANK_TCDM_ROW_BYTES,
            "bank_mode1_output_page_span": mode1_output_page_span,
            "bank_mode1_half_output_bytes": p["hidden_size"],
            "bank_mode1_output_span_bytes": p["M_total"] * p["A_token_bytes"],
        }
    )

    if p["indiv_B_block_stride"] * p["bank_weight_depth_scale"] != mode0_chunk_span:
        raise ValueError("Mode0 logical block and physical chunk spans disagree")
    if (
        p["indiv_down_B_block_stride"] * p["bank_weight_depth_scale"]
        != mode1_chunk_span
    ):
        raise ValueError("Mode1 logical block and physical chunk spans disagree")
    return p


def derive_mixed_workload_params(config: dict) -> dict:
    """Derive bank-aware individual and legacy L1.5 shared layouts together.

    The bank derivation remains authoritative for common/runtime fields and the
    individual-expert ABI. Only the ``l15_*`` placement fields are imported
    from the legacy derivation, so its padded token stride cannot leak into the
    dense individual path.
    """
    p = derive_bank_workload_params(config)
    l15 = derive_l15_workload_params(config)
    p.update({key: value for key, value in l15.items() if key.startswith("l15_")})
    return p
