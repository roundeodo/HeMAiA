"""Canonical L15 physical layout for the multi-cluster MoE workload.

The hardware shape family is fixed by the selected SNAX configuration. Model
dimensions and token count remain parameters. Both datagen and DFG generation
must use this module so bank coloring and row strides cannot drift apart.
"""

L15_ALIGNMENT_BYTES = 1024
L15_BANK_WORD_BYTES = 8
L15_TOKEN_PADDING_BYTES = 32
L15_B1_COLOR_BYTES = 272
L15_W2L_COLOR_BYTES = 128
L15_MODE1_D_COLOR_BYTES = 256
L15_CFG_WORDS = 91

SHAPE_DIMS = (
    ("S0", 0, 8, 8, 4),
    ("S1", 1, 4, 8, 8),
    ("S2", 2, 2, 8, 16),
)


def align_up(value: int, alignment: int = L15_ALIGNMENT_BYTES) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def colored_offset(offset: int, color_bytes: int = 0) -> int:
    return align_up(offset) + int(color_bytes)


def token_row_stride(payload_bytes: int) -> int:
    stride = int(payload_bytes) + L15_TOKEN_PADDING_BYTES
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
    padded_words = token_row_stride(payload_bytes) // L15_BANK_WORD_BYTES
    if payload_words % 64 != 0 or padded_words % 64 == 0:
        raise ValueError("L15 padding must rotate token rows across 64 TCDM banks")


def derive_workload_params(config: dict) -> dict:
    """Derive every tensor dimension, byte size and L15 offset once.

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
        raise ValueError("hardware shape does not match the L15 ABI")

    if shape != 0:
        raise ValueError("multi_cluster_MoE router/shared path requires S0")

    total_tokens = int(config["total_tokens"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_experts = int(config["num_indiv_experts"])
    down_mesh_col = 2 * int(mesh_col)
    if total_tokens <= 0 or total_tokens % int(mesh_row) != 0:
        raise ValueError("total_tokens must be a positive multiple of S0 meshRow")
    if hidden_size <= 0 or hidden_size % int(tile_size) != 0:
        raise ValueError("hidden_size must be divisible by S0 tileSize")
    if intermediate_size <= 0 or intermediate_size % int(tile_size) != 0:
        raise ValueError("intermediate_size must be divisible by S0 tileSize")
    if num_experts <= 0 or num_experts % (2 * int(mesh_col)) != 0:
        raise ValueError("num_indiv_experts must be divisible by 2*S0 meshCol")
    if intermediate_size % (2 * int(mesh_col)) != 0:
        raise ValueError("intermediate_size must be divisible by 2*S0 meshCol")
    if hidden_size % (2 * down_mesh_col) != 0:
        raise ValueError("hidden_size must be divisible by 2*down_meshCol")

    m1 = total_tokens // int(mesh_row)
    input_k1 = hidden_size // int(tile_size)
    hidden_n1 = intermediate_size // (2 * int(mesh_col))
    down_k1 = intermediate_size // int(tile_size)
    down_n1 = hidden_size // (2 * down_mesh_col)
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
        "num_shared_experts": int(config["num_shared_experts"]),
        "top_k": int(config["top_k"]),
        "dynamic_slot_count": int(config["moe_dynamic_slot_count"]),
        "dynamic_arg_slot_bytes": 192,
        "addNonZeroC": int(config["addNonZeroC"]),
        "addZeroC": int(config["addZeroC"]),
        "accumPrevC": int(config["accumPrevC"]),
        "total_tokens": total_tokens,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }
    for prefix, k1, n1 in (
        ("router", input_k1, router_n1),
        ("indiv", input_k1, hidden_n1),
        ("shared", input_k1, hidden_n1),
        ("indiv_down", down_k1, down_n1),
        ("shared_down", down_k1, down_n1),
    ):
        p[f"{prefix}_M1"] = m1
        p[f"{prefix}_N1"] = n1
        p[f"{prefix}_K1"] = k1
        p[f"{prefix}_M2"] = 1
        p[f"{prefix}_N2"] = 2
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
    if p["dynamic_slot_count"] < p["num_indiv_experts"]:
        raise ValueError("moe_dynamic_slot_count must cover all individual experts")

    p["M_total"] = total_tokens
    p["max_tokens_per_expert"] = p["M_total"]
    input_k = hidden_size
    if input_k % p["A_tileSize"] != 0:
        raise ValueError("input K must be divisible by A_tileSize")
    p["input_hidden"] = input_k
    p["A_token_bytes"] = input_k * 2
    p["A_token_padded_bytes"] = token_row_stride(p["A_token_bytes"])
    p["A_total_padded_bytes"] = p["M_total"] * p["A_token_padded_bytes"]
    p["router_A_tile_padded_bytes"] = (
        p["router_M1"] * p["A_meshRow"] * p["A_token_padded_bytes"]
    )
    p["router_A_padded_bytes"] = (
        p["router_M2"] * p["router_A_tile_padded_bytes"]
    )
    validate_bank_rotation(p["A_token_bytes"])

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

    # The full-size L15 reference stores A first and uses S0 physical weight order.
    n0_total = p["shared_hidden"]
    n1_total = p["shared_output_width"]
    if n1_total % 2 != 0:
        raise ValueError("dual-VC output width must split evenly")
    n1_per_vc = n1_total // 2
    p["l15_a_row_stride"] = p["A_token_padded_bytes"]
    p["l15_a_data_bytes"] = p["A_total_padded_bytes"]
    p["l15_b_data_length"] = (input_k // 8) * (n0_total // 4) * 16
    p["l15_w2_data_length"] = (n0_total // 8) * (n1_per_vc // 4) * 16
    p["l15_mode0_output_bytes"] = p["M_total"] * n0_total * 2
    p["l15_mode0_row_bytes"] = n0_total * 2
    p["l15_mode1_payload_bytes_per_row"] = n1_total * 2
    p["l15_mode1_padded_bytes"] = p["A_total_padded_bytes"]
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
        token_buffer_bytes=p["l15_a_data_bytes"],
        mode0_output_bytes=p["l15_mode0_output_bytes"],
        mode1_output_bytes=p["l15_mode1_padded_bytes"],
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
