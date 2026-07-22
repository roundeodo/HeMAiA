"""Shared production DFG builder for one dynamic individual-expert slot."""


def build_dynamic_expert_slot_chain(
    *,
    add_node,
    add_edge,
    make_block_args,
    input_ready,
    s1_block_count,
    s3_block_count,
    dma_core_id,
    gemm_core_id,
    label_prefix="",
):
    """Build the common S1 through store chain used by production and tests.

    ``add_node`` receives ``(core_id, kernel_name, kernel_args, label)``. The
    label is diagnostic only; the production graph may ignore it.
    """
    if s1_block_count <= 0 or s3_block_count <= 0:
        raise ValueError("dynamic MoE slot requires non-zero S1/S3 block counts")

    def label(name):
        return f"{label_prefix}_{name}" if label_prefix else name

    slot_args = make_block_args(0)
    block0_args = make_block_args(0)
    s1_load = add_node(
        dma_core_id,
        "__snax_bingo_kernel_moe_dyn_load_s1_stage",
        block0_args,
        label("S1_LOAD_STAGE"),
    )
    s1_config = add_node(
        gemm_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_configure_gate_up_block0",
        block0_args,
        label("S1_CONFIG_BLOCK0_DURING_LOAD0"),
    )
    s1_compute = add_node(
        gemm_core_id,
        "__snax_bingo_kernel_moe_dyn_compute_s1_stage_pc",
        block0_args,
        label("S1_COMPUTE_STAGE"),
    )
    add_edge(input_ready, s1_load)
    add_edge(input_ready, s1_config)
    add_edge(s1_config, s1_compute)

    prefetch_s2_down = add_node(
        dma_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down",
        slot_args,
        label("S2_DOWN_PREFETCH"),
    )
    compute_gate_up_full = add_node(
        gemm_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full",
        slot_args,
        label("S2_COMPUTE_REMAINDER"),
    )
    for predecessor in (s1_load, s1_compute):
        add_edge(predecessor, prefetch_s2_down)
        add_edge(predecessor, compute_gate_up_full)

    s3_load = add_node(
        dma_core_id,
        "__snax_bingo_kernel_moe_dyn_load_s3_stage",
        block0_args,
        label("S3_LOAD_STAGE"),
    )
    s3_config = add_node(
        gemm_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_configure_down_block0",
        block0_args,
        label("S3_CONFIG_BLOCK0_DURING_LOAD0"),
    )
    s3_compute = add_node(
        gemm_core_id,
        "__snax_bingo_kernel_moe_dyn_compute_s3_stage_pc",
        block0_args,
        label("S3_COMPUTE_STAGE"),
    )
    for predecessor in (compute_gate_up_full, prefetch_s2_down):
        add_edge(predecessor, s3_load)
        add_edge(predecessor, s3_config)
    add_edge(s3_config, s3_compute)

    prefetch_s4_next_s1 = add_node(
        dma_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1",
        slot_args,
        label("S4_PREFETCH_OR_PREPARE_STORE"),
    )
    # Keep this independent from S4 compute so next-slot prefetch/store setup
    # can overlap the current slot's down-full work.
    add_edge(s3_load, prefetch_s4_next_s1)

    compute_down_full = add_node(
        gemm_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full",
        slot_args,
        label("S4_COMPUTE_REMAINDER"),
    )
    add_edge(s3_compute, compute_down_full)

    store = add_node(
        dma_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_store_and_gather_next",
        slot_args,
        label("STORE_GATHER_NEXT"),
    )
    add_edge(compute_down_full, store)
    add_edge(prefetch_s4_next_s1, store)
    return {
        "store": store,
        "s1_load": s1_load,
        "s1_compute": s1_compute,
        "s2_prefetch": prefetch_s2_down,
        "s2_compute": compute_gate_up_full,
        "s3_load": s3_load,
        "s3_compute": s3_compute,
        "s4_prepare": prefetch_s4_next_s1,
        "s4_compute": compute_down_full,
    }
