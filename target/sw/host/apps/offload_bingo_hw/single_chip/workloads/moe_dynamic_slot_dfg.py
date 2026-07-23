"""Shared production DFG builders for one dynamic individual-expert slot."""


SLOT_IMPLEMENTATIONS = ("discrete", "optimized")
DEFAULT_SLOT_IMPLEMENTATION = "optimized"

GATHER_KERNELS = {
    "discrete": "__snax_bingo_kernel_moe_dynamic_expert_gather_s1",
    "optimized": "__snax_bingo_kernel_moe_dyn_opt_gather_s1",
}

OPTIMIZED_STAGE_KERNELS = {
    "s1_load": "__snax_bingo_kernel_moe_dyn_opt_load_s1_stage",
    "s1_config": "__snax_bingo_kernel_moe_dyn_opt_config_s1_block0",
    "s1_compute": "__snax_bingo_kernel_moe_dyn_opt_compute_s1_stage",
    "s2_prefetch": "__snax_bingo_kernel_moe_dyn_opt_prefetch_s2",
    "s2_compute": "__snax_bingo_kernel_moe_dyn_opt_compute_s2",
    "s3_load": "__snax_bingo_kernel_moe_dyn_opt_load_s3_stage",
    "s3_config": "__snax_bingo_kernel_moe_dyn_opt_config_s3_block0",
    "s3_compute": "__snax_bingo_kernel_moe_dyn_opt_compute_s3_stage",
    "s4_prefetch": "__snax_bingo_kernel_moe_dyn_opt_prefetch_s4",
    "s4_compute": "__snax_bingo_kernel_moe_dyn_opt_compute_s4",
    "store": "__snax_bingo_kernel_moe_dyn_opt_store_gather",
}


def dynamic_expert_gather_kernel(implementation):
    _validate_implementation(implementation)
    return GATHER_KERNELS[implementation]


def _validate_implementation(implementation):
    if implementation not in SLOT_IMPLEMENTATIONS:
        raise ValueError(
            f"unknown slot implementation {implementation!r}; "
            f"expected one of {SLOT_IMPLEMENTATIONS}"
        )


def _build_discrete_slot_chain(
    *,
    add_node,
    add_edge,
    make_block_args,
    input_ready,
    s1_block_count,
    s3_block_count,
    dma_core_id,
    gemm_core_id,
    label,
):
    """Build the original ABI-visible, one-node-per-block chain."""
    s1_loads = []
    s1_computes = []
    for block in range(s1_block_count):
        block_args = make_block_args(block)
        load = add_node(
            dma_core_id,
            "__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block",
            block_args,
            label(f"S1_LOAD_BLOCK_{block}"),
        )
        if block == 0:
            config = add_node(
                gemm_core_id,
                "__snax_bingo_kernel_moe_dynamic_expert_configure_gate_up_block0",
                block_args,
                label("S1_CONFIG_BLOCK0_DURING_LOAD0"),
            )
        compute_kernel = (
            "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block_pc"
            if block == 0
            else "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block"
        )
        compute = add_node(
            gemm_core_id,
            compute_kernel,
            block_args,
            label(f"S1_COMPUTE_BLOCK_{block}"),
        )
        add_edge(input_ready, load)
        if block == 0:
            add_edge(input_ready, config)
            add_edge(config, compute)
        else:
            add_edge(s1_loads[block - 1], load)
            if block >= 2:
                add_edge(s1_computes[block - 2], load)
            add_edge(s1_computes[block - 1], compute)
        add_edge(load, compute)
        s1_loads.append(load)
        s1_computes.append(compute)

    slot_args = make_block_args(0)
    s2_prefetch = add_node(
        dma_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down",
        slot_args,
        label("S2_DOWN_PREFETCH"),
    )
    s2_compute = add_node(
        gemm_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full",
        slot_args,
        label("S2_COMPUTE_REMAINDER"),
    )
    add_edge(s1_computes[-1], s2_prefetch)
    add_edge(s1_computes[-1], s2_compute)

    s3_loads = []
    s3_computes = []
    for block in range(s3_block_count):
        block_args = make_block_args(block)
        load = add_node(
            dma_core_id,
            "__snax_bingo_kernel_moe_dynamic_expert_load_down_block",
            block_args,
            label(f"S3_LOAD_BLOCK_{block}"),
        )
        if block == 0:
            config = add_node(
                gemm_core_id,
                "__snax_bingo_kernel_moe_dynamic_expert_configure_down_block0",
                block_args,
                label("S3_CONFIG_BLOCK0_DURING_LOAD0"),
            )
        compute_kernel = (
            "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block_pc"
            if block == 0
            else "__snax_bingo_kernel_moe_dynamic_expert_compute_down_block"
        )
        compute = add_node(
            gemm_core_id,
            compute_kernel,
            block_args,
            label(f"S3_COMPUTE_BLOCK_{block}"),
        )
        add_edge(s2_compute, load)
        add_edge(s2_prefetch, load)
        if block == 0:
            add_edge(s2_compute, config)
            add_edge(s2_prefetch, config)
            add_edge(config, compute)
        else:
            add_edge(s3_loads[block - 1], load)
            if block >= 2:
                add_edge(s3_computes[block - 2], load)
            add_edge(s3_computes[block - 1], compute)
        add_edge(load, compute)
        s3_loads.append(load)
        s3_computes.append(compute)

    s4_prepare = add_node(
        dma_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1",
        slot_args,
        label("S4_PREFETCH_OR_PREPARE_STORE"),
    )
    add_edge(s3_loads[-1], s4_prepare)
    s4_compute = add_node(
        gemm_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_compute_down_full",
        slot_args,
        label("S4_COMPUTE_REMAINDER"),
    )
    add_edge(s3_computes[-1], s4_compute)
    store = add_node(
        dma_core_id,
        "__snax_bingo_kernel_moe_dynamic_expert_store_and_gather_next",
        slot_args,
        label("STORE_GATHER_NEXT"),
    )
    add_edge(s4_compute, store)
    add_edge(s4_prepare, store)
    return {
        "store": store,
        "s1_load": s1_loads,
        "s1_compute": s1_computes,
        "s2_prefetch": s2_prefetch,
        "s2_compute": s2_compute,
        "s3_load": s3_loads,
        "s3_compute": s3_computes,
        "s4_prepare": s4_prepare,
        "s4_compute": s4_compute,
    }


def _build_optimized_slot_chain(
    *,
    kernels,
    add_node,
    add_edge,
    make_block_args,
    input_ready,
    dma_core_id,
    gemm_core_id,
    label,
):
    """Build the production one-node-per-stage pipelined slot chain."""
    slot_args = make_block_args(0)
    block0_args = make_block_args(0)
    s1_load = add_node(
        dma_core_id, kernels["s1_load"], block0_args, label("S1_LOAD_STAGE")
    )
    s1_config = add_node(
        gemm_core_id,
        kernels["s1_config"],
        block0_args,
        label("S1_CONFIG_BLOCK0_DURING_LOAD0"),
    )
    s1_compute = add_node(
        gemm_core_id,
        kernels["s1_compute"],
        block0_args,
        label("S1_COMPUTE_STAGE"),
    )
    add_edge(input_ready, s1_load)
    add_edge(input_ready, s1_config)
    add_edge(s1_config, s1_compute)

    s2_prefetch = add_node(
        dma_core_id,
        kernels["s2_prefetch"],
        slot_args,
        label("S2_DOWN_PREFETCH"),
    )
    s2_compute = add_node(
        gemm_core_id,
        kernels["s2_compute"],
        slot_args,
        label("S2_COMPUTE_REMAINDER"),
    )
    for predecessor in (s1_load, s1_compute):
        add_edge(predecessor, s2_prefetch)
        add_edge(predecessor, s2_compute)

    s3_load = add_node(
        dma_core_id, kernels["s3_load"], block0_args, label("S3_LOAD_STAGE")
    )
    s3_config = add_node(
        gemm_core_id,
        kernels["s3_config"],
        block0_args,
        label("S3_CONFIG_BLOCK0_DURING_LOAD0"),
    )
    s3_compute = add_node(
        gemm_core_id,
        kernels["s3_compute"],
        block0_args,
        label("S3_COMPUTE_STAGE"),
    )
    for predecessor in (s2_compute, s2_prefetch):
        add_edge(predecessor, s3_load)
        add_edge(predecessor, s3_config)
    add_edge(s3_config, s3_compute)

    s4_prepare = add_node(
        dma_core_id,
        kernels["s4_prefetch"],
        slot_args,
        label("S4_PREFETCH_OR_PREPARE_STORE"),
    )
    add_edge(s3_load, s4_prepare)
    # The optimized S4 workers share the S3 synchronization control words.
    add_edge(s3_config, s4_prepare)

    s4_compute = add_node(
        gemm_core_id,
        kernels["s4_compute"],
        slot_args,
        label("S4_COMPUTE_REMAINDER"),
    )
    add_edge(s3_compute, s4_compute)

    store = add_node(
        dma_core_id, kernels["store"], slot_args, label("STORE_GATHER_NEXT")
    )
    add_edge(s4_compute, store)
    add_edge(s4_prepare, store)
    return {
        "store": store,
        "s1_load": s1_load,
        "s1_compute": s1_compute,
        "s2_prefetch": s2_prefetch,
        "s2_compute": s2_compute,
        "s3_load": s3_load,
        "s3_compute": s3_compute,
        "s4_prepare": s4_prepare,
        "s4_compute": s4_compute,
    }


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
    implementation=DEFAULT_SLOT_IMPLEMENTATION,
    label_prefix="",
):
    """Build either the comparison chain or the production optimized chain."""
    if s1_block_count <= 0 or s3_block_count <= 0:
        raise ValueError("dynamic MoE slot requires non-zero S1/S3 block counts")
    _validate_implementation(implementation)

    def label(name):
        return f"{label_prefix}_{name}" if label_prefix else name

    common = {
        "add_node": add_node,
        "add_edge": add_edge,
        "make_block_args": make_block_args,
        "input_ready": input_ready,
        "dma_core_id": dma_core_id,
        "gemm_core_id": gemm_core_id,
        "label": label,
    }
    if implementation == "discrete":
        return _build_discrete_slot_chain(
            **common,
            s1_block_count=s1_block_count,
            s3_block_count=s3_block_count,
        )
    return _build_optimized_slot_chain(
        **common,
        kernels=OPTIMIZED_STAGE_KERNELS,
    )
