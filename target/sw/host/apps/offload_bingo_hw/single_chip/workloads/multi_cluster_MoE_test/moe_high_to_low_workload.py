"""Optional fixed-order production-API workloads for multi_cluster_MoE_test."""

from bingo_kernel_args import (
    HostBingoKernelCheckResultArgs,
    SnaxBingoKernelIdma1dCopyArgs,
    SnaxBingoKernelMoeDynamicExpertBlockArgs,
)
from bingo_mem_handle import BingoMemAlloc, BingoMemSymbol
from bingo_node import BingoNode
from moe_dynamic_slot_dfg import (
    OPTIMIZED_STAGE_KERNELS,
    build_dynamic_expert_slot_chain,
    dynamic_expert_gather_kernel,
)
from moe_test_schedule import (
    C_TAIL_SMOKE_PROFILE,
    DMA_NONE,
    DYNAMIC_DESC_PROFILE,
    ENDS_INWARD_PROFILE,
    HIGH_TO_LOW_DMA_SERIAL_EIDS,
    HIGH_TO_LOW_PROFILE,
    LOW_TO_HIGH_PROFILE,
    S1_STAGE_SMOKE_PROFILE,
    SHAPE_C,
    SHAPE_M,
    STATIC_DESC_PROFILE,
    SlotSpec,
    audit_high_to_low_schedule,
    audit_ends_inward_schedule,
    audit_dynamic_desc_schedule,
    audit_low_to_high_schedule,
    audit_static_desc_schedule,
    cross_cluster_dma_release_edges,
)


GEMM_CORE = 0
DMA_CORE = 1
HOST_CORE = 2
PROD_CLUSTERS = (("c0", 0), ("c1", 1))
HEADER_BYTES = 64
SLOT_BYTES = 384
STATIC_BYTES = 192
PIPELINE_CTRL_SLOT_BYTES = 1024
PIPELINE_CTRL_BANK_OFFSET = 448


def _setup_name(schedule_profile: str) -> str:
    if schedule_profile == S1_STAGE_SMOKE_PROFILE:
        return "SETUP_S1_STAGE_SMOKE"
    if schedule_profile == C_TAIL_SMOKE_PROFILE:
        return "SETUP_C_TAIL_SMOKE"
    if schedule_profile == LOW_TO_HIGH_PROFILE:
        return "SETUP_LOW_TO_HIGH"
    if schedule_profile == ENDS_INWARD_PROFILE:
        return "SETUP_ENDS_INWARD"
    if schedule_profile == STATIC_DESC_PROFILE:
        return "SETUP_STATIC_DESC"
    if schedule_profile == DYNAMIC_DESC_PROFILE:
        return "SETUP_DYNAMIC_DESC"
    return "SETUP_HIGH_TO_LOW_E3_S3_STOREDBG"


def _schedule_tag(schedule_profile: str) -> str:
    if schedule_profile == LOW_TO_HIGH_PROFILE:
        return "LOW_TO_HIGH"
    if schedule_profile == ENDS_INWARD_PROFILE:
        return "ENDS_INWARD"
    if schedule_profile == STATIC_DESC_PROFILE:
        return "STATIC_DESC"
    if schedule_profile == DYNAMIC_DESC_PROFILE:
        return "DYNAMIC_DESC"
    return "HIGH_TO_LOW"


def _offset(handle, byte_offset: int):
    if byte_offset == 0:
        return handle
    if isinstance(handle, BingoMemSymbol):
        return BingoMemSymbol(handle.symbol_name, offset=handle.offset + byte_offset)
    return f"{handle.get_c_var_name()} + {byte_offset}"


def _pack_ctrl(slot: SlotSpec) -> int:
    skip_s2 = int(slot.s2_m_exec == 0)
    skip_s4 = int(slot.s4_m_exec == 0)
    return (
        1
        | (int(slot.skip_s1) << 1)
        | (int(slot.skip_s3) << 2)
        | (skip_s2 << 3)
        | (skip_s4 << 4)
        | (slot.s1_shape << 5)
        | (slot.s3_shape << 7)
        | (slot.s1_dma << 9)
        | (slot.s3_dma << 11)
        | (slot.cluster_index << 13)
        | (slot.local_slot << 14)
    )


def _pack_dma_slot(valid: bool, dma: int, slot: int) -> int:
    return ((1 | (dma << 1)) << (slot * 3)) if valid else 0


class HighToLowSetupArgs(HostBingoKernelCheckResultArgs):
    """Initialize all runtime records for the selected fixed-order queues."""

    def __init__(self, p, mh, queues):
        setup_name = _setup_name(p["schedule_profile"])
        super().__init__(
            golden_data_addr=mh["input"],
            output_data_addr=mh["input"],
            data_size=1,
            name=setup_name,
        )
        self.p = p
        self.mh = mh
        self.queues = queues
        for prefix, _cluster in PROD_CLUSTERS:
            for name in (
                "layout",
                "gate",
                "up",
                "down",
                "gate_out",
                "pipeline_ctrl",
                "prod_output_l3",
            ):
                setattr(self, f"_abi_memref_{prefix}_{name}", mh[f"{prefix}_{name}"])

    def _addr(self, value, handle_name_map, as_64bit=True):
        assignments = {}
        self._process_addr(
            value,
            "value",
            assignments,
            handle_name_map,
            split_64bit=False,
            as_64bit=as_64bit,
        )
        return assignments["value"]

    def _static_init_lines(self, prefix, cluster, handle_name_map):
        p = self.p
        mh = self.mh
        slots = self.queues[prefix]
        row_stride = p["token_row_stride"]
        s3_row_bytes = p["token_payload_bytes"] // (2 * p["s3_block_count"])
        down_half_bytes = p["down_weight_bytes"] // 2
        down_b_n_stride = [p["down_K"] * (16 << shape) for shape in range(3)]
        down_a_m_stride = [
            ((p["down_K"] * 8) // (p["base_mesh_col"] << shape)) * 64
            for shape in range(3)
        ]
        down_d_m_stride = [
            (p["base_mesh_row"] >> shape) * row_stride for shape in range(3)
        ]

        static_l3 = self._addr(mh[f"{prefix}_prod_static_l3"], handle_name_map)
        runtime_l3 = self._addr(mh[f"{prefix}_prod_runtime_l3"], handle_name_map)
        runtime_l1 = self._addr(
            mh[f"{prefix}_prod_runtime_l1"], handle_name_map, as_64bit=False
        )
        token_refs_l1 = self._addr(
            mh[f"{prefix}_token_refs_l1"], handle_name_map, as_64bit=False
        )
        input_l3 = self._addr(mh["input"], handle_name_map)
        gate_l3 = self._addr(mh[f"{prefix}_gate_src"], handle_name_map)
        up_l3 = self._addr(mh[f"{prefix}_up_src"], handle_name_map)
        down_l3 = self._addr(mh[f"{prefix}_down_src"], handle_name_map)
        output_l3 = self._addr(mh[f"{prefix}_prod_output_l3"], handle_name_map)
        l1_a = self._addr(mh[f"{prefix}_a"], handle_name_map, as_64bit=False)
        l1_gate = self._addr(mh[f"{prefix}_gate"], handle_name_map, as_64bit=False)
        l1_up = self._addr(mh[f"{prefix}_up"], handle_name_map, as_64bit=False)
        l1_down = self._addr(mh[f"{prefix}_down"], handle_name_map, as_64bit=False)
        l1_d = self._addr(mh[f"{prefix}_gate_out"], handle_name_map, as_64bit=False)
        l1_down_d = self._addr(mh[f"{prefix}_out"], handle_name_map, as_64bit=False)
        l1_scratch = self._addr(
            mh[f"{prefix}_gate_scratch"], handle_name_map, as_64bit=False
        )
        static_name = f"prod_{prefix}_st"
        runtime_name = f"prod_{prefix}_rt"
        runtime_bytes = HEADER_BYTES + len(slots) * SLOT_BYTES
        output_bytes = (max(slot.expert_id for slot in slots) + 1) * p[
            "prod_output_expert_stride_bytes"
        ]
        lines = [
            f"__snax_bingo_moe_dynamic_expert_static_args_t *{static_name} = "
            f"(__snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t){static_l3};",
            f"uint8_t *{runtime_name} = (uint8_t *)(uintptr_t){runtime_l3};",
            f"memset({static_name}, 0, sizeof(*{static_name}));",
            f"memset({runtime_name}, 0, {runtime_bytes}u);",
        ]
        if p["prod_clear_output"]:
            lines.append(
                f"memset((void *)(uintptr_t){output_l3}, 0, {output_bytes}u);"
            )
        lines += [
            f"{static_name}->token_refs_addr = {token_refs_l1};",
            f"{static_name}->input_A_l3_base = {input_l3};",
            f"{static_name}->indiv_gate_B_l3 = {gate_l3};",
            f"{static_name}->indiv_up_B_l3 = {up_l3};",
            f"{static_name}->indiv_down_B_l3 = {down_l3};",
            f"{static_name}->output_l3_base = {output_l3};",
            f"{static_name}->active_state_l1_addr = {runtime_l1};",
            f"{static_name}->l1_a_addr = {l1_a};",
            f"{static_name}->l1_b_gate_addr = {l1_gate};",
            f"{static_name}->l1_b_up_addr = {l1_up};",
            f"{static_name}->l1_b_down_addr = {l1_down};",
            f"{static_name}->l1_d_addr = {l1_d};",
            f"{static_name}->l1_down_d_addr = {l1_down_d};",
            f"{static_name}->l1_d1_scratch_addr = {l1_scratch};",
            f"{static_name}->A_token_bytes = {p['token_payload_bytes']}u;",
            "/* Logical expert ids stay distinct, but this focused workload maps all",
            " * of them onto one synthetic physical weight backing per cluster. */",
            f"{static_name}->indiv_B_expert_stride = 0u;",
            f"{static_name}->indiv_down_B_expert_stride = 0u;",
            f"{static_name}->indiv_B_block_stride = {p['gate_block_bytes']}u;",
            f"{static_name}->indiv_down_B_block_stride = {p['down_block_bytes']}u;",
            f"{static_name}->s1_block_count = {p['s1_block_count']}u;",
            f"{static_name}->s3_block_count = {p['s3_block_count']}u;",
            f"{static_name}->indiv_K1 = {p['gate_K']}u;",
            f"{static_name}->indiv_N_per_block = {p['indiv_N_per_block']}u;",
            f"{static_name}->indiv_down_K1 = {p['down_K']}u;",
            f"{static_name}->indiv_down_N_per_block = {p['indiv_down_N_per_block']}u;",
            f"{static_name}->rescale_mult = 1u;",
            f"{static_name}->rescale_shift = 0u;",
            f"{static_name}->output_expert_stride_bytes = "
            f"{p['prod_output_expert_stride_bytes']}u;",
            f"{static_name}->max_tokens_per_expert = "
            f"{p['prod_max_tokens_per_expert']}u;",
            f"{static_name}->A_row_stride = {row_stride}u;",
            f"{static_name}->s3_row_bytes = {s3_row_bytes}u;",
            f"{static_name}->down_half_weight_bytes = {down_half_bytes}u;",
            f"{static_name}->down_b_k_section = {p['down_K'] * 8 * 2}u;",
            f"{static_name}->down_b_n_stride[0] = {down_b_n_stride[0]}u;",
            f"{static_name}->down_b_n_stride[1] = {down_b_n_stride[1]}u;",
            f"{static_name}->down_b_n_stride[2] = {down_b_n_stride[2]}u;",
            f"{static_name}->down_a_m_stride[0] = {down_a_m_stride[0]}u;",
            f"{static_name}->down_a_m_stride[1] = {down_a_m_stride[1]}u;",
            f"{static_name}->down_a_m_stride[2] = {down_a_m_stride[2]}u;",
            f"{static_name}->down_d_m_stride[0] = {down_d_m_stride[0]}u;",
            f"{static_name}->down_d_m_stride[1] = {down_d_m_stride[1]}u;",
            f"{static_name}->down_d_m_stride[2] = {down_d_m_stride[2]}u;",
            f"((uint32_t *){runtime_name})[2] = {len(self.queues['c0'])}u;",
            f"((uint32_t *){runtime_name})[3] = {len(self.queues['c1'])}u;",
        ]
        return lines, runtime_name, l1_d

    def _slot_init_lines(self, slot, runtime_name, l1_d):
        p = self.p
        slot_name = f"prod_{slot.cluster_name}_slot{slot.local_slot}"
        slot_offset = HEADER_BYTES + slot.local_slot * SLOT_BYTES
        s1_n = p["indiv_N_per_block"] // (
            p["base_mesh_col"] << slot.s1_shape
        )
        s2_n = (p["s1_block_count"] * p["indiv_N_per_block"]) // (
            p["base_mesh_col"] << SHAPE_C
        )
        s3_n = p["indiv_down_N_per_block"] // (
            p["base_mesh_col"] << slot.s3_shape
        )
        s4_n = (p["s3_block_count"] * p["indiv_down_N_per_block"]) // (
            p["base_mesh_col"] << SHAPE_C
        )
        dma_slot_vd = _pack_dma_slot(
            slot.s2_prefetch_dma != DMA_NONE, slot.s2_prefetch_dma, 2
        ) | _pack_dma_slot(
            slot.s4_prefetch_dma != DMA_NONE, slot.s4_prefetch_dma, 3
        )
        lines = [
            f"__snax_bingo_kernel_moe_dynamic_expert_args_t *{slot_name} = "
            f"(__snax_bingo_kernel_moe_dynamic_expert_args_t *)"
            f"({runtime_name} + {slot_offset}u);",
            f"{slot_name}->ctrl = {_pack_ctrl(slot)}u;",
            f"{slot_name}->expert_id = {slot.expert_id}u;",
            f"{slot_name}->token_ref_start = {slot.token_start_rank}u;",
            f"{slot_name}->ntokens = {slot.ntokens}u;",
            f"{slot_name}->m_s2_exec = {slot.s2_m_exec}u;",
            f"{slot_name}->m_s4_exec = {slot.s4_m_exec}u;",
            f"{slot_name}->dma_slot_vd = {dma_slot_vd}u;",
            f"{slot_name}->dma_slot_eids = 0u;",
        ]
        for block in range(p["s1_block_count"]):
            output_offset = block * p["bank_mode0_output_block_span"]
            lines += [
                f"{slot_name}->s1_call[{block}].valid = {int(not slot.skip_s1)}u;",
                f"{slot_name}->s1_call[{block}].output_D0_addr = "
                f"{l1_d} + {output_offset}u;",
                f"{slot_name}->s1_call[{block}].N = {s1_n}u;",
                f"{slot_name}->s1_call[{block}].array_shape = {slot.s1_shape}u;",
            ]
        for block in range(p["s3_block_count"]):
            lines += [
                f"{slot_name}->s3_call[{block}].valid = {int(not slot.skip_s3)}u;",
                f"{slot_name}->s3_call[{block}].N = {s3_n}u;",
                f"{slot_name}->s3_call[{block}].array_shape = {slot.s3_shape}u;",
                f"{slot_name}->s3_call[{block}].reserved = 0u;",
            ]
        lines += [
            f"{slot_name}->s2_call.valid = {int(slot.s2_m_exec != 0)}u;",
            f"{slot_name}->s2_call.token_start = {slot.s2_token_start}u;",
            f"{slot_name}->s2_call.reserved = 0u;",
            f"{slot_name}->s2_call.M = {slot.s2_m_exec}u;",
            f"{slot_name}->s2_call.N = {s2_n}u;",
            f"{slot_name}->s2_call.array_shape = {SHAPE_C}u;",
            f"{slot_name}->s4_call.valid = {int(slot.s4_m_exec != 0)}u;",
            f"{slot_name}->s4_call.token_start = {slot.s4_token_start}u;",
            f"{slot_name}->s4_call.reserved0 = 0u;",
            f"{slot_name}->s4_call.reserved1 = 0u;",
            f"{slot_name}->s4_call.M = {slot.s4_m_exec}u;",
            f"{slot_name}->s4_call.N = {s4_n}u;",
            f"{slot_name}->s4_call.array_shape = {SHAPE_C}u;",
        ]
        return lines

    def get_post_init_code(self, args_var, handle_name_map):
        del args_var
        lines = [
            f"_Static_assert(BINGO_MOE_DYNAMIC_ARG_SLOT_BYTES == {SLOT_BYTES}u, "
            '"test and production dynamic slot ABI diverged");',
            f"_Static_assert(BINGO_MOE_STATIC_ARG_SLOT_BYTES == {STATIC_BYTES}u, "
            '"test and production static slot ABI diverged");',
        ]
        for prefix, cluster in PROD_CLUSTERS:
            static_lines, runtime_name, l1_d = self._static_init_lines(
                prefix, cluster, handle_name_map
            )
            lines += static_lines
            for slot in self.queues[prefix]:
                lines += self._slot_init_lines(slot, runtime_name, l1_d)
        return lines


def define_high_to_low_memory(p, queues):
    mh = {
        "input": BingoMemSymbol("moe_test_input_A"),
        "prod_slot_token_refs": BingoMemSymbol("moe_test_prod_slot_token_refs"),
    }
    for prefix, cluster in PROD_CLUSTERS:
        for slot in queues[prefix]:
            mh[f"{prefix}_e{slot.expert_id:02d}_golden"] = BingoMemSymbol(
                f"moe_test_{prefix}_e{slot.expert_id:02d}_golden"
            )
        mh[f"{prefix}_gate_src"] = BingoMemSymbol(f"moe_test_{prefix}_gate_B")
        mh[f"{prefix}_up_src"] = BingoMemSymbol(f"moe_test_{prefix}_up_B")
        mh[f"{prefix}_down_src"] = BingoMemSymbol(f"moe_test_{prefix}_down_B")
        layout = BingoMemAlloc(
            f"moe_test_{prefix}_prod_l1_layout",
            size=p["bank_tcdm_size"],
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=p["bank_tcdm_row_bytes"],
        )
        mh[f"{prefix}_layout"] = layout
        mh[f"{prefix}_a"] = _offset(layout, p["bank_delta_local_a"])
        mh[f"{prefix}_gate"] = _offset(layout, p["bank_delta_local_b0"])
        mh[f"{prefix}_up"] = _offset(layout, p["bank_delta_local_b1"])
        mh[f"{prefix}_down"] = _offset(
            layout, p["bank_mode1_region_offset"] + p["bank_delta_local_b0"]
        )
        mh[f"{prefix}_gate_out"] = _offset(layout, p["bank_delta_local_d0"])
        mh[f"{prefix}_gate_scratch"] = mh[f"{prefix}_gate_out"]
        mh[f"{prefix}_out"] = _offset(layout, p["bank_delta_local_mode1_d0"])
        mh[f"{prefix}_token_refs_l1"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_token_refs_l1",
            size=p["prod_token_refs_bytes"],
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=64,
        )
        runtime_bytes = HEADER_BYTES + len(queues[prefix]) * SLOT_BYTES
        mh[f"{prefix}_prod_static_l3"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_static_l3", size=STATIC_BYTES, mem_level="L3"
        )
        mh[f"{prefix}_prod_runtime_l3"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_runtime_l3", size=runtime_bytes, mem_level="L3"
        )
        mh[f"{prefix}_prod_static_l1"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_static_l1",
            size=STATIC_BYTES,
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=64,
        )
        mh[f"{prefix}_prod_runtime_l1"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_runtime_l1",
            size=runtime_bytes,
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=64,
        )
        mh[f"{prefix}_pipeline_ctrl"] = BingoMemAlloc(
            f"moe_test_{prefix}_pipeline_ctrl",
            size=len(queues[prefix]) * PIPELINE_CTRL_SLOT_BYTES,
            mem_level="L1",
            chip_id=0,
            cluster_id=cluster,
            alignment=p["bank_tcdm_row_bytes"],
        )
        if p["prod_clear_output"]:
            output_bytes = (
                max(slot.expert_id for slot in queues[prefix]) + 1
            ) * p["prod_output_expert_stride_bytes"]
        else:
            output_bytes = p["prod_output_bytes"]
        mh[f"{prefix}_prod_output_l3"] = BingoMemAlloc(
            f"moe_test_{prefix}_prod_output_l3", size=output_bytes, mem_level="L3"
        )
    return mh


def _add_node(dfg, cluster, core, kernel, args, node_name=""):
    node = BingoNode(
        assigned_chiplet_id=0,
        assigned_cluster_id=cluster,
        assigned_core_id=core,
        node_name=node_name,
        kernel_name=kernel,
        kernel_args=args,
    )
    dfg.bingo_add_node(node)
    return node


def _add_copy(dfg, cluster, src, dst, size, node_name):
    return _add_node(
        dfg,
        cluster,
        DMA_CORE,
        "__snax_bingo_kernel_idma_1d_copy",
        SnaxBingoKernelIdma1dCopyArgs(src, dst, size),
        node_name,
    )


def _add_scope_marker(dfg, cluster, args, begin, node_name):
    kernel = (
        "__snax_bingo_kernel_moe_dyn_opt_cluster_begin"
        if begin
        else "__snax_bingo_kernel_moe_dyn_opt_cluster_end"
    )
    return _add_node(dfg, cluster, DMA_CORE, kernel, args, node_name)


def add_high_to_low_schedule(dfg, p, mh, queues, slot_implementation):
    if p["schedule_profile"] == HIGH_TO_LOW_PROFILE:
        audit_high_to_low_schedule(queues)
    elif p["schedule_profile"] == LOW_TO_HIGH_PROFILE:
        audit_low_to_high_schedule(queues)
    elif p["schedule_profile"] == ENDS_INWARD_PROFILE:
        audit_ends_inward_schedule(queues)
    elif p["schedule_profile"] == STATIC_DESC_PROFILE:
        audit_static_desc_schedule(queues)
    elif p["schedule_profile"] == DYNAMIC_DESC_PROFILE:
        audit_dynamic_desc_schedule(queues)
    tag = _schedule_tag(p["schedule_profile"])
    setup = _add_node(
        dfg,
        0,
        HOST_CORE,
        "__host_bingo_kernel_check_result",
        HighToLowSetupArgs(p, mh, queues),
        _setup_name(p["schedule_profile"]),
    )
    runtime_ready = {}
    gathers = {}
    scope_begins = {}
    scope_args = {}
    for prefix, cluster in PROD_CLUSTERS:
        refs_to_l1 = _add_copy(
            dfg,
            cluster,
            mh["prod_slot_token_refs"],
            mh[f"{prefix}_token_refs_l1"],
            p["prod_token_refs_bytes"],
            f"{prefix.upper()}_{tag}_LOAD_TOKEN_REFS",
        )
        static_to_l1 = _add_copy(
            dfg,
            cluster,
            mh[f"{prefix}_prod_static_l3"],
            mh[f"{prefix}_prod_static_l1"],
            STATIC_BYTES,
            f"{prefix.upper()}_{tag}_LOAD_STATIC_ABI",
        )
        runtime_bytes = HEADER_BYTES + len(queues[prefix]) * SLOT_BYTES
        runtime_to_l1 = _add_copy(
            dfg,
            cluster,
            mh[f"{prefix}_prod_runtime_l3"],
            mh[f"{prefix}_prod_runtime_l1"],
            runtime_bytes,
            f"{prefix.upper()}_{tag}_LOAD_DYNAMIC_ABI",
        )
        dfg.bingo_add_edge(setup, refs_to_l1)
        dfg.bingo_add_edge(refs_to_l1, static_to_l1)
        dfg.bingo_add_edge(static_to_l1, runtime_to_l1)
        runtime_ready[prefix] = runtime_to_l1

        slot0_args = SnaxBingoKernelMoeDynamicExpertBlockArgs(
            _offset(mh[f"{prefix}_prod_runtime_l1"], HEADER_BYTES),
            mh[f"{prefix}_prod_static_l1"],
            _offset(mh[f"{prefix}_pipeline_ctrl"], PIPELINE_CTRL_BANK_OFFSET),
            0,
        )
        scope_args[prefix] = slot0_args
        scope_begins[prefix] = _add_scope_marker(
            dfg,
            cluster,
            slot0_args,
            True,
            f"{prefix.upper()}_{tag}_SCOPE_BEGIN",
        )
        gathers[prefix] = _add_node(
            dfg,
            cluster,
            DMA_CORE,
            dynamic_expert_gather_kernel(slot_implementation),
            slot0_args,
            f"{prefix.upper()}_{tag}_SLOT0_GATHER",
        )

    # This one-time common release starts both cluster-local streams together.
    # Their task chains then advance independently except at explicit DMA edges.
    for ready in runtime_ready.values():
        for scope_begin in scope_begins.values():
            dfg.bingo_add_edge(ready, scope_begin)
    for prefix in gathers:
        dfg.bingo_add_edge(scope_begins[prefix], gathers[prefix])

    final_stores = {}
    slot_chains = {}
    slot_chains_by_eid = {}
    for prefix, cluster in PROD_CLUSTERS:
        predecessor = gathers[prefix]
        for slot in queues[prefix]:
            slot_addr = _offset(
                mh[f"{prefix}_prod_runtime_l1"],
                HEADER_BYTES + slot.local_slot * SLOT_BYTES,
            )
            pipeline_ctrl_addr = _offset(
                mh[f"{prefix}_pipeline_ctrl"],
                PIPELINE_CTRL_BANK_OFFSET
                + slot.local_slot * PIPELINE_CTRL_SLOT_BYTES,
            )
            static_addr = mh[f"{prefix}_prod_static_l1"]

            def make_block_args(
                block,
                slot_addr=slot_addr,
                static_addr=static_addr,
                ctrl=pipeline_ctrl_addr,
            ):
                return SnaxBingoKernelMoeDynamicExpertBlockArgs(
                    slot_addr,
                    static_addr,
                    ctrl,
                    block,
                )

            chain = build_dynamic_expert_slot_chain(
                add_node=lambda core, kernel, args, label, cluster=cluster: _add_node(
                    dfg, cluster, core, kernel, args, label
                ),
                add_edge=dfg.bingo_add_edge,
                add_descriptor_sequence=lambda producer, consumer: dfg.add_edge(
                    producer, consumer, descriptor_sequence=True
                ),
                make_block_args=make_block_args,
                input_ready=predecessor,
                s1_block_count=p["s1_block_count"],
                s3_block_count=p["s3_block_count"],
                dma_core_id=DMA_CORE,
                gemm_core_id=GEMM_CORE,
                implementation=slot_implementation,
                label_prefix=(
                    f"{prefix.upper()}_{tag}_SLOT{slot.local_slot}_"
                    f"E{slot.expert_id}_T{slot.ntokens}"
                ),
            )
            slot_chains[(prefix, slot.local_slot)] = chain
            slot_chains_by_eid.setdefault(slot.expert_id, chain)
            predecessor = chain["store"]
        final_stores[prefix] = predecessor

    scope_ends = {}
    for prefix, cluster in PROD_CLUSTERS:
        scope_ends[prefix] = _add_scope_marker(
            dfg,
            cluster,
            scope_args[prefix],
            False,
            f"{prefix.upper()}_{tag}_SCOPE_END",
        )
        dfg.bingo_add_edge(final_stores[prefix], scope_ends[prefix])

    # The descending reference has one global BOTH-DMA tail.  Constrain only
    # the DMA release/acquire boundary; the next cluster may still configure
    # block0 while it waits, and neither cluster waits for the peer's store.
    present_serial_eids = (
        tuple(
            eid
            for eid in HIGH_TO_LOW_DMA_SERIAL_EIDS
            if eid in slot_chains_by_eid
        )
        if p["schedule_profile"] == HIGH_TO_LOW_PROFILE
        else ()
    )
    for previous_eid, current_eid in zip(
        present_serial_eids,
        present_serial_eids[1:],
    ):
        previous_s3_load = slot_chains_by_eid[previous_eid]["s3_load"]
        current_s1_load = slot_chains_by_eid[current_eid]["s1_load"]
        if isinstance(previous_s3_load, list):
            previous_s3_load = previous_s3_load[-1]
        if isinstance(current_s1_load, list):
            current_s1_load = current_s1_load[0]
        dfg.bingo_add_edge(previous_s3_load, current_s1_load)

    if p["schedule_profile"] in (ENDS_INWARD_PROFILE, DYNAMIC_DESC_PROFILE):
        stage_keys = {
            "S1": "s1_load",
            "S2PF": "s2_prefetch",
            "S3": "s3_load",
            "S4PF": "s4_prepare",
        }
        for previous, current in cross_cluster_dma_release_edges(queues):
            previous_chain = slot_chains[(previous[0], previous[1])]
            current_chain = slot_chains[(current[0], current[1])]
            previous_node = previous_chain[stage_keys[previous[3]]]
            current_node = current_chain[stage_keys[current[3]]]
            if isinstance(previous_node, list):
                previous_node = previous_node[-1]
            if isinstance(current_node, list):
                current_node = current_node[0]
            dfg.bingo_add_edge(previous_node, current_node)

    if p["schedule_profile"] == LOW_TO_HIGH_PROFILE:
        # The final E0 action is one SPLIT decision. Both slices wait until
        # E1/C0 and E2/C1 have completed, then enter their local pipelines.
        for prefix, peer_prefix in (("c0", "c1"), ("c1", "c0")):
            split_slot = queues[prefix][-1]
            peer_previous = queues[peer_prefix][-2]
            split_chain = slot_chains[(prefix, split_slot.local_slot)]
            peer_store = slot_chains[(peer_prefix, peer_previous.local_slot)]["store"]
            entries = split_chain["s1_load"]
            if not isinstance(entries, list):
                entries = [entries]
            for entry in entries:
                dfg.bingo_add_edge(peer_store, entry)
            dfg.bingo_add_edge(peer_store, split_chain["s1_config"])

    checks = []
    previous_check = None
    ordered_slots = sorted(
        (
            (slot.expert_id, prefix, slot)
            for prefix, _cluster in PROD_CLUSTERS
            for slot in queues[prefix]
        )
    )
    for _expert_id, prefix, slot in ordered_slots:
        name = (
            f"{prefix.upper()}_{tag}_E{slot.expert_id}_"
            f"R{slot.token_start_rank}_OUTPUT"
        )
        output = _offset(
            mh[f"{prefix}_prod_output_l3"],
            slot.expert_id * p["prod_output_expert_stride_bytes"]
            + slot.token_start_rank * p["token_payload_bytes"],
        )
        check = _add_node(
            dfg,
            0,
            HOST_CORE,
            "__host_bingo_kernel_check_result",
            HostBingoKernelCheckResultArgs(
                golden_data_addr=mh[f"{prefix}_e{slot.expert_id:02d}_golden"],
                output_data_addr=output,
                data_size=slot.ntokens * p["token_payload_bytes"],
                name=name,
            ),
            name,
        )
        if previous_check is None:
            for scope_end in scope_ends.values():
                dfg.bingo_add_edge(scope_end, check)
        else:
            dfg.bingo_add_edge(previous_check, check)
        checks.append(check)
        previous_check = check
    return checks


def add_s1_stage_smoke_schedule(dfg, p, mh, queues):
    """Build only gather + S1 load/config/compute for the C/C tail smoke test."""
    setup = _add_node(
        dfg,
        0,
        HOST_CORE,
        "__host_bingo_kernel_check_result",
        HighToLowSetupArgs(p, mh, queues),
        "SETUP_S1_STAGE_SMOKE",
    )
    runtime_ready = {}
    gathers = {}
    for prefix, cluster in PROD_CLUSTERS:
        refs_to_l1 = _add_copy(
            dfg,
            cluster,
            mh["prod_slot_token_refs"],
            mh[f"{prefix}_token_refs_l1"],
            p["prod_token_refs_bytes"],
            f"{prefix.upper()}_S1_SMOKE_LOAD_TOKEN_REFS",
        )
        static_to_l1 = _add_copy(
            dfg,
            cluster,
            mh[f"{prefix}_prod_static_l3"],
            mh[f"{prefix}_prod_static_l1"],
            STATIC_BYTES,
            f"{prefix.upper()}_S1_SMOKE_LOAD_STATIC_ABI",
        )
        runtime_bytes = HEADER_BYTES + len(queues[prefix]) * SLOT_BYTES
        runtime_to_l1 = _add_copy(
            dfg,
            cluster,
            mh[f"{prefix}_prod_runtime_l3"],
            mh[f"{prefix}_prod_runtime_l1"],
            runtime_bytes,
            f"{prefix.upper()}_S1_SMOKE_LOAD_DYNAMIC_ABI",
        )
        dfg.bingo_add_edge(setup, refs_to_l1)
        dfg.bingo_add_edge(refs_to_l1, static_to_l1)
        dfg.bingo_add_edge(static_to_l1, runtime_to_l1)
        runtime_ready[prefix] = runtime_to_l1

        slot0_args = SnaxBingoKernelMoeDynamicExpertBlockArgs(
            _offset(mh[f"{prefix}_prod_runtime_l1"], HEADER_BYTES),
            mh[f"{prefix}_prod_static_l1"],
            _offset(mh[f"{prefix}_pipeline_ctrl"], PIPELINE_CTRL_BANK_OFFSET),
            0,
        )
        gathers[prefix] = _add_node(
            dfg,
            cluster,
            DMA_CORE,
            dynamic_expert_gather_kernel("optimized"),
            slot0_args,
            f"{prefix.upper()}_S1_SMOKE_SLOT0_GATHER",
        )

    for ready in runtime_ready.values():
        for gather in gathers.values():
            dfg.bingo_add_edge(ready, gather)

    s1_loads = {}
    s1_computes = []
    for prefix, cluster in PROD_CLUSTERS:
        predecessor = gathers[prefix]
        for slot in queues[prefix]:
            slot_addr = _offset(
                mh[f"{prefix}_prod_runtime_l1"],
                HEADER_BYTES + slot.local_slot * SLOT_BYTES,
            )
            pipeline_ctrl_addr = _offset(
                mh[f"{prefix}_pipeline_ctrl"],
                PIPELINE_CTRL_BANK_OFFSET
                + slot.local_slot * PIPELINE_CTRL_SLOT_BYTES,
            )
            block_args = SnaxBingoKernelMoeDynamicExpertBlockArgs(
                slot_addr,
                mh[f"{prefix}_prod_static_l1"],
                pipeline_ctrl_addr,
                0,
            )
            label = (
                f"{prefix.upper()}_S1_SMOKE_SLOT{slot.local_slot}_"
                f"E{slot.expert_id}_T{slot.ntokens}"
            )
            s1_load = _add_node(
                dfg,
                cluster,
                DMA_CORE,
                OPTIMIZED_STAGE_KERNELS["s1_load"],
                block_args,
                f"{label}_S1_LOAD_STAGE",
            )
            s1_config = _add_node(
                dfg,
                cluster,
                GEMM_CORE,
                OPTIMIZED_STAGE_KERNELS["s1_config"],
                block_args,
                f"{label}_S1_CONFIG_BLOCK0_DURING_LOAD0",
            )
            s1_compute = _add_node(
                dfg,
                cluster,
                GEMM_CORE,
                OPTIMIZED_STAGE_KERNELS["s1_compute"],
                block_args,
                f"{label}_S1_COMPUTE_STAGE",
            )
            dfg.bingo_add_edge(predecessor, s1_load)
            dfg.bingo_add_edge(predecessor, s1_config)
            dfg.bingo_add_edge(s1_config, s1_compute)
            s1_loads[slot.expert_id] = s1_load
            s1_computes.append(s1_compute)

    if 23 in s1_loads and 24 in s1_loads:
        dfg.bingo_add_edge(s1_loads[23], s1_loads[24])

    done = _add_node(
        dfg,
        0,
        HOST_CORE,
        "__host_bingo_kernel_check_result",
        HostBingoKernelCheckResultArgs(
            golden_data_addr=mh["input"],
            output_data_addr=mh["input"],
            data_size=1,
            name="S1_STAGE_SMOKE_DONE",
        ),
        "S1_STAGE_SMOKE_DONE",
    )
    for s1_compute in s1_computes:
        dfg.bingo_add_edge(s1_compute, done)
    return [done]
