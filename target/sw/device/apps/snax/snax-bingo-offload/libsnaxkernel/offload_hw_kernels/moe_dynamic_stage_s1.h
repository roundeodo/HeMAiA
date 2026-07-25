// Internal dynamic-MoE module; include through moe_dynamic.h.
#pragma once

// Optimized S1 gather/load/compute API.
//
// Production DFGs use exactly one optimized entry per worker and stage:
//   DM core: opt_gather, opt_load_s1, opt_prefetch_s2, opt_load_s3,
//            opt_prefetch_s4, opt_store_gather
//   VC core: opt_config_s1, opt_compute_s1, opt_compute_s2, opt_config_s3,
//            opt_compute_s3, opt_compute_s4
// Per-block comparison entry points live in moe_dynamic_discrete.h. Production
// kernels never dispatch to the comparison implementation.
//
// main_bingo.py fixes each kernel to either GEMM core 0 or DM core 1. These
// workload-private entry points therefore do not repeat core-id checks at run time.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_init_output_gaps(void *arg)
{
    const __snax_bingo_kernel_moe_init_output_gaps_args_t *cfg =
        (const __snax_bingo_kernel_moe_init_output_gaps_args_t *)arg;
    uint32_t gap_bytes = cfg->row_stride_bytes - cfg->row_payload_bytes;
    for (uint32_t row = 0u; row < cfg->rows; row++) {
        __moe_zero_tcdm(
            cfg->output_base + row * cfg->row_stride_bytes +
                cfg->row_payload_bytes,
            gap_bytes);
    }
    return BINGO_RET_SUCC;
}

static inline void __moe_initialize_slot_state(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);

    s1->valid = 0u;
    s1->csr_prepared_stage = MOE_CSR_PREPARED_NONE;
    s1->xdma_prepared_stage = MOE_XDMA_PREPARED_NONE;
    s1->csr_prepared_reserved = 0u;
    s2->valid = 0u;
    s2->sync_enabled = 0u;
    s2->compute_done = 0u;
    s2->prefetch_done = 0u;
    s2->store_prepared = 0u;
    s2->reserved = 0u;

    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        asm volatile("fence rw, rw" ::: "memory");
        return;
    }

    uint32_t ctrl = cfg->ctrl;
    if (MOE_DYN_CTRL_SKIP_S1(ctrl) == 0u) {
        uint32_t expert_id = cfg->expert_id;
        s1->gate_src_base = st->indiv_gate_B_l3 +
            (uint64_t)expert_id * st->indiv_B_expert_stride;
        s1->up_src_base = st->indiv_up_B_l3 +
            (uint64_t)expert_id * st->indiv_B_expert_stride;
        s1->gate_dst_base = st->l1_b_gate_addr;
        s1->up_dst_base = st->l1_b_up_addr;
        s1->block_bytes = st->indiv_B_block_stride;
        s1->block_count = st->s1_block_count;
        s1->binding = MOE_DYN_CTRL_DMA_S1(ctrl);
        s1->valid = 1u;
    }

    uint32_t pf_slot = MOE_DYN_DMA_SLOT_S2_PREFETCH;
    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, pf_slot) != 0u) {
        uint32_t pf_expert = MOE_DYN_DMA_EID(cfg->dma_slot_eids, pf_slot);
        s2->down_src_base = st->indiv_down_B_l3 +
            (uint64_t)pf_expert * st->indiv_down_B_expert_stride;
        s2->down_dst_base = st->l1_b_down_addr;
        s2->half_bytes = st->down_half_weight_bytes;
        s2->block_bytes = st->indiv_down_B_block_stride;
        s2->block_count = st->s3_block_count;
        s2->s1_block_count = st->s1_block_count;
        s2->binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, pf_slot);
        s2->reserved = pf_expert;
        s2->valid = 1u;
        s2->sync_enabled = cfg->s2_call.valid != 0u;
    }
    __moe_pipeline_publish(
        &s1->csr_prepared_reserved, MOE_PIPELINE_INIT_COOKIE);
}

static inline void __moe_initialize_slot_if_needed(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    asm volatile("fence rw, rw" ::: "memory");
    if (__moe_s1_dma_ctrl(blk)->csr_prepared_reserved !=
        MOE_PIPELINE_INIT_COOKIE) {
        __moe_initialize_slot_state(blk, cfg, st);
    }
}

static inline void __moe_initialize_next_slot(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    volatile uint32_t *state =
        (volatile uint32_t *)(uintptr_t)st->active_state_l1_addr;
    uint32_t active_idx = (MOE_DYN_CTRL_CLUSTER(cfg->ctrl) == 0u) ?
        MOE_DYN_RT_C2_ACTIVE_SLOTS : MOE_DYN_RT_C3_ACTIVE_SLOTS;
    if (MOE_DYN_CTRL_SLOT_ID(cfg->ctrl) + 1u >= state[active_idx]) return;

    __snax_bingo_kernel_moe_dynamic_expert_block_args_t next_blk = *blk;
    next_blk.task_arg_addr += BINGO_MOE_DYNAMIC_ARG_SLOT_BYTES;
    next_blk.pipeline_ctrl_addr += MOE_PIPELINE_CTRL_SLOT_BYTES;
    next_blk.block_idx = 0u;
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *next_cfg =
        (const __snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        next_blk.task_arg_addr;
    __moe_initialize_slot_if_needed(&next_blk, next_cfg, st);
}

static inline void __moe_submit_token_gather(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    const uint16_t *token_refs =
        (const uint16_t *)(uintptr_t)st->token_refs_addr;
    uint32_t token_ref_start =
        cfg->expert_id * st->max_tokens_per_expert + cfg->token_ref_start;
    uint32_t repeats = st->A_token_bytes / MOE_BANK_A_TOKEN_TILE_BYTES;

    for (uint32_t local_t = 0u; local_t < cfg->ntokens; local_t++) {
        uint16_t token_ref = token_refs[token_ref_start + local_t];
        uint32_t token_id = BINGO_MOE_TOKEN_REF_TOKEN(token_ref);
        uint64_t src = st->input_A_l3_base +
            (uint64_t)token_id * (uint64_t)st->A_row_stride;
        uint64_t dst = __moe_dyn_l1_wide(__moe_bank_a_addr(
            __moe_dyn_input_base(cfg, st), local_t, st->A_token_bytes));
        snrt_dma_start_2d_wideptr(
            dst, src, MOE_BANK_A_TOKEN_TILE_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_A_TOKEN_TILE_BYTES, repeats);
    }
}

static inline void __moe_wait_token_gather(void)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START);
    snrt_dma_wait_all();
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_gather_s1(void *arg)
{
    __snax_bingo_kernel_moe_dynamic_expert_block_args_t *node =
        (__snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        node->task_arg_addr;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        node->static_arg_addr;

    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        __moe_initialize_slot_state(node, cfg, st);
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    MOE_INDIV_PRINT(
        "[INDIV_BEGIN] C%u slot=%u eid=%u start=%u ntok=%u "
        "shape_s1=%u shape_s3=%u skip=%u%u%u%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        cfg->token_ref_start, cfg->ntokens,
        MOE_DYN_CTRL_SHAPE_S1(cfg->ctrl), MOE_DYN_CTRL_SHAPE_S3(cfg->ctrl),
        MOE_DYN_CTRL_SKIP_S1(cfg->ctrl), MOE_DYN_CTRL_SKIP_S2(cfg->ctrl),
        MOE_DYN_CTRL_SKIP_S3(cfg->ctrl), MOE_DYN_CTRL_SKIP_S4(cfg->ctrl));

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    __moe_submit_token_gather(cfg, st);
    __moe_initialize_slot_state(node, cfg, st);
    __moe_prepare_s1_xdma_shape(node, cfg, st);
    __moe_wait_token_gather();
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_GATHER_DONE] C%u slot=%u eid=%u ntok=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        cfg->ntokens, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_GATHER_S1,
        MOE_PROFILE_RESOURCE_IDMA, 0u,
        cfg->ntokens * st->A_token_bytes, 0u, BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}

__attribute__((always_inline)) static inline void
__moe_transfer_s1_block(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __moe_s1_dma_ctrl_t *s1, uint32_t n)
{
    MOE_PROFILE_BEGIN(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_START);
    uint32_t block_bytes = s1->block_bytes;
    uint32_t weight_offset = n * block_bytes;
    uint32_t gate_dst = __moe_bank_weight_block_addr(
        s1->gate_dst_base, n, block_bytes);
    uint32_t up_dst = __moe_bank_weight_block_addr(
        s1->up_dst_base, n, block_bytes);
    uint32_t dma_binding = s1->binding;
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t configure_xdma =
        n == 0u &&
        !__moe_xdma_stage_is_prepared(blk, MOE_XDMA_PREPARED_S1);
    __moe_dyn_copy_pair_2d(
        dma_binding,
        __moe_dyn_l1_wide(gate_dst),
        s1->gate_src_base + weight_offset,
        __moe_dyn_l1_wide(up_dst),
        s1->up_src_base + weight_offset,
        block_bytes, configure_xdma);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_LOAD_S1_DONE] C%u slot=%u eid=%u block=%u dma=%u "
        "bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, dma_binding, block_bytes, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_END);
    MOE_PROFILE_COMMIT(
        (void *)blk, cfg, profile, MOE_PROFILE_STAGE_LOAD_S1,
        __moe_profile_dma_resource(dma_binding), n,
        2u * block_bytes, 0u, BINGO_RET_SUCC);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_config_s1_block0(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (!__moe_dyn_slot_active_this_round(cfg, st) ||
        cfg->s1_call[0].valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    const __snax_bingo_moe_dyn_s1_call_args_t *call = &cfg->s1_call[0];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    __moe_bank_configure_mode0(
        __moe_dyn_input_base(cfg, st),
        __moe_bank_weight_block_addr(
            st->l1_b_gate_addr, 0u, st->indiv_B_block_stride),
        __moe_bank_weight_block_addr(
            st->l1_b_up_addr, 0u, st->indiv_B_block_stride),
        __moe_bank_mode0_output_addr(
            __moe_dyn_intermediate_base(cfg, st), 0u, 0u,
            st->indiv_N_per_block,
            st->s1_block_count),
        st->indiv_K1, call->N, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    return BINGO_RET_SUCC;
}

__attribute__((always_inline)) static inline void
__moe_run_s1_block(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t n,
    uint32_t configure_block0,
    uint32_t s4_csr_layout)
{
    MOE_PROFILE_BEGIN(profile);
    const __snax_bingo_moe_dyn_s1_call_args_t *call = &cfg->s1_call[n];
    if (call->valid == 0u) {
        MOE_PROFILE_COMMIT(
            (void *)blk, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S1,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    if (n == 0u && configure_block0 != 0u) {
        __moe_bank_configure_mode0(
            __moe_dyn_input_base(cfg, st),
            __moe_bank_weight_block_addr(
                st->l1_b_gate_addr, n, st->indiv_B_block_stride),
            __moe_bank_weight_block_addr(
                st->l1_b_up_addr, n, st->indiv_B_block_stride),
            __moe_bank_mode0_output_addr(
                __moe_dyn_intermediate_base(cfg, st), 0u, n,
                st->indiv_N_per_block, st->s1_block_count),
            st->indiv_K1, call->N, call->array_shape,
            st->rescale_mult, st->rescale_shift);
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    if (n + 1u < st->s1_block_count) {
        uint32_t next_block = n + 1u;
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
        csrw_ss(BASE_PTR_READER_1_LOW,
            __moe_bank_weight_block_addr(
                st->l1_b_gate_addr, next_block, st->indiv_B_block_stride));
        csrw_ss(BASE_PTR_READER_2_LOW,
            __moe_bank_weight_block_addr(
                st->l1_b_up_addr, next_block, st->indiv_B_block_stride));
        csrw_ss(BASE_PTR_WRITER_0_LOW,
            __moe_bank_mode0_output_addr(
                __moe_dyn_intermediate_base(cfg, st), 0u, next_block,
                st->indiv_N_per_block, st->s1_block_count));
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    } else {
        __moe_prepare_after_s1(blk, cfg, st, s4_csr_layout);
    }
    moe_wait_dual_vc_and_streamer();
    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S1_DONE] C%u slot=%u eid=%u block=%u shape=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, call->array_shape, call->N, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_END);
    MOE_PROFILE_COMMIT(
        (void *)blk, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S1,
        MOE_PROFILE_RESOURCE_VERSACORE, n,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, BINGO_RET_SUCC);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_load_s1_stage(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    if (!__moe_dyn_slot_active_this_round(cfg, st) || s1->valid == 0u) {
        return BINGO_RET_SUCC;
    }
    uint32_t block_count = s1->block_count;
    for (uint32_t n = 0u; n < block_count; n++) {
        if (n == 1u) {
            __moe_pipeline_wait_cookie(
                &s1->csr_prepared_reserved, MOE_PIPELINE_INIT_COOKIE,
                1u);
        } else if (n >= 2u) {
            __moe_pipeline_wait(&sync->compute_done, n - 1u);
        }
        __moe_transfer_s1_block(blk, cfg, s1, n);
        __moe_pipeline_publish(&sync->prefetch_done, n + 1u);
    }
    __moe_prepare_s4pf_xdma_shape(blk, cfg, st);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_compute_s1_stage(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    if (!__moe_dyn_slot_active_this_round(cfg, st) || s1->valid == 0u) {
        return BINGO_RET_SUCC;
    }
    /* Read every dynamic call record before allowing load1 to enter the B-bank
     * path. This preserves the guarded handoff that avoided the earlier
     * task-argument/TCDM conflict while keeping load1 overlapped with compute0. */
    uint32_t call_checksum = 0u;
    for (uint32_t n = 0u; n < s1->block_count; n++) {
        call_checksum ^= cfg->s1_call[n].valid;
        call_checksum ^= cfg->s1_call[n].N;
        call_checksum ^= cfg->s1_call[n].array_shape;
    }
    asm volatile("" : : "r"(call_checksum) : "memory");
    __moe_pipeline_publish(
        &s1->csr_prepared_reserved,
        MOE_PIPELINE_INIT_COOKIE | MOE_PIPELINE_COMPUTE_READY_BIT);

    for (uint32_t n = 0u; n < s1->block_count; n++) {
        __moe_pipeline_wait(&sync->prefetch_done, n + 1u);
        __moe_run_s1_block(
            blk, cfg, st, n, 0u, MOE_S4_CSR_LAYOUT_BLOCK_SYNC);
        if (n + 2u < s1->block_count) {
            __moe_pipeline_publish(&sync->compute_done, n + 1u);
        }
    }

    /* The compute consumer cannot finish before the final producer transfer,
     * so it is the unique safe point to recycle the counters for S2. */
    asm volatile("fence rw, rw" ::: "memory");
    sync->compute_done = 0u;
    sync->prefetch_done = 0u;
    asm volatile("fence rw, rw" ::: "memory");
    return BINGO_RET_SUCC;
}
