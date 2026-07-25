// Discrete per-block comparison ABI. Production DFGs use opt_* stage APIs.
#pragma once

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    uint32_t n = blk->block_idx;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    const __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    if (s1->valid == 0u || n >= s1->block_count) {
        MOE_PROFILE_BEGIN(profile);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_LOAD_S1,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    __moe_transfer_s1_block(blk, cfg, s1, n);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (__moe_dyn_slot_active_this_round(cfg, st)) {
        __moe_run_s1_block(blk, cfg, st, blk->block_idx, 1u, 0u);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block_pc(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (__moe_dyn_slot_active_this_round(cfg, st)) {
        __moe_run_s1_block(blk, cfg, st, blk->block_idx, 0u, 0u);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_load_down_block(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    uint32_t ctrl = cfg->ctrl;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        return BINGO_RET_SUCC;
    }
    if (MOE_DYN_CTRL_SKIP_S3(ctrl) != 0u) {
        MOE_PROFILE_BEGIN(profile);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_LOAD_S3,
            MOE_PROFILE_RESOURCE_NONE, blk->block_idx, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

    uint32_t block_bytes = st->indiv_down_B_block_stride;
    uint64_t down_src = st->indiv_down_B_l3 +
        (uint64_t)cfg->expert_id * st->indiv_down_B_expert_stride;
    __moe_transfer_s3_block(
        blk, cfg, st, blk->block_idx, down_src, block_bytes,
        st->s3_block_count, MOE_DYN_CTRL_DMA_S3(ctrl));
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_configure_down_block0(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (__moe_dyn_slot_active_this_round(cfg, st) &&
        cfg->s3_call[0].valid != 0u) {
        __moe_configure_s3_block0(
            blk, cfg, st, MOE_S4_CSR_LAYOUT_SEQUENTIAL);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_block(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (__moe_dyn_slot_active_this_round(cfg, st)) {
        __moe_run_s3_block(
            blk, cfg, st, blk->block_idx, 1u,
            MOE_S4_CSR_LAYOUT_SEQUENTIAL);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_block_pc(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (__moe_dyn_slot_active_this_round(cfg, st)) {
        __moe_run_s3_block(
            blk, cfg, st, blk->block_idx, 0u,
            MOE_S4_CSR_LAYOUT_SEQUENTIAL);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);
    uint32_t slot = MOE_DYN_DMA_SLOT_S4_PREFETCH;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    __moe_initialize_next_slot(blk, cfg, st);
    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        if (__moe_dyn_has_output(cfg) != 0u &&
            s2->store_prepared == 0u) {
            __moe_dyn_prepare_store_xdma(cfg, st);
            __moe_pipeline_publish(&s2->store_prepared, 1u);
        }
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_START);
    uint32_t expert_id = MOE_DYN_DMA_EID(cfg->dma_slot_eids, slot);
    uint32_t weight_bytes = st->indiv_B_expert_stride;
    uint32_t dma_binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint64_t gate_src = st->indiv_gate_B_l3 +
        (uint64_t)expert_id * st->indiv_B_expert_stride;
    uint64_t up_src = st->indiv_up_B_l3 +
        (uint64_t)expert_id * st->indiv_B_expert_stride;
    uint32_t store_prepared_early = 0u;
    for (uint32_t n = 0u; n < st->s1_block_count; n++) {
        uint32_t src_offset = n * st->indiv_B_block_stride;
        uint32_t gate_dst = __moe_bank_weight_block_addr(
            st->l1_b_gate_addr, n, st->indiv_B_block_stride);
        uint32_t up_dst = __moe_bank_weight_block_addr(
            st->l1_b_up_addr, n, st->indiv_B_block_stride);
        uint32_t configure_xdma =
            n == 0u &&
            !__moe_xdma_stage_is_prepared(
                blk, MOE_XDMA_PREPARED_S4PF);
        if (n + 1u == st->s1_block_count &&
            __moe_dyn_has_output(cfg) != 0u) {
            __moe_dyn_2d_pair_pending_t pending;
            __moe_dyn_start_pair_2d(
                dma_binding,
                __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                __moe_dyn_l1_wide(up_dst), up_src + src_offset,
                st->indiv_B_block_stride, configure_xdma, &pending);
            __moe_dyn_prepare_store_xdma(cfg, st);
            __moe_dyn_wait_pair_2d(&pending);
            __moe_pipeline_publish(&s2->store_prepared, 1u);
            store_prepared_early = 1u;
        } else {
            __moe_dyn_copy_pair_2d(
                dma_binding,
                __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                __moe_dyn_l1_wide(up_dst), up_src + src_offset,
                st->indiv_B_block_stride, configure_xdma);
        }
    }
    if (__moe_dyn_has_output(cfg) != 0u &&
        store_prepared_early == 0u) {
        __moe_dyn_prepare_store_xdma(cfg, st);
        __moe_pipeline_publish(&s2->store_prepared, 1u);
    }
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_PREFETCH_S4_DONE] C%u slot=%u eid=%u target_eid=%u "
        "dma=%u bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        expert_id, MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot), weight_bytes,
        BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
        __moe_profile_dma_resource(dma_binding), 0u,
        2u * weight_bytes, 0u, BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_full(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (__moe_dyn_slot_active_this_round(cfg, st)) {
        __moe_run_s2_compute(
            blk, cfg, st, MOE_S4_CSR_LAYOUT_SEQUENTIAL);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_compute_down_full(void *arg)
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
    if (!__moe_dyn_slot_active_this_round(cfg, st)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_BEGIN(profile);
    const __snax_bingo_moe_dyn_s4_call_args_t *call = &cfg->s4_call;
    if (call->valid == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    uint32_t n_tiles =
        __moe_dyn_stage_block_n(call->N, st->s3_block_count);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t token_start = call->token_start;
    uint32_t token_step = __moe_dyn_shape_m(call->array_shape);
    uint32_t intermediate_base = __moe_dyn_intermediate_base(cfg, st);
    uint32_t output_base = __moe_dyn_output_base(cfg, st);
    for (uint32_t mt = 0u; mt < call->M; mt++) {
        uint32_t token = token_start + mt * token_step;
        for (uint32_t n = 0u; n < st->s3_block_count; n++) {
            if (mt == 0u && n == 0u &&
                !__moe_csr_stage_is_prepared(
                    blk, MOE_CSR_PREPARED_S4)) {
                __moe_bank_configure_mode1(
                    __moe_bank_mode0_output_addr(
                        intermediate_base, token, 0u,
                        st->indiv_N_per_block,
                        st->s1_block_count),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr, n,
                        st->indiv_down_B_block_stride),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr + 64u, n,
                        st->indiv_down_B_block_stride),
                    __moe_bank_mode1_output_addr(
                        output_base, token, n,
                        st->indiv_down_N_per_block, st->s3_block_count),
                    __moe_bank_mode1_output_addr(
                        output_base + 64u, token, n,
                        st->indiv_down_N_per_block, st->s3_block_count),
                    st->indiv_down_K1, n_tiles, call->array_shape,
                    st->rescale_mult, st->rescale_shift);
                __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S4);
            }
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
            moe_start_dual_vc_and_streamer();

            uint32_t next_n = n + 1u;
            uint32_t next_mt = mt;
            if (next_n == st->s3_block_count) {
                next_n = 0u;
                next_mt++;
            }
            if (next_mt < call->M) {
                uint32_t next_token = token_start +
                    next_mt * token_step;
                __moe_bank_patch_mode1_run_bases(
                    __moe_bank_mode0_output_addr(
                        intermediate_base, next_token, 0u,
                        st->indiv_N_per_block, st->s1_block_count),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr, next_n,
                        st->indiv_down_B_block_stride),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr + 64u, next_n,
                        st->indiv_down_B_block_stride),
                    __moe_bank_mode1_output_addr(
                        output_base, next_token, next_n,
                        st->indiv_down_N_per_block, st->s3_block_count),
                    __moe_bank_mode1_output_addr(
                        output_base + 64u, next_token, next_n,
                        st->indiv_down_N_per_block, st->s3_block_count));
            }
            moe_wait_dual_vc_and_streamer();
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
        }
    }
    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S4_DONE] C%u slot=%u eid=%u M=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        call->M, call->N, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
        MOE_PROFILE_RESOURCE_VERSACORE, 0u,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dynamic_expert_prepare_store_xdma_2d(void *arg)
{
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (!__moe_dyn_slot_active_this_round(cfg, st)) return BINGO_RET_SUCC;
    if (__moe_dyn_has_output(cfg) == 0u) {
        return BINGO_RET_SUCC;
    }
    __moe_s2_prefetch_ctrl_t *s2 = __moe_s2_prefetch_ctrl(blk);
    if (s2->store_prepared != 0u) return BINGO_RET_SUCC;
    __moe_dyn_prepare_store_xdma(cfg, st);
    __moe_pipeline_publish(&s2->store_prepared, 1u);
    return BINGO_RET_SUCC;
}
