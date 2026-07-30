// Optimized S3 down-projection pipeline and shared block primitives.
#pragma once

__attribute__((always_inline)) static inline void
__moe_transfer_s3_block(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t n,
    uint64_t down_src,
    uint32_t block_bytes,
    uint32_t block_count,
    uint32_t dma_binding)
{
    MOE_PROFILE_BEGIN(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_DOWN_START);
    uint32_t left_offset = n * block_bytes;
    uint32_t right_offset = (block_count + n) * block_bytes;
    uint32_t left_dst = __moe_bank_down_weight_block_addr(
        st->l1_b_down_addr, n, block_bytes);
    uint32_t right_dst = __moe_bank_down_weight_block_addr(
        st->l1_b_down_addr + 64u, n, block_bytes);
    uint32_t configure_xdma =
        n == 0u &&
        !__moe_xdma_stage_is_prepared(blk, MOE_XDMA_PREPARED_S3);

    MOE_PROFILE_RESOURCE_BEGIN(profile);
    __moe_dyn_copy_pair_2d(
        dma_binding,
        __moe_dyn_l1_wide(left_dst), down_src + left_offset,
        __moe_dyn_l1_wide(right_dst), down_src + right_offset,
        block_bytes, configure_xdma);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_LOAD_S3_DONE] C%u slot=%u eid=%u block=%u dma=%u "
        "bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, dma_binding, block_bytes, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_LOAD_DOWN_END);
    MOE_PROFILE_COMMIT(
        (void *)blk, cfg, profile, MOE_PROFILE_STAGE_LOAD_S3,
        __moe_profile_dma_resource(dma_binding), n,
        2u * block_bytes, 0u, BINGO_RET_SUCC);
}

__attribute__((always_inline)) static inline void
__moe_configure_s3_block0(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t s4_csr_layout)
{
    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    if (s4_csr_layout != MOE_S4_CSR_LAYOUT_SEQUENTIAL) {
        sync->sync_enabled = 0u;
        sync->reserved = 0u;
        asm volatile("fence rw, rw" ::: "memory");
    }

    sync->compute_done = 0u;
    sync->prefetch_done = 0u;
    __moe_pipeline_publish(
        &__moe_s1_dma_ctrl(blk)->csr_prepared_reserved,
        MOE_PIPELINE_S3_SYNC_COOKIE);

    if (__moe_csr_stage_is_prepared(blk, MOE_CSR_PREPARED_S3)) return;

    const __snax_bingo_moe_dyn_s3_call_args_t *call = &cfg->s3_call[0];
    __moe_bank_configure_mode1(
        __moe_dyn_intermediate_base(cfg, st),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr, 0u, st->indiv_down_B_block_stride),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr + 64u, 0u, st->indiv_down_B_block_stride),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st), 0u, 0u,
            st->indiv_down_N_per_block, st->s3_block_count),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st) + 64u, 0u, 0u,
            st->indiv_down_N_per_block, st->s3_block_count),
        st->indiv_down_K1, call->N, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S3);
}

__attribute__((always_inline)) static inline void
__moe_run_s3_block(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t n,
    uint32_t configure_block0,
    uint32_t s4_csr_layout)
{
    const __snax_bingo_moe_dyn_s3_call_args_t *call = &cfg->s3_call[n];
    MOE_PROFILE_BEGIN(profile);
    if (call->valid == 0u) {
        MOE_PROFILE_COMMIT(
            (void *)blk, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S3,
            MOE_PROFILE_RESOURCE_NONE, n, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return;
    }

    if (n == 0u && configure_block0 != 0u &&
        !__moe_csr_stage_is_prepared(blk, MOE_CSR_PREPARED_S3)) {
        __moe_configure_s3_block0(blk, cfg, st, s4_csr_layout);
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();

    if (n + 1u < st->s3_block_count) {
        uint32_t next_block = n + 1u;
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
        csrw_ss(BASE_PTR_READER_1_LOW,
            __moe_bank_down_weight_block_addr(
                st->l1_b_down_addr, next_block,
                st->indiv_down_B_block_stride));
        csrw_ss(BASE_PTR_READER_2_LOW,
            __moe_bank_down_weight_block_addr(
                st->l1_b_down_addr + 64u, next_block,
                st->indiv_down_B_block_stride));
        csrw_ss(BASE_PTR_WRITER_0_LOW,
            __moe_bank_mode1_output_addr(
                __moe_dyn_output_base(cfg, st), 0u, next_block,
                st->indiv_down_N_per_block, st->s3_block_count));
        csrw_ss(BASE_PTR_WRITER_1_LOW,
            __moe_bank_mode1_output_addr(
                __moe_dyn_output_base(cfg, st) + 64u, 0u, next_block,
                st->indiv_down_N_per_block, st->s3_block_count));
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    } else {
        (void)__moe_prepare_s4_csr(blk, cfg, st, s4_csr_layout);
    }

    moe_wait_dual_vc_and_streamer();
    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S3_DONE] C%u slot=%u eid=%u block=%u shape=%u N=%u "
        "rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        n, call->array_shape, call->N, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_END);
    MOE_PROFILE_COMMIT(
        (void *)blk, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S3,
        MOE_PROFILE_RESOURCE_VERSACORE, n,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, BINGO_RET_SUCC);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_load_s3_stage(void *arg)
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
    if (!__moe_dyn_slot_active_this_round(cfg, st) ||
        MOE_DYN_CTRL_SKIP_S3(ctrl) != 0u ||
        cfg->s3_call[0].valid == 0u) {
        return BINGO_RET_SUCC;
    }

    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    __moe_pipeline_wait_cookie(
        &s1->csr_prepared_reserved, MOE_PIPELINE_S3_SYNC_COOKIE, 0u);

    uint32_t block_count = st->s3_block_count;
    uint32_t block_bytes = st->indiv_down_B_block_stride;
    uint32_t dma_binding = MOE_DYN_CTRL_DMA_S3(ctrl);
    uint32_t weight_eid = MOE_DYN_DMA_EID(
        cfg->dma_slot_eids, MOE_DYN_DMA_SLOT_S3);
    uint64_t down_src = st->indiv_down_B_l3 +
        (uint64_t)weight_eid * st->indiv_down_B_expert_stride;
    for (uint32_t n = 0u; n < block_count; n++) {
        if (n == 1u) {
            __moe_pipeline_wait_cookie(
                &s1->csr_prepared_reserved,
                MOE_PIPELINE_S3_SYNC_COOKIE, 1u);
        } else if (n >= 2u) {
            __moe_pipeline_wait(&sync->compute_done, n - 1u);
        }

        __moe_transfer_s3_block(
            blk, cfg, st, n, down_src, block_bytes,
            block_count, dma_binding);
        __moe_pipeline_publish(&sync->prefetch_done, n + 1u);
    }

    /* S1's early S4 shape is overwritten by S3 weight traffic.  S3 DMA is now
     * drained while the final S3 compute remains active, so this is the last
     * safe point to stage the runtime-selected S4 xDMA shape.  Initialize the
     * next slot here as well: this DM-core tail is hidden by the remaining S3
     * compute instead of delaying S4's first prefetch block. */
    __moe_prepare_s4pf_xdma_shape(blk, cfg, st);
    __moe_initialize_next_slot(blk, cfg, st);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_config_s3_block0(void *arg)
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
            blk, cfg, st, MOE_S4_CSR_LAYOUT_PHASE_BATCHED);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_compute_s3_stage(void *arg)
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
    if (!__moe_dyn_slot_active_this_round(cfg, st) ||
        MOE_DYN_CTRL_SKIP_S3(ctrl) != 0u ||
        cfg->s3_call[0].valid == 0u) {
        return BINGO_RET_SUCC;
    }

    uint32_t call_checksum = 0u;
    for (uint32_t n = 0u; n < st->s3_block_count; n++) {
        call_checksum ^= cfg->s3_call[n].valid;
        call_checksum ^= cfg->s3_call[n].N;
        call_checksum ^= cfg->s3_call[n].array_shape;
    }
    asm volatile("" : : "r"(call_checksum) : "memory");

    __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    __moe_pipeline_publish(
        &s1->csr_prepared_reserved,
        MOE_PIPELINE_S3_SYNC_COOKIE | MOE_PIPELINE_COMPUTE_READY_BIT);

    for (uint32_t n = 0u; n < st->s3_block_count; n++) {
        __moe_pipeline_wait(&sync->prefetch_done, n + 1u);
        __moe_run_s3_block(
            blk, cfg, st, n, 0u, MOE_S4_CSR_LAYOUT_PHASE_BATCHED);
        __moe_pipeline_publish(&sync->compute_done, n + 1u);
    }
    return BINGO_RET_SUCC;
}
