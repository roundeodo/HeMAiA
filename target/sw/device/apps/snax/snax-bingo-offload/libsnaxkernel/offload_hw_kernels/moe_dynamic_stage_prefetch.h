// Internal dynamic-MoE module; include through moe_dynamic.h.
#pragma once

__attribute__((always_inline)) static inline void
__moe_s2pf_submit_idma_phase(
    const __moe_s2_prefetch_ctrl_t *s2,
    uint32_t phase,
    uint32_t side)
{
    uint32_t repeats = s2->block_bytes / MOE_BANK_WEIGHT_ROW_BYTES;
    for (uint32_t block = phase; block < s2->block_count; block += 2u) {
        uint64_t dst = 0u;
        uint64_t src = 0u;
        __moe_dyn_s2pf_single_address(s2, block, side, &dst, &src);
        snrt_dma_start_2d_wideptr(
            dst, src, MOE_BANK_WEIGHT_ROW_BYTES,
            MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
    }
}

__attribute__((always_inline)) static inline int32_t
__moe_s2pf_start_both_phase(
    const __moe_s2_prefetch_ctrl_t *s2,
    uint32_t phase)
{
    int32_t previous = __moe_dyn_xdma_start_remote_begin();
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    __moe_s2pf_submit_idma_phase(s2, phase, 0u);
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    return __moe_dyn_xdma_start_remote_commit(previous);
}

__attribute__((always_inline)) static inline void
__moe_s2pf_prepare_next_xdma_phase_shape(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __moe_s2_prefetch_ctrl_t *s2)
{
    uint32_t phase0_blocks = __moe_s4_blocks_in_phase(s2->block_count, 0u);
    uint32_t phase1_blocks = __moe_s4_blocks_in_phase(s2->block_count, 1u);
    if (phase0_blocks != phase1_blocks) {
        __moe_prepare_s2pf_xdma_phase_shape(blk, 1u);
    }
}

__attribute__((always_inline)) static inline void
__moe_s2pf_run_both(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    const __moe_s2_prefetch_ctrl_t *s2)
{
    int32_t last_xdma_task = __moe_s2pf_start_both_phase(s2, 0u);
    if (__moe_s4_blocks_in_phase(s2->block_count, 1u) != 0u) {
        __moe_s2pf_prepare_next_xdma_phase_shape(blk, s2);
        __moe_dyn_prepare_s2pf_both_xdma_address(s2, 1u);
        last_xdma_task = __moe_s2pf_start_both_phase(s2, 1u);
    }
    __moe_dyn_prepare_after_s2pf_xdma(blk, cfg, st);
    __moe_dyn_wait_both_2d(last_xdma_task);
}

__attribute__((always_inline)) static inline int32_t
__moe_s2pf_submit_xdma_phase(
    const __moe_s2_prefetch_ctrl_t *s2,
    uint32_t phase)
{
    (void)__moe_dyn_start_single_2d_preloaded_xdma();
    __moe_dyn_prepare_s2pf_single_xdma_address(s2, phase, 1u);
    return __moe_dyn_start_single_2d_preloaded_xdma();
}

__attribute__((always_inline)) static inline void
__moe_s2pf_run_xdma(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    const __moe_s2_prefetch_ctrl_t *s2)
{
    int32_t last_xdma_task = __moe_s2pf_submit_xdma_phase(s2, 0u);
    if (__moe_s4_blocks_in_phase(s2->block_count, 1u) != 0u) {
        __moe_s2pf_prepare_next_xdma_phase_shape(blk, s2);
        __moe_dyn_prepare_s2pf_single_xdma_address(s2, 1u, 0u);
        last_xdma_task = __moe_s2pf_submit_xdma_phase(s2, 1u);
    }
    __moe_dyn_prepare_after_s2pf_xdma(blk, cfg, st);
    __moe_dyn_wait_single_2d_xdma(last_xdma_task);
}

__attribute__((always_inline)) static inline void
__moe_s2pf_run_idma(const __moe_s2_prefetch_ctrl_t *s2)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    for (uint32_t phase = 0u; phase < 2u; phase++) {
        __moe_s2pf_submit_idma_phase(s2, phase, 0u);
        __moe_s2pf_submit_idma_phase(s2, phase, 1u);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    __moe_dyn_wait_single_2d_idma();
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_prefetch_s2(void *arg)
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
    MOE_PROFILE_BEGIN(profile);
    if (s2->valid == 0u) {
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S2,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    if (MOE_DYN_CTRL_S2PF_EARLY(cfg->ctrl) == 0u &&
        MOE_DYN_CTRL_S2PF_RUNTIME_RELEASE(cfg->ctrl) != 0u &&
        __moe_s1_dma_ctrl(blk)->valid != 0u) {
        __moe_s1_dma_ctrl_t *s1 = __moe_s1_dma_ctrl(blk);
        __moe_pipeline_wait(&s1->compute_done, s1->block_count);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S2_START);
    uint32_t half_bytes = s2->half_bytes;
    uint32_t dma_binding = s2->binding;
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    if (dma_binding == MOE_DYN_DMA_BOTH) {
        __moe_s2pf_run_both(blk, cfg, st, s2);
    } else if (dma_binding == MOE_DYN_DMA_XDMA) {
        __moe_s2pf_run_xdma(blk, cfg, st, s2);
    } else {
        __moe_s2pf_run_idma(s2);
    }
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_PREFETCH_S2_DONE] C%u slot=%u eid=%u target_eid=%u "
        "dma=%u bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        s2->reserved, dma_binding, half_bytes, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S2_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S2,
        __moe_profile_dma_resource(dma_binding), 0u,
        2u * half_bytes, 0u, BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}

__attribute__((always_inline)) static inline void
__moe_s4pf_phase_marker(uint32_t phase, uint32_t begin)
{
    if (phase == 0u) {
        if (begin != 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE0_START);
        } else {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE0_END);
        }
    } else {
        if (begin != 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE1_START);
        } else {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE1_END);
        }
    }
}

__attribute__((always_inline)) static inline void
__moe_s4pf_prepare_store(
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t prepare_store)
{
    if (prepare_store != 0u && __moe_dyn_has_output(cfg) != 0u) {
        __moe_dyn_prepare_store_xdma(cfg, st);
    }
}

__attribute__((always_inline)) static inline void
__moe_s4pf_phase_idma(
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t phase,
    uint32_t prepare_store)
{
    uint32_t phase_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, phase);
    if (phase_blocks == 0u) {
        __moe_s4pf_prepare_store(cfg, st, prepare_store);
        return;
    }
    __moe_s4pf_phase_marker(phase, 1u);
    uint32_t block_bytes = st->indiv_B_block_stride;
    uint32_t repeats = block_bytes / MOE_BANK_WEIGHT_ROW_BYTES;
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    for (uint32_t n = phase; n < st->s1_block_count; n += 2u) {
        snrt_dma_start_2d_wideptr(
            __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
                st->l1_b_gate_addr, n, block_bytes)),
            gate_src + (uint64_t)n * block_bytes,
            MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_TCDM_ROW_BYTES,
            MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        snrt_dma_start_2d_wideptr(
            __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
                st->l1_b_up_addr, n, block_bytes)),
            up_src + (uint64_t)n * block_bytes,
            MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_TCDM_ROW_BYTES,
            MOE_BANK_WEIGHT_ROW_BYTES, repeats);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    __moe_s4pf_prepare_store(cfg, st, prepare_store);
    __moe_dyn_wait_pair_2d_idma();
    __moe_s4pf_phase_marker(phase, 0u);
}

__attribute__((always_inline)) static inline void
__moe_s4pf_phase_xdma(
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t phase,
    uint32_t preload_next_phase,
    uint32_t prepare_store)
{
    uint32_t phase_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, phase);
    if (phase_blocks == 0u) {
        __moe_s4pf_prepare_store(cfg, st, prepare_store);
        return;
    }
    __moe_s4pf_phase_marker(phase, 1u);
    uint64_t gate_dst = 0u;
    uint64_t gate_src_phase = 0u;
    uint64_t up_dst = 0u;
    uint64_t up_src_phase = 0u;
    __moe_dyn_s4pf_phase_addresses(
        st, gate_src, up_src, phase,
        &gate_dst, &gate_src_phase, &up_dst, &up_src_phase);
    int32_t last_task = __moe_dyn_start_pair_2d_preloaded_xdma(
        up_dst, up_src_phase);
    if (preload_next_phase != 0u) {
        uint32_t next_phase = phase ^ 1u;
        __moe_dyn_s4pf_phase_addresses(
            st, gate_src, up_src, next_phase,
            &gate_dst, &gate_src_phase, &up_dst, &up_src_phase);
        xdma_memcpy_fast_set_addresses(gate_src_phase, gate_dst);
    }
    __moe_s4pf_prepare_store(cfg, st, prepare_store);
    __moe_dyn_wait_pair_2d_xdma(last_task);
    __moe_s4pf_phase_marker(phase, 0u);
}

__attribute__((always_inline)) static inline void
__moe_s4pf_phase_both(
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t phase,
    uint32_t preload_next_phase,
    uint32_t prepare_store)
{
    uint32_t phase_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, phase);
    if (phase_blocks == 0u) {
        __moe_s4pf_prepare_store(cfg, st, prepare_store);
        return;
    }
    __moe_s4pf_phase_marker(phase, 1u);
    int32_t previous = __moe_dyn_xdma_start_remote_begin();
    uint32_t block_bytes = st->indiv_B_block_stride;
    uint32_t repeats = block_bytes / MOE_BANK_WEIGHT_ROW_BYTES;
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
    for (uint32_t n = phase; n < st->s1_block_count; n += 2u) {
        snrt_dma_start_2d_wideptr(
            __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
                st->l1_b_gate_addr, n, block_bytes)),
            gate_src + (uint64_t)n * block_bytes,
            MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_TCDM_ROW_BYTES,
            MOE_BANK_WEIGHT_ROW_BYTES, repeats);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
    int32_t xdma_task = __moe_dyn_xdma_start_remote_commit(previous);
    if (preload_next_phase != 0u) {
        uint32_t next_phase = phase ^ 1u;
        uint64_t gate_dst = 0u;
        uint64_t gate_src_phase = 0u;
        uint64_t up_dst = 0u;
        uint64_t up_src_phase = 0u;
        __moe_dyn_s4pf_phase_addresses(
            st, gate_src, up_src, next_phase,
            &gate_dst, &gate_src_phase, &up_dst, &up_src_phase);
        xdma_memcpy_fast_set_addresses(up_src_phase, up_dst);
    }
    __moe_s4pf_prepare_store(cfg, st, prepare_store);
    __moe_dyn_wait_both_2d(xdma_task);
    __moe_s4pf_phase_marker(phase, 0u);
}

__attribute__((always_inline)) static inline void
__moe_s4pf_wait_compute(
    __moe_s2_prefetch_ctrl_t *s2,
    uint32_t compute_active,
    uint32_t step)
{
    if (step != 0u && compute_active != 0u) {
        __moe_pipeline_wait(&s2->sync_enabled, step);
    }
}

__attribute__((always_inline)) static inline void
__moe_s4pf_run_idma(
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    __moe_s2_prefetch_ctrl_t *s2,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t initial_phase,
    uint32_t compute_active)
{
    for (uint32_t step = 0u; step < 2u; step++) {
        __moe_s4pf_wait_compute(s2, compute_active, step);
        uint32_t phase = __moe_s4_phase_at_step(initial_phase, step);
        __moe_s4pf_phase_idma(
            cfg, st, gate_src, up_src, phase, step == 1u);
        __moe_pipeline_publish(&s2->reserved, step + 1u);
    }
}

__attribute__((always_inline)) static inline void
__moe_s4pf_run_xdma(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    __moe_s2_prefetch_ctrl_t *s2,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t initial_phase,
    uint32_t compute_active)
{
    uint32_t first_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, initial_phase);
    uint32_t second_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, initial_phase ^ 1u);
    uint32_t equal_phase_sizes = first_blocks == second_blocks;
    for (uint32_t step = 0u; step < 2u; step++) {
        __moe_s4pf_wait_compute(s2, compute_active, step);
        uint32_t phase = __moe_s4_phase_at_step(initial_phase, step);
        if (step != 0u && equal_phase_sizes == 0u &&
            second_blocks != 0u) {
            __moe_dyn_prepare_s4pf_xdma_phase(
                blk, st, gate_src, up_src,
                MOE_DYN_DMA_XDMA, phase);
        }
        __moe_s4pf_phase_xdma(
            cfg, st, gate_src, up_src, phase,
            step == 0u && equal_phase_sizes != 0u && second_blocks != 0u,
            step == 1u);
        __moe_pipeline_publish(&s2->reserved, step + 1u);
    }
}

__attribute__((always_inline)) static inline void
__moe_s4pf_run_both(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    __moe_s2_prefetch_ctrl_t *s2,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t initial_phase,
    uint32_t compute_active)
{
    uint32_t first_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, initial_phase);
    uint32_t second_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, initial_phase ^ 1u);
    uint32_t equal_phase_sizes = first_blocks == second_blocks;
    for (uint32_t step = 0u; step < 2u; step++) {
        __moe_s4pf_wait_compute(s2, compute_active, step);
        uint32_t phase = __moe_s4_phase_at_step(initial_phase, step);
        if (step != 0u && equal_phase_sizes == 0u &&
            second_blocks != 0u) {
            __moe_dyn_prepare_s4pf_xdma_phase(
                blk, st, gate_src, up_src,
                MOE_DYN_DMA_BOTH, phase);
        }
        __moe_s4pf_phase_both(
            cfg, st, gate_src, up_src, phase,
            step == 0u && equal_phase_sizes != 0u && second_blocks != 0u,
            step == 1u);
        __moe_pipeline_publish(&s2->reserved, step + 1u);
    }
}

/* Production S4 prefetch: transfer all next-S1 blocks in one bank phase before
 * exchanging phase ownership with the S4 compute worker. */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_prefetch_s4(void *arg)
{
    MOE_PROFILE_BEGIN(profile);
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
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) != 0u ||
        cfg->s3_call[0].valid == 0u) {
        __moe_initialize_next_slot(blk, cfg, st);
    }

    uint32_t initial_phase = __moe_s4_block_initial_phase(st);
    uint32_t compute_active = cfg->s4_call.valid != 0u &&
        cfg->s4_call.M != 0u;
    uint32_t prefetch_valid = MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot);
    uint32_t dma_binding = prefetch_valid != 0u ?
        MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot) : 0u;

    if (prefetch_valid == 0u) {
        if (__moe_dyn_has_output(cfg) != 0u) {
            __moe_dyn_prepare_store_xdma(cfg, st);
            __moe_pipeline_publish(&s2->store_prepared, 1u);
        }
        __moe_pipeline_publish(&s2->reserved, 2u);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_START);
    uint32_t weight_eid = MOE_DYN_DMA_EID(cfg->dma_slot_eids, slot);
    uint32_t weight_bytes = st->indiv_B_expert_stride;
    uint64_t gate_src = st->indiv_gate_B_l3 +
        (uint64_t)weight_eid * weight_bytes;
    uint64_t up_src = st->indiv_up_B_l3 +
        (uint64_t)weight_eid * weight_bytes;

    /* S3 load may finish two blocks before S3 compute. Wait until the
     * penultimate S3 block releases the phase selected for the first S4 DMA;
     * the final S3 block then reads the opposite phase. */
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u &&
        cfg->s3_call[0].valid != 0u && st->s3_block_count > 1u) {
        __moe_pipeline_wait(
            &s2->compute_done, st->s3_block_count - 1u);
    }

    MOE_PROFILE_RESOURCE_BEGIN(profile);
    if (dma_binding == MOE_DYN_DMA_BOTH) {
        __moe_s4pf_run_both(
            blk, cfg, st, s2, gate_src, up_src,
            initial_phase, compute_active);
    } else if (dma_binding == MOE_DYN_DMA_XDMA) {
        __moe_s4pf_run_xdma(
            blk, cfg, st, s2, gate_src, up_src,
            initial_phase, compute_active);
    } else {
        __moe_s4pf_run_idma(
            cfg, st, s2, gate_src, up_src,
            initial_phase, compute_active);
    }

    if (__moe_dyn_has_output(cfg) != 0u) {
        __moe_pipeline_publish(&s2->store_prepared, 1u);
    }
    MOE_PROFILE_RESOURCE_END(profile);

    MOE_INDIV_PRINT(
        "[INDIV_PREFETCH_S4_PHASE_DONE] C%u slot=%u eid=%u "
        "target_eid=%u dma=%u bytes=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        expert_id, dma_binding, weight_bytes, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
        __moe_profile_dma_resource(dma_binding), 0u,
        2u * weight_bytes, 0u, BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}

/* S2 consumes the final call record lowered by CVA6 in MoEPrepare. */
