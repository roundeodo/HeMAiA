// Internal dynamic-MoE module; include through moe_dynamic.h.
#pragma once

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
        __moe_prepare_s3_xdma_shape(blk, cfg, st);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S2,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S2_START);
    uint32_t half_bytes = s2->half_bytes;
    uint32_t block_bytes = s2->block_bytes;
    uint32_t dma_binding = s2->binding;
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t prepare_s3_during_final_dma =
        MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u &&
        __moe_dyn_binding_uses_xdma(MOE_DYN_CTRL_DMA_S3(cfg->ctrl)) != 0u;
    uint32_t s3_prepared_early = 0u;
    for (uint32_t n = 0u; n < s2->block_count; n++) {
        if (s2->sync_enabled != 0u && n != 0u) {
            uint32_t target = n < s2->s1_block_count ? n : s2->s1_block_count;
            __moe_pipeline_wait(&s2->compute_done, target);
        }
        uint32_t src_offset = n * block_bytes;
        uint32_t left_dst = __moe_bank_down_weight_block_addr(
            s2->down_dst_base, n, block_bytes);
        uint32_t right_dst = __moe_bank_down_weight_block_addr(
            s2->down_dst_base + 64u, n, block_bytes);
        if (prepare_s3_during_final_dma != 0u &&
            n + 1u == s2->block_count) {
            __moe_dyn_2d_pair_pending_t pending;
            __moe_dyn_start_pair_2d(
                dma_binding,
                __moe_dyn_l1_wide(left_dst),
                s2->down_src_base + src_offset,
                __moe_dyn_l1_wide(right_dst),
                s2->down_src_base + half_bytes + src_offset,
                block_bytes, n == 0u, &pending);
            __moe_prepare_s3_xdma_shape(blk, cfg, st);
            __moe_dyn_wait_pair_2d(&pending);
            s3_prepared_early = 1u;
        } else {
            __moe_dyn_copy_pair_2d(
                dma_binding,
                __moe_dyn_l1_wide(left_dst),
                s2->down_src_base + src_offset,
                __moe_dyn_l1_wide(right_dst),
                s2->down_src_base + half_bytes + src_offset,
                block_bytes, n == 0u);
        }
        if (s2->sync_enabled != 0u && n + 1u < s2->block_count) {
            __moe_pipeline_publish(&s2->prefetch_done, n + 1u);
        }
    }
    if (prepare_s3_during_final_dma != 0u &&
        s3_prepared_early == 0u) {
        __moe_prepare_s3_xdma_shape(blk, cfg, st);
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
__moe_dyn_prefetch_s4_blocks(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    __moe_s2_prefetch_ctrl_t *s2,
    uint32_t sync_steps,
    uint32_t compute_active,
    uint32_t initial_phase,
    uint32_t phase_steps0,
    uint32_t phase_steps1,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t dma_binding)
{
    uint32_t dma_runs_done = 0u;
    uint32_t xdma_addresses_preloaded = 0u;
    uint32_t scheduled = sync_steps != 0u ? __moe_s4_schedule_step(
        phase_steps0, phase_steps1, initial_phase, 0u) : 0u;

    for (uint32_t step = 0u; step < sync_steps; step++) {
        if (step != 0u && compute_active != 0u) {
            __moe_pipeline_wait(&s2->sync_enabled, step);
        }

        uint32_t selected_phase = scheduled & 1u;
        uint32_t n = scheduled;
        if (n < st->s1_block_count) {
            if (selected_phase == 0u) {
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE0_START);
            } else {
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE1_START);
            }

            uint32_t src_offset = n * st->indiv_B_block_stride;
            uint32_t gate_dst = __moe_bank_weight_block_addr(
                st->l1_b_gate_addr, n, st->indiv_B_block_stride);
            uint32_t up_dst = __moe_bank_weight_block_addr(
                st->l1_b_up_addr, n, st->indiv_B_block_stride);
            uint32_t configure_xdma = dma_runs_done == 0u &&
                !__moe_xdma_stage_is_prepared(
                    blk, MOE_XDMA_PREPARED_S4PF);
            uint32_t final_dma_block =
                dma_runs_done + 1u == st->s1_block_count;
            uint32_t prepare_store = final_dma_block != 0u &&
                __moe_dyn_has_output(cfg) != 0u;
            __moe_dyn_2d_pair_pending_t pending;

            if (dma_binding == MOE_DYN_DMA_IDMA && prepare_store != 0u) {
                __moe_dyn_start_final_pair_2d_and_prepare_store(
                    dma_binding,
                    __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                    __moe_dyn_l1_wide(up_dst), up_src + src_offset,
                    st->indiv_B_block_stride, configure_xdma,
                    cfg, st, &pending);
            } else if (dma_binding == MOE_DYN_DMA_IDMA) {
                __moe_dyn_start_pair_2d(
                    dma_binding,
                    __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                    __moe_dyn_l1_wide(up_dst), up_src + src_offset,
                    st->indiv_B_block_stride, configure_xdma, &pending);
            } else {
                if (xdma_addresses_preloaded == 0u) {
                    if (configure_xdma != 0u) {
                        BINGO_TRACE_MARKER(
                            BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
                        xdma_memcpy_2d_fast_configure(
                            MOE_BANK_WEIGHT_ROW_BYTES,
                            MOE_BANK_WEIGHT_ROW_BYTES,
                            MOE_BANK_TCDM_ROW_BYTES,
                            st->indiv_B_block_stride /
                                MOE_BANK_WEIGHT_ROW_BYTES);
                        BINGO_TRACE_MARKER(
                            BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
                    }
                    __moe_dyn_prepare_pair_2d_xdma_addresses(
                        dma_binding,
                        __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                        __moe_dyn_l1_wide(up_dst), up_src + src_offset);
                }
                __moe_dyn_start_pair_2d_preloaded_xdma(
                    dma_binding,
                    __moe_dyn_l1_wide(gate_dst), gate_src + src_offset,
                    __moe_dyn_l1_wide(up_dst), up_src + src_offset,
                    st->indiv_B_block_stride, &pending);
                if (prepare_store != 0u) {
                    __moe_dyn_prepare_store_xdma(cfg, st);
                }
            }

            if (step + 1u < sync_steps) {
                scheduled = __moe_s4_schedule_step(
                    phase_steps0, phase_steps1, initial_phase, step + 1u);
            }

            xdma_addresses_preloaded = 0u;
            if (dma_binding != MOE_DYN_DMA_IDMA &&
                step + 1u < sync_steps &&
                scheduled < st->s1_block_count) {
                uint32_t next_src_offset =
                    scheduled * st->indiv_B_block_stride;
                uint32_t next_gate_dst = __moe_bank_weight_block_addr(
                    st->l1_b_gate_addr, scheduled,
                    st->indiv_B_block_stride);
                uint32_t next_up_dst = __moe_bank_weight_block_addr(
                    st->l1_b_up_addr, scheduled,
                    st->indiv_B_block_stride);
                __moe_dyn_prepare_pair_2d_xdma_addresses(
                    dma_binding,
                    __moe_dyn_l1_wide(next_gate_dst),
                    gate_src + next_src_offset,
                    __moe_dyn_l1_wide(next_up_dst),
                    up_src + next_src_offset);
                xdma_addresses_preloaded = 1u;
            }

            if (prepare_store != 0u) {
                __moe_pipeline_publish(&s2->store_prepared, 1u);
            }
            __moe_dyn_wait_pair_2d(&pending);
            dma_runs_done++;

            if (selected_phase == 0u) {
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE0_END);
            } else {
                BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE1_END);
            }
        }

        if (n >= st->s1_block_count && step + 1u < sync_steps) {
            scheduled = __moe_s4_schedule_step(
                phase_steps0, phase_steps1, initial_phase, step + 1u);
        }

        __moe_pipeline_publish(&s2->reserved, step + 1u);
    }
}

/* Production S4 prefetch: advance one bank-safe block step at a time with the
 * S4 compute worker. The two workers use reserved and sync_enabled as their
 * monotonically increasing completion counters. */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_prefetch_s4(void *arg)
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
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) != 0u ||
        cfg->s3_call[0].valid == 0u) {
        __moe_initialize_next_slot(blk, cfg, st);
    }

    uint32_t initial_phase = __moe_s4_block_initial_phase(st);
    uint32_t compute_active = cfg->s4_call.valid != 0u &&
        cfg->s4_call.M != 0u;
    uint32_t m_tiles = compute_active != 0u ? cfg->s4_call.M : 0u;
    uint32_t sync_steps = __moe_s4_phase_schedule_length(
        st->s1_block_count, st->s3_block_count, m_tiles);

    if (MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot) == 0u) {
        if (__moe_dyn_has_output(cfg) != 0u &&
            s2->store_prepared == 0u) {
            __moe_dyn_prepare_store_xdma(cfg, st);
            __moe_pipeline_publish(&s2->store_prepared, 1u);
        }
        __moe_pipeline_publish(&s2->reserved, sync_steps);
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_PREFETCH_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_NO_PREFETCH,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

    uint32_t dma_binding = MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_PREFETCH_S4_START);
    uint32_t expert_id = MOE_DYN_DMA_EID(cfg->dma_slot_eids, slot);
    uint32_t weight_bytes = st->indiv_B_expert_stride;
    uint64_t gate_src = st->indiv_gate_B_l3 +
        (uint64_t)expert_id * weight_bytes;
    uint64_t up_src = st->indiv_up_B_l3 +
        (uint64_t)expert_id * weight_bytes;
    uint32_t phase_steps0 = __moe_s4_phase_steps(
        st->s1_block_count, st->s3_block_count, m_tiles, 0u);
    uint32_t phase_steps1 = __moe_s4_phase_steps(
        st->s1_block_count, st->s3_block_count, m_tiles, 1u);

    /* S3 load may finish two blocks before S3 compute. Wait until the
     * penultimate S3 block releases the phase selected for the first S4 DMA;
     * the final S3 block then reads the opposite phase. */
    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u &&
        cfg->s3_call[0].valid != 0u && st->s3_block_count > 1u) {
        __moe_pipeline_wait(
            &s2->compute_done, st->s3_block_count - 1u);
    }

    MOE_PROFILE_RESOURCE_BEGIN(profile);
    if (dma_binding == MOE_DYN_DMA_IDMA) {
        __moe_dyn_prefetch_s4_blocks(
            blk, cfg, st, s2, sync_steps, compute_active, initial_phase,
            phase_steps0, phase_steps1, gate_src, up_src,
            MOE_DYN_DMA_IDMA);
    } else if (dma_binding == MOE_DYN_DMA_XDMA) {
        __moe_dyn_prefetch_s4_blocks(
            blk, cfg, st, s2, sync_steps, compute_active, initial_phase,
            phase_steps0, phase_steps1, gate_src, up_src,
            MOE_DYN_DMA_XDMA);
    } else {
        __moe_dyn_prefetch_s4_blocks(
            blk, cfg, st, s2, sync_steps, compute_active, initial_phase,
            phase_steps0, phase_steps1, gate_src, up_src,
            MOE_DYN_DMA_BOTH);
    }

    if (__moe_dyn_has_output(cfg) != 0u &&
        s2->store_prepared == 0u) {
        __moe_dyn_prepare_store_xdma(cfg, st);
        __moe_pipeline_publish(&s2->store_prepared, 1u);
    }
    MOE_PROFILE_RESOURCE_END(profile);

    MOE_INDIV_PRINT(
        "[INDIV_PREFETCH_S4_BLOCK_DONE] C%u slot=%u eid=%u "
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
