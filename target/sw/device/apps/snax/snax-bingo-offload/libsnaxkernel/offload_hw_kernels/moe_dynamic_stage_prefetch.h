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
    uint32_t transfer_started = 0u;
    for (uint32_t step = 0u; step < s2->transfer_count; step++) {
        if (s2->sync_enabled != 0u && step != 0u) {
            __moe_pipeline_wait(&s2->compute_done, step);
        }

        uint32_t phase = step & 1u;
        uint32_t phase_group = step >> 1u;
        for (uint32_t sub = 0u; sub < s2->transfers_per_step; sub++) {
            uint32_t block = 0u;
            uint32_t side = 0u;
            uint32_t slice_ordinal =
                phase_group * s2->transfers_per_step + sub;
            if (__moe_s4_dma_run(
                    s2->block_count, 2u, phase, slice_ordinal,
                    &block, &side) == 0u) {
                continue;
            }

            uint32_t src_offset = block * block_bytes + side * half_bytes;
            uint32_t dst = __moe_bank_down_weight_block_addr(
                s2->down_dst_base + side * 64u, block, block_bytes);
            __moe_dyn_2d_pair_pending_t pending;
            __moe_dyn_start_single_2d(
                dma_binding, __moe_dyn_l1_wide(dst),
                s2->down_src_base + src_offset, block_bytes,
                transfer_started == 0u, &pending);
            transfer_started++;

            if (prepare_s3_during_final_dma != 0u &&
                transfer_started == 2u * s2->block_count) {
                __moe_prepare_s3_xdma_shape(blk, cfg, st);
                s3_prepared_early = 1u;
            }
            __moe_dyn_wait_pair_2d(&pending);
        }
        if (s2->sync_enabled != 0u) {
            __moe_pipeline_publish(&s2->prefetch_done, step + 1u);
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
__moe_dyn_prefetch_s4_phase(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint64_t gate_src,
    uint64_t up_src,
    uint32_t dma_binding,
    uint32_t phase,
    uint32_t xdma_addresses_preloaded,
    uint32_t preload_next_phase,
    uint32_t prepare_store)
{
    uint32_t phase_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, phase);
    if (phase_blocks == 0u) {
        if (prepare_store != 0u && __moe_dyn_has_output(cfg) != 0u) {
            __moe_dyn_prepare_store_xdma(cfg, st);
        }
        return;
    }

    if (phase == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE0_START);
    } else {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE1_START);
    }

    uint32_t block_bytes = st->indiv_B_block_stride;
    uint32_t src_offset = phase * block_bytes;
    uint32_t gate_dst = __moe_bank_weight_block_addr(
        st->l1_b_gate_addr, phase, block_bytes);
    uint32_t up_dst = __moe_bank_weight_block_addr(
        st->l1_b_up_addr, phase, block_bytes);
    uint64_t gate_src_phase = gate_src + src_offset;
    uint64_t up_src_phase = up_src + src_offset;
    uint64_t gate_dst_phase = __moe_dyn_l1_wide(gate_dst);
    uint64_t up_dst_phase = __moe_dyn_l1_wide(up_dst);
    int32_t xdma_task0 = -1;
    int32_t xdma_task1 = -1;
    uint32_t wait_idma = 0u;

    if (__moe_dyn_binding_uses_xdma(dma_binding) != 0u) {
        if (!__moe_xdma_stage_is_prepared(blk, MOE_XDMA_PREPARED_S4PF) ||
            xdma_addresses_preloaded == 0u) {
            if (!__moe_xdma_stage_is_prepared(
                    blk, MOE_XDMA_PREPARED_S4PF)) {
                __moe_prepare_s4pf_xdma_phase_shape(
                    blk, block_bytes, st->s1_block_count, phase);
            }
            if (dma_binding == MOE_DYN_DMA_XDMA) {
                xdma_memcpy_fast_set_addresses(
                    gate_src_phase, gate_dst_phase);
            } else {
                xdma_memcpy_fast_set_addresses(up_src_phase, up_dst_phase);
            }
        }

        xdma_task0 = (int32_t)xdma_start();
        if (dma_binding == MOE_DYN_DMA_XDMA) {
            xdma_memcpy_fast_set_addresses(up_src_phase, up_dst_phase);
            xdma_task1 = (int32_t)xdma_start();
        }
    }

    if (dma_binding != MOE_DYN_DMA_XDMA) {
        uint32_t repeats = block_bytes / MOE_BANK_WEIGHT_ROW_BYTES;
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        for (uint32_t n = phase; n < st->s1_block_count; n += 2u) {
            snrt_dma_start_2d_wideptr(
                __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
                    st->l1_b_gate_addr, n, block_bytes)),
                gate_src + (uint64_t)n * block_bytes,
                MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_TCDM_ROW_BYTES,
                MOE_BANK_WEIGHT_ROW_BYTES, repeats);
            if (dma_binding == MOE_DYN_DMA_IDMA) {
                snrt_dma_start_2d_wideptr(
                    __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
                        st->l1_b_up_addr, n, block_bytes)),
                    up_src + (uint64_t)n * block_bytes,
                    MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_TCDM_ROW_BYTES,
                    MOE_BANK_WEIGHT_ROW_BYTES, repeats);
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        wait_idma = 1u;
    }

    if (preload_next_phase != 0u &&
        __moe_dyn_binding_uses_xdma(dma_binding) != 0u) {
        uint32_t next_phase = phase ^ 1u;
        uint32_t next_offset = next_phase * block_bytes;
        if (dma_binding == MOE_DYN_DMA_XDMA) {
            xdma_memcpy_fast_set_addresses(
                gate_src + next_offset,
                __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
                    st->l1_b_gate_addr, next_phase, block_bytes)));
        } else {
            xdma_memcpy_fast_set_addresses(
                up_src + next_offset,
                __moe_dyn_l1_wide(__moe_bank_weight_block_addr(
                    st->l1_b_up_addr, next_phase, block_bytes)));
        }
    }

    if (prepare_store != 0u && __moe_dyn_has_output(cfg) != 0u) {
        __moe_dyn_prepare_store_xdma(cfg, st);
    }

    if (wait_idma != 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
    }
    if (xdma_task0 >= 0) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
        xdma_remote_wait((uint32_t)xdma_task0);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
    }
    if (xdma_task1 >= 0) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
        xdma_remote_wait((uint32_t)xdma_task1);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
    }

    if (phase == 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE0_END);
    } else {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4PF_PHASE1_END);
    }
}

/* Production S4 prefetch: transfer all next-S1 blocks in one bank phase before
 * exchanging phase ownership with the S4 compute worker. */
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
    uint32_t prefetch_valid = MOE_DYN_VD_VALID(cfg->dma_slot_vd, slot);
    uint32_t dma_binding = prefetch_valid != 0u ?
        MOE_DYN_VD_DMA(cfg->dma_slot_vd, slot) : 0u;

    if (prefetch_valid == 0u) {
        if (__moe_dyn_has_output(cfg) != 0u &&
            s2->store_prepared == 0u) {
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
    uint32_t first_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, initial_phase);
    uint32_t second_blocks = __moe_s4_blocks_in_phase(
        st->s1_block_count, initial_phase ^ 1u);
    uint32_t equal_phase_sizes = first_blocks == second_blocks;
    for (uint32_t step = 0u; step < 2u; step++) {
        if (step != 0u && compute_active != 0u) {
            __moe_pipeline_wait(&s2->sync_enabled, step);
        }

        uint32_t phase = __moe_s4_phase_at_step(initial_phase, step);
        if (step != 0u && equal_phase_sizes == 0u &&
            __moe_dyn_binding_uses_xdma(dma_binding) != 0u &&
            __moe_s4_blocks_in_phase(st->s1_block_count, phase) != 0u) {
            __moe_prepare_s4pf_xdma_phase_shape(
                blk, st->indiv_B_block_stride,
                st->s1_block_count, phase);
        }
        __moe_dyn_prefetch_s4_phase(
            blk, cfg, st, gate_src, up_src, dma_binding, phase,
            step != 0u && equal_phase_sizes != 0u &&
                __moe_dyn_binding_uses_xdma(dma_binding) != 0u,
            step == 0u && equal_phase_sizes != 0u && second_blocks != 0u,
            step == 1u);
        __moe_pipeline_publish(&s2->reserved, step + 1u);
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
