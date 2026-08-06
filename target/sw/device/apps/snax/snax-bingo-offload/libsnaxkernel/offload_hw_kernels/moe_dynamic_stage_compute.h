// Internal dynamic-MoE module; include through moe_dynamic.h.
#pragma once

__attribute__((always_inline)) static inline void
__moe_run_s2_compute(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t s4_csr_layout,
    uint32_t configure_block0)
{
    MOE_PROFILE_BEGIN(profile);
    const __snax_bingo_moe_dyn_s2_call_args_t *call = &cfg->s2_call;
    if (call->valid == 0u) {
        MOE_PROFILE_COMMIT(
            (void *)blk, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S2,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return;
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    uint32_t m_tiles = call->M;
    uint32_t block_count = st->s1_block_count;
    uint32_t n_tiles = __moe_dyn_stage_block_n(call->N, block_count);
    uint32_t token_start = call->token_start;
    uint32_t token_step = __moe_dyn_shape_m(call->array_shape);
    uint32_t input_base = __moe_dyn_input_base(cfg, st);
    uint32_t intermediate_base = __moe_dyn_intermediate_base(cfg, st);

    for (uint32_t mt = 0u; mt < m_tiles; mt++) {
        uint32_t token = token_start + mt * token_step;
        for (uint32_t n = 0u; n < block_count; n++) {
            if (configure_block0 != 0u && mt == 0u && n == 0u &&
                !__moe_csr_stage_is_prepared(blk, MOE_CSR_PREPARED_S2)) {
                __moe_bank_configure_mode0(
                    __moe_bank_a_addr(
                        input_base, token, st->A_token_bytes),
                    __moe_bank_weight_block_addr(
                        st->l1_b_gate_addr, n, st->indiv_B_block_stride),
                    __moe_bank_weight_block_addr(
                        st->l1_b_up_addr, n, st->indiv_B_block_stride),
                    __moe_bank_mode0_output_addr(
                        intermediate_base, token, n,
                        st->indiv_N_per_block, block_count),
                    st->indiv_K1, n_tiles, call->array_shape,
                    st->rescale_mult, st->rescale_shift);
                __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S2);
            }

            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
            moe_start_dual_vc_and_streamer();
            uint32_t next_n = n + 1u;
            uint32_t next_mt = mt;
            if (next_n == block_count) {
                next_n = 0u;
                next_mt++;
            }
            if (next_mt < m_tiles) {
                uint32_t next_token = token_start + next_mt * token_step;
                __moe_bank_patch_mode0_run_bases(
                    __moe_bank_a_addr(
                        input_base, next_token, st->A_token_bytes),
                    __moe_bank_weight_block_addr(
                        st->l1_b_gate_addr, next_n,
                        st->indiv_B_block_stride),
                    __moe_bank_weight_block_addr(
                        st->l1_b_up_addr, next_n,
                        st->indiv_B_block_stride),
                    __moe_bank_mode0_output_addr(
                        intermediate_base, next_token, next_n,
                        st->indiv_N_per_block, block_count));
            } else {
                __moe_prepare_after_s2(blk, cfg, st, s4_csr_layout);
            }
            moe_wait_dual_vc_and_streamer();
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
        }
    }

    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S2_DONE] C%u slot=%u eid=%u M=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        m_tiles, call->N, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_END);
    MOE_PROFILE_COMMIT(
        (void *)blk, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S2,
        MOE_PROFILE_RESOURCE_VERSACORE, 0u,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, BINGO_RET_SUCC);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_compute_s2(void *arg)
{
    MOE_PROFILE_BEGIN(stage_profile);
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk =
        (const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *)arg;
    const __snax_bingo_moe_dynamic_expert_static_args_t *st =
        (const __snax_bingo_moe_dynamic_expert_static_args_t *)(uintptr_t)
        blk->static_arg_addr;
    __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg =
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)
        blk->task_arg_addr;
    if (!__moe_dyn_slot_active_this_round(cfg, st) ||
        cfg->s2_call.valid == 0u) {
        MOE_PROFILE_COMMIT(
            arg, cfg, stage_profile, MOE_PROFILE_STAGE_COMPUTE_S2,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }
    MOE_PROFILE_RESOURCE_BEGIN(stage_profile);
    __moe_run_s2_compute(
        blk, cfg, st, MOE_S4_CSR_LAYOUT_PHASE_BATCHED, 0u);
    MOE_PROFILE_RESOURCE_END(stage_profile);
    MOE_PROFILE_COMMIT(
        arg, cfg, stage_profile, MOE_PROFILE_STAGE_COMPUTE_S2,
        MOE_PROFILE_RESOURCE_VERSACORE, 0u, 0u, 0u, BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}

/* S4 consumes the final call record lowered by CVA6 in MoEPrepare.  One RUN
 * traverses every down-weight block resident in the selected bank phase. */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_compute_s4(void *arg)
{
    MOE_PROFILE_BEGIN(profile);
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
        MOE_PROFILE_COMMIT(
            arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
            MOE_PROFILE_RESOURCE_NONE, 0u, 0u,
            MOE_PROFILE_FLAG_SKIPPED | MOE_PROFILE_FLAG_CTRL_SKIP,
            BINGO_RET_SUCC);
        return BINGO_RET_SUCC;
    }

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

    __moe_s2_prefetch_ctrl_t *sync = __moe_s2_prefetch_ctrl(blk);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);

    uint32_t token_start = call->token_start;
    uint32_t token_step = __moe_dyn_shape_m(call->array_shape);
    uint32_t m_tiles = call->M;
    uint32_t intermediate_base = __moe_dyn_intermediate_base(cfg, st);
    uint32_t output_base = __moe_dyn_output_base(cfg, st);
    uint32_t initial_phase = __moe_s4_block_initial_phase(st);
    uint32_t synchronize = MOE_DYN_VD_VALID(
        cfg->dma_slot_vd, MOE_DYN_DMA_SLOT_S4_PREFETCH);

    for (uint32_t step = 0u; step < 2u; step++) {
        if (step != 0u && synchronize != 0u) {
            __moe_pipeline_wait(&sync->reserved, step);
        }

        uint32_t phase = __moe_s4_phase_at_step(initial_phase, step);
        uint32_t phase_blocks = __moe_s4_blocks_in_phase(
            st->s3_block_count, phase);
        if (phase == 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4COMP_PHASE0_START);
        } else {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4COMP_PHASE1_START);
        }

        for (uint32_t mt = 0u; mt < m_tiles && phase_blocks != 0u; mt++) {
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
            moe_start_dual_vc_and_streamer();

            if (mt + 1u < m_tiles) {
                uint32_t next_token = token_start + (mt + 1u) * token_step;
                __moe_bank_patch_mode1_run_bases(
                    __moe_bank_mode0_output_addr(
                        intermediate_base, next_token, 0u,
                        st->indiv_N_per_block, st->s1_block_count),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr, phase,
                        st->indiv_down_B_block_stride),
                    __moe_bank_down_weight_block_addr(
                        st->l1_b_down_addr + 64u, phase,
                        st->indiv_down_B_block_stride),
                    __moe_bank_mode1_output_addr(
                        output_base, next_token, phase,
                        st->indiv_down_N_per_block, st->s3_block_count),
                    __moe_bank_mode1_output_addr(
                        output_base + 64u, next_token, phase,
                        st->indiv_down_N_per_block, st->s3_block_count));
            } else if (step == 0u) {
                uint32_t next_phase = __moe_s4_phase_at_step(
                    initial_phase, 1u);
                if (__moe_s4_blocks_in_phase(
                        st->s3_block_count, next_phase) != 0u) {
                    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
                    __moe_configure_s4_phase(
                        cfg, st, token_start, next_phase);
                    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
                }
            } else {
                (void)__moe_prepare_next_slot_s2_csr(blk, cfg, st);
            }

            moe_wait_dual_vc_and_streamer();
            BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
        }

        if (phase_blocks == 0u && step == 0u) {
            uint32_t next_phase = __moe_s4_phase_at_step(initial_phase, 1u);
            if (__moe_s4_blocks_in_phase(
                    st->s3_block_count, next_phase) != 0u) {
                BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
                __moe_configure_s4_phase(cfg, st, token_start, next_phase);
                BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
            }
        }

        if (phase == 0u) {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4COMP_PHASE0_END);
        } else {
            BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_S4COMP_PHASE1_END);
        }
        if (synchronize != 0u) {
            __moe_pipeline_publish(&sync->sync_enabled, step + 1u);
        }
    }

    MOE_PROFILE_CAPTURE_VC_COUNTER(profile);
    MOE_PROFILE_RESOURCE_END(profile);
    MOE_INDIV_PRINT(
        "[INDIV_S4_PHASE_DONE] C%u slot=%u eid=%u M=%u N=%u rc=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        m_tiles, call->N, BINGO_RET_SUCC);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_END);
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_COMPUTE_S4,
        MOE_PROFILE_RESOURCE_VERSACORE, 0u,
        MOE_PROFILE_RESOURCE_UNITS(profile), 0u, BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}
