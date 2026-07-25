// Internal dynamic-MoE module; include through moe_dynamic.h.
#pragma once

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_dyn_opt_store_gather(void *arg)
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
    MOE_PROFILE_BEGIN(profile);

    uint32_t row_stride = st->A_row_stride;
    uint64_t expert_out_base = st->output_l3_base +
        (uint64_t)cfg->expert_id * (uint64_t)st->output_expert_stride_bytes;
    uint32_t slot = MOE_DYN_CTRL_SLOT_ID(cfg->ctrl);
    volatile uint32_t *state =
        (volatile uint32_t *)(uintptr_t)st->active_state_l1_addr;
    uint32_t active_idx = (MOE_DYN_CTRL_CLUSTER(cfg->ctrl) == 0u) ?
        MOE_DYN_RT_C2_ACTIVE_SLOTS : MOE_DYN_RT_C3_ACTIVE_SLOTS;
    uint32_t has_next = (slot + 1u < state[active_idx]);
    __snax_bingo_kernel_moe_dynamic_expert_args_t *next_cfg = has_next ?
        (__snax_bingo_kernel_moe_dynamic_expert_args_t *)(uintptr_t)(
            (uintptr_t)cfg + BINGO_MOE_DYNAMIC_ARG_SLOT_BYTES) : 0;
    uint32_t store_bytes = 0u;
    uint32_t idma_bytes = 0u;
    int32_t xdma_task0 = -1;
    __snax_bingo_kernel_moe_dynamic_expert_block_args_t next_blk;

    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_START);
    MOE_PROFILE_RESOURCE_BEGIN(profile);
    if (has_next) {
        next_blk = *blk;
        next_blk.task_arg_addr += BINGO_MOE_DYNAMIC_ARG_SLOT_BYTES;
        next_blk.pipeline_ctrl_addr += MOE_PIPELINE_CTRL_SLOT_BYTES;
        next_blk.block_idx = 0u;
    }

    if (MOE_DYN_CTRL_SKIP_S3(cfg->ctrl) == 0u ||
        MOE_DYN_CTRL_SKIP_S4(cfg->ctrl) == 0u) {
        uint64_t dst = expert_out_base +
            (uint64_t)cfg->token_ref_start * (uint64_t)row_stride;
        uint32_t output_base = __moe_dyn_output_base(cfg, st);
        uint64_t src = __moe_dyn_l1_wide(output_base);
        store_bytes = cfg->ntokens * st->A_token_bytes;
        MOE_INDIV_PRINT(
            "[INDIV_STORE_BEGIN] C%u slot=%u eid=%u start=%u ntok=%u "
            "src=0x%08x_%08x dst=0x%08x_%08x "
            "row_bytes=%u row_stride=%u bytes=%u\r\n",
            snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
            cfg->token_ref_start, cfg->ntokens,
            (uint32_t)(src >> 32u), (uint32_t)src,
            (uint32_t)(dst >> 32u), (uint32_t)dst,
            st->A_token_bytes, row_stride, store_bytes);
        xdma_task0 = (int32_t)xdma_start_remote();
    }

    if (has_next) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_START);
        idma_bytes = next_cfg->ntokens * st->A_token_bytes;
        __moe_submit_token_gather(next_cfg, st);
        __moe_initialize_slot_if_needed(&next_blk, next_cfg, st);
        if (xdma_task0 < 0 || cfg->ntokens <= MOE_BANK_TOKEN_LANES) {
            __moe_prepare_s1_xdma_shape(&next_blk, next_cfg, st);
        }
    }

    /* Slot parity keeps the current output and next input in opposite 16-bank
     * groups, so all output pages may drain while the next gather is active. */
    if (xdma_task0 >= 0 && cfg->ntokens > MOE_BANK_TOKEN_LANES) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
        xdma_remote_wait((uint32_t)xdma_task0);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);

        uint32_t page_span = __moe_bank_token_page_span(st->A_token_bytes);
        uint32_t output_base = __moe_dyn_output_base(cfg, st);
        for (uint32_t token_start = MOE_BANK_TOKEN_LANES;
             token_start < cfg->ntokens;
             token_start += MOE_BANK_TOKEN_LANES) {
            uint32_t remaining = cfg->ntokens - token_start;
            uint32_t count = remaining < MOE_BANK_TOKEN_LANES ?
                remaining : MOE_BANK_TOKEN_LANES;
            uint32_t page = token_start / MOE_BANK_TOKEN_LANES;
            uint64_t src0 = __moe_dyn_l1_wide(
                output_base + page * page_span);
            uint64_t dst = expert_out_base +
                (uint64_t)(cfg->token_ref_start + token_start) *
                    (uint64_t)row_stride;
            __moe_bank_patch_store_page(src0, dst, count);
            int32_t page_task0 = (int32_t)xdma_start_remote();
            xdma_remote_wait((uint32_t)page_task0);
        }
        xdma_task0 = -1;
        if (has_next) {
            __moe_prepare_s1_xdma_shape(&next_blk, next_cfg, st);
        }
    }

    if (has_next) {
        __moe_wait_token_gather();
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_GATHER_S1_END);
    }

    if (xdma_task0 >= 0) {
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START);
        xdma_remote_wait((uint32_t)xdma_task0);
        BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END);
    }
    MOE_PROFILE_RESOURCE_END(profile);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_STORE_END);

    MOE_INDIV_PRINT(
        "[INDIV_DONE] C%u slot=%u eid=%u start=%u ntok=%u "
        "store_bytes=%u next_gather_bytes=%u\r\n",
        snrt_cluster_idx(), MOE_DYN_CTRL_SLOT_ID(cfg->ctrl), cfg->expert_id,
        cfg->token_ref_start, cfg->ntokens, store_bytes, idma_bytes);
    uint32_t profile_resource = MOE_PROFILE_RESOURCE_NONE;
    if (store_bytes != 0u) {
        profile_resource = idma_bytes != 0u ? MOE_PROFILE_RESOURCE_DMA_BOTH :
                                             MOE_PROFILE_RESOURCE_XDMA;
    } else if (idma_bytes != 0u) {
        profile_resource = MOE_PROFILE_RESOURCE_IDMA;
    }
    MOE_PROFILE_COMMIT(
        arg, cfg, profile, MOE_PROFILE_STAGE_STORE,
        profile_resource,
        0u, store_bytes + idma_bytes, 0u,
        BINGO_RET_SUCC);
    return BINGO_RET_SUCC;
}
