// Internal dynamic-MoE module; include through moe_dynamic.h.
#pragma once

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_stage_tokens_2d(void *arg)
{
    const __snax_bingo_kernel_moe_stage_tokens_2d_args_t *cfg =
        (const __snax_bingo_kernel_moe_stage_tokens_2d_args_t *)arg;
    uint32_t repeats = cfg->token_bytes / MOE_BANK_A_TOKEN_TILE_BYTES;
    for (uint32_t token = 0u; token < cfg->token_count; token++) {
        snrt_dma_start_2d_wideptr(
            __moe_dyn_l1_wide(
                __moe_bank_a_addr(
                    cfg->dst_addr, token, cfg->token_bytes)),
            cfg->src_addr + (uint64_t)token * cfg->token_bytes,
            MOE_BANK_A_TOKEN_TILE_BYTES, MOE_BANK_TCDM_ROW_BYTES,
            MOE_BANK_A_TOKEN_TILE_BYTES, repeats);
    }
    snrt_dma_wait_all();
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_stage_weight_pair_2d(void *arg)
{
    const __snax_bingo_kernel_moe_stage_weight_pair_2d_args_t *cfg =
        (const __snax_bingo_kernel_moe_stage_weight_pair_2d_args_t *)arg;

    /* Bingo task arguments live in TCDM. Parse them before any concurrent VC
     * run can monopolize the A banks; the transfer loop must not reread cfg. */
    const uint64_t src0_base = cfg->src0_addr;
    const uint64_t src1_base = cfg->src1_addr;
    const uint32_t dst0_base = cfg->dst0_addr;
    const uint32_t dst1_base = cfg->dst1_addr;
    const uint32_t bytes_per_block = cfg->bytes_per_block;
    const uint32_t block_count = cfg->block_count;
    const uint32_t binding = cfg->binding;
    const uint32_t phase_xor = cfg->phase_xor;

    uint32_t repeats = bytes_per_block / MOE_BANK_WEIGHT_ROW_BYTES;
    if (binding == MOE_DYN_DMA_IDMA) {
        /* Submit the complete resident tensor before the single wait. With the
         * workload ABI this is at most 2*BINGO_MOE_MAX_BLOCKS descriptors. */
#ifdef BINGO_MOE_IDMA_2D_PROBE_COUNTERS
        const enum snrt_perf_cnt_type probe_types[] = {
            SNRT_PERF_CNT_DMA_BUSY,
            SNRT_PERF_CNT_DMA_AR_DONE,
            SNRT_PERF_CNT_DMA_R_DONE,
            SNRT_PERF_CNT_DMA_AW_DONE,
            SNRT_PERF_CNT_DMA_W_DONE,
            SNRT_PERF_CNT_DMA_B_DONE,
            SNRT_PERF_CNT_DMA_AR_STALL,
            SNRT_PERF_CNT_DMA_AW_STALL,
            SNRT_PERF_CNT_TCDM_ACCESSED,
            SNRT_PERF_CNT_TCDM_CONGESTED,
        };
        for (uint32_t i = 0u; i < 10u; i++) {
            snrt_reset_perf_counter((enum snrt_perf_cnt)i);
            snrt_start_perf_counter(
                (enum snrt_perf_cnt)i, probe_types[i], 0u);
        }
        uint32_t probe_start = snrt_mcycle();
#endif
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_START);
        for (uint32_t block = 0u; block < block_count; block++) {
            uint32_t src_offset = block * bytes_per_block;
            uint32_t dst0 = __moe_bank_weight_block_addr_phase(
                dst0_base, block, bytes_per_block, phase_xor);
            uint32_t dst1 = __moe_bank_weight_block_addr_phase(
                dst1_base, block, bytes_per_block, phase_xor);
            snrt_dma_start_2d_wideptr(
                __moe_dyn_l1_wide(dst0), src0_base + src_offset,
                MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_TCDM_ROW_BYTES,
                MOE_BANK_WEIGHT_ROW_BYTES, repeats);
            snrt_dma_start_2d_wideptr(
                __moe_dyn_l1_wide(dst1), src1_base + src_offset,
                MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_TCDM_ROW_BYTES,
                MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_CFG_END);
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_START);
        snrt_dma_wait_all();
        BINGO_TRACE_MARKER(BINGO_TRACE_IDMA_RUN_END);
#ifdef BINGO_MOE_IDMA_2D_PROBE_COUNTERS
        uint32_t probe_cycles = snrt_mcycle() - probe_start;
        for (uint32_t i = 0u; i < 10u; i++) {
            snrt_stop_perf_counter((enum snrt_perf_cnt)i);
        }
        printf_safe(
            "IDMA2D_PROBE cycles=%u busy=%u ar=%u r=%u aw=%u w=%u b=%u "
            "ar_stall=%u aw_stall=%u tcdm=%u tcdm_cong=%u rows=%u\r\n",
            probe_cycles,
            snrt_get_perf_counter(SNRT_PERF_CNT0),
            snrt_get_perf_counter(SNRT_PERF_CNT1),
            snrt_get_perf_counter(SNRT_PERF_CNT2),
            snrt_get_perf_counter(SNRT_PERF_CNT3),
            snrt_get_perf_counter(SNRT_PERF_CNT4),
            snrt_get_perf_counter(SNRT_PERF_CNT5),
            snrt_get_perf_counter(SNRT_PERF_CNT6),
            snrt_get_perf_counter(SNRT_PERF_CNT7),
            snrt_get_perf_counter(SNRT_PERF_CNT8),
            snrt_get_perf_counter(SNRT_PERF_CNT9),
            2u * block_count * repeats);
#endif
        return BINGO_RET_SUCC;
    }

    /* Shape is invariant across blocks. Configure xDMA once and only commit
     * new source/destination addresses for each following descriptor. */
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START);
    xdma_memcpy_2d_fast_configure(
        MOE_BANK_WEIGHT_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES,
        MOE_BANK_TCDM_ROW_BYTES, repeats);
    BINGO_TRACE_MARKER(BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END);
    int32_t last_xdma_task = -1;
    uint64_t last_xdma_src = 0u;
    uint64_t last_xdma_dst = 0u;
    for (uint32_t block = 0u; block < block_count; block++) {
        uint32_t src_offset = block * bytes_per_block;
        uint64_t dst0 = __moe_dyn_l1_wide(__moe_bank_weight_block_addr_phase(
            dst0_base, block, bytes_per_block, phase_xor));
        uint64_t dst1 = __moe_dyn_l1_wide(__moe_bank_weight_block_addr_phase(
            dst1_base, block, bytes_per_block, phase_xor));
        uint64_t src0 = src0_base + src_offset;
        uint64_t src1 = src1_base + src_offset;

        if (binding == MOE_DYN_DMA_XDMA) {
            xdma_memcpy_fast_set_addresses(src0, dst0);
            last_xdma_task = (int32_t)xdma_start_remote();
            xdma_memcpy_fast_set_addresses(src1, dst1);
            last_xdma_task = (int32_t)xdma_start_remote();
        } else {
            xdma_memcpy_fast_set_addresses(src1, dst1);
            last_xdma_task = (int32_t)xdma_start_remote();
            snrt_dma_start_2d_wideptr(
                dst0, src0, MOE_BANK_WEIGHT_ROW_BYTES,
                MOE_BANK_TCDM_ROW_BYTES, MOE_BANK_WEIGHT_ROW_BYTES, repeats);
        }
        last_xdma_src = src1;
        last_xdma_dst = dst1;
    }
    if (binding == MOE_DYN_DMA_BOTH) snrt_dma_wait_all();
    __moe_dyn_wait_xdma(last_xdma_dst, last_xdma_src, last_xdma_task);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_store_tokens_2d(void *arg)
{
    const __snax_bingo_kernel_moe_store_tokens_2d_args_t *cfg =
        (const __snax_bingo_kernel_moe_store_tokens_2d_args_t *)arg;
    uint32_t page_span = __moe_bank_token_page_span(cfg->token_bytes);
    for (uint32_t token_start = 0u; token_start < cfg->token_count;
         token_start += MOE_BANK_TOKEN_LANES) {
        uint32_t remaining = cfg->token_count - token_start;
        uint32_t count = remaining < MOE_BANK_TOKEN_LANES ?
            remaining : MOE_BANK_TOKEN_LANES;
        uint32_t page = token_start / MOE_BANK_TOKEN_LANES;
        uint64_t src0 = __moe_dyn_l1_wide(
            cfg->src_d0_addr + page * page_span);
        uint64_t src1 = __moe_dyn_l1_wide(
            cfg->src_d1_addr + page * page_span);
        uint64_t dst = cfg->dst_addr +
            (uint64_t)token_start * cfg->token_bytes;
        if (token_start == 0u) {
            __moe_bank_configure_store(src0, dst, count, cfg->token_bytes);
        } else {
            __moe_bank_patch_store_page(src0, dst, count);
        }
        int32_t task0 = (int32_t)xdma_start_remote();
        xdma_memcpy_fast_set_addresses(
            src1, dst + cfg->token_bytes / 2u);
        int32_t task1 = (int32_t)xdma_start_remote();
        __moe_dyn_wait_xdma(dst, src0, task0);
        __moe_dyn_wait_xdma(dst + cfg->token_bytes / 2u, src1, task1);
    }
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_bank_moe_full(void *arg)
{
    const __snax_bingo_kernel_dual_vc_bank_moe_full_args_t *cfg =
        (const __snax_bingo_kernel_dual_vc_bank_moe_full_args_t *)arg;
    uint32_t s1_block_count = cfg->s1_block_count;
    uint32_t s3_block_count = cfg->s3_block_count;

    uint32_t mode0_block_bytes =
        cfg->hidden_size * cfg->chunk_cols / 2u;
    uint32_t mode1_block_bytes =
        cfg->intermediate_size * cfg->chunk_cols / 2u;
    uint32_t mode1_region =
        ((s1_block_count + 1u) / 2u) * mode0_block_bytes * 8u;
    uint32_t mode0_k = cfg->hidden_size / MOE_DUAL_VC_TILE_SIZE_0;
    uint32_t mode1_k = cfg->intermediate_size / MOE_DUAL_VC_TILE_SIZE_0;
    uint32_t mode1_b0 = cfg->tcdm_base + mode1_region +
        MOE_BANK_B0_PING_OFFSET;
    uint32_t mode1_b1 = cfg->tcdm_base + mode1_region +
        MOE_BANK_B1_PING_OFFSET;

    for (uint32_t token_start = 0u; token_start < cfg->token_count;
         token_start += MOE_BANK_TOKEN_LANES) {
        uint32_t remaining = cfg->token_count - token_start;
        uint32_t valid = remaining < MOE_BANK_TOKEN_LANES ?
            remaining : MOE_BANK_TOKEN_LANES;
        uint32_t shape = valid > 4u ? 0u : (valid > 2u ? 1u : 2u);
        uint32_t n_tiles = cfg->chunk_cols / __moe_dyn_meshcol(shape);

        for (uint32_t block = 0u; block < s1_block_count; block++) {
            uint32_t A_addr = __moe_bank_a_addr(
                cfg->tcdm_base, token_start,
                cfg->hidden_size * sizeof(int16_t));
            uint32_t B0_addr = __moe_bank_weight_block_addr(
                cfg->tcdm_base + MOE_BANK_B0_PING_OFFSET,
                block, mode0_block_bytes);
            uint32_t B1_addr = __moe_bank_weight_block_addr(
                cfg->tcdm_base + MOE_BANK_B1_PING_OFFSET,
                block, mode0_block_bytes);
            uint32_t D_addr = __moe_bank_mode0_output_addr(
                cfg->tcdm_base + MOE_BANK_MODE0_D_OFFSET,
                token_start, block, cfg->chunk_cols, s1_block_count);
            if (block == 0u) {
                __moe_bank_configure_mode0(
                    A_addr, B0_addr, B1_addr, D_addr,
                    mode0_k, n_tiles, shape,
                    cfg->rescale_mult, cfg->rescale_shift);
            } else {
                __moe_bank_patch_mode0_block_bases(
                    B0_addr, B1_addr, D_addr);
            }
            moe_start_dual_vc_and_streamer();
            moe_wait_dual_vc_and_streamer();
        }

        for (uint32_t block = 0u; block < s3_block_count; block++) {
            uint32_t A_addr = __moe_bank_mode0_output_addr(
                cfg->tcdm_base + MOE_BANK_MODE0_D_OFFSET,
                token_start, 0u, cfg->chunk_cols, s1_block_count);
            uint32_t B0_addr = __moe_bank_down_weight_block_addr(
                mode1_b0, block, mode1_block_bytes);
            uint32_t B1_addr = __moe_bank_down_weight_block_addr(
                mode1_b1, block, mode1_block_bytes);
            uint32_t D0_addr = __moe_bank_mode1_output_addr(
                cfg->tcdm_base + MOE_BANK_MODE1_D0_OFFSET,
                token_start, block, cfg->chunk_cols, s3_block_count);
            uint32_t D1_addr = __moe_bank_mode1_output_addr(
                cfg->tcdm_base + MOE_BANK_MODE1_D1_OFFSET,
                token_start, block, cfg->chunk_cols, s3_block_count);
            if (block == 0u) {
                __moe_bank_configure_mode1(
                    A_addr, B0_addr, B1_addr, D0_addr, D1_addr,
                    mode1_k, n_tiles, shape,
                    cfg->rescale_mult, cfg->rescale_shift);
            } else {
                __moe_bank_patch_mode1_block_bases(
                    B0_addr, B1_addr, D0_addr, D1_addr);
            }
            moe_start_dual_vc_and_streamer();
            moe_wait_dual_vc_and_streamer();
        }
    }
    return BINGO_RET_SUCC;
}
