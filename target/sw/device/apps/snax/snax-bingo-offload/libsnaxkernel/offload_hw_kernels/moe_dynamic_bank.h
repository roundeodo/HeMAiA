// Bank-aware addressing and stage CSR preparation.
#pragma once

static inline uint64_t __moe_dyn_l1_wide(uint32_t local_addr)
{
    return chiplet_addr_transform((uint64_t)local_addr);
}
static inline uint32_t __moe_bank_token_page_span(uint32_t token_bytes)
{
    return (token_bytes / MOE_BANK_A_TOKEN_TILE_BYTES) *
        MOE_BANK_TCDM_ROW_BYTES;
}

static inline uint32_t __moe_bank_paged_token_addr(
    uint32_t base, uint32_t token, uint32_t page_span,
    uint32_t lane_bytes)
{
    return base + (token / MOE_BANK_TOKEN_LANES) * page_span +
        (token % MOE_BANK_TOKEN_LANES) * lane_bytes;
}

static inline uint32_t __moe_bank_a_addr(
    uint32_t base, uint32_t token, uint32_t token_bytes)
{
    return __moe_bank_paged_token_addr(
        base, token, __moe_bank_token_page_span(token_bytes),
        MOE_BANK_A_TOKEN_TILE_BYTES);
}

static inline uint32_t __moe_bank_mode0_output_addr(
    uint32_t base, uint32_t token_start, uint32_t block,
    uint32_t chunk_cols, uint32_t block_count)
{
    uint32_t block_span =
        (chunk_cols / MOE_BANK_TOKEN_LANES) * MOE_BANK_TCDM_ROW_BYTES;
    return __moe_bank_paged_token_addr(
               base, token_start, block_count * block_span,
               MOE_BANK_A_TOKEN_TILE_BYTES) +
        block * block_span;
}

static inline uint32_t __moe_bank_mode1_output_addr(
    uint32_t base, uint32_t token_start, uint32_t block,
    uint32_t chunk_cols, uint32_t block_count)
{
    uint32_t block_span = (chunk_cols / 4u) * MOE_BANK_TCDM_ROW_BYTES;
    return __moe_bank_paged_token_addr(
               base, token_start, block_count * block_span, 8u) +
        block * block_span;
}

__attribute__((always_inline)) static inline void
__moe_bank_patch_mode0_block_bases(
    uint32_t B0_addr, uint32_t B1_addr, uint32_t D_addr)
{
    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(BASE_PTR_WRITER_0_LOW, D_addr);
}

__attribute__((always_inline)) static inline void
__moe_bank_patch_mode0_run_bases(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr, uint32_t D_addr)
{
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    __moe_bank_patch_mode0_block_bases(B0_addr, B1_addr, D_addr);
}

__attribute__((always_inline)) static inline void
__moe_bank_patch_mode1_block_bases(
    uint32_t B0_addr, uint32_t B1_addr,
    uint32_t D0_addr, uint32_t D1_addr)
{
    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
}

__attribute__((always_inline)) static inline void
__moe_bank_patch_mode1_run_bases(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr,
    uint32_t D0_addr, uint32_t D1_addr)
{
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    __moe_bank_patch_mode1_block_bases(B0_addr, B1_addr, D0_addr, D1_addr);
}

static inline void __moe_bank_configure_mode0(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr, uint32_t D_addr,
    uint32_t K, uint32_t N, uint32_t array_shape,
    uint32_t rscl_mult, uint32_t rscl_shift)
{
    uint32_t meshRow = __moe_dyn_shape_m(array_shape);
    uint32_t panel_span = (K / 4u) * MOE_BANK_TCDM_ROW_BYTES;
    int32_t Asl[2] = {8, 16};
    int32_t Atb[6] = {(int32_t)K, (int32_t)N, 1, 1, 1, 1};
    int32_t Ats[6] = {MOE_BANK_TCDM_ROW_BYTES, 0, 0, 0, 0, 0};
    int32_t Bsl[2] = {8, (int32_t)panel_span};
    int32_t Btb[4] = {4, (int32_t)(K / 4u), (int32_t)N, 1};
    int32_t Bts[4] = {
        16, MOE_BANK_TCDM_ROW_BYTES,
        (int32_t)((1u << array_shape) * panel_span), 0};
    int32_t Dsl[1] = {8};
    int32_t Dtb[4];
    int32_t Dts[4];
    if (array_shape == 0u) {
        Dtb[0] = 1; Dtb[1] = 8; Dtb[2] = 2; Dtb[3] = (int32_t)(N / 2u);
        Dts[0] = 8; Dts[1] = 16; Dts[2] = 8; Dts[3] = 512;
    } else if (array_shape == 1u) {
        Dtb[0] = 2; Dtb[1] = 4; Dtb[2] = 1; Dtb[3] = (int32_t)N;
        Dts[0] = 8; Dts[1] = 16; Dts[2] = 0; Dts[3] = 512;
    } else {
        Dtb[0] = 2; Dtb[1] = 2; Dtb[2] = 2; Dtb[3] = (int32_t)N;
        Dts[0] = 8; Dts[1] = 512; Dts[2] = 16; Dts[3] = 1024;
    }
    int32_t disabled_bound[4] = {0, 0, 0, 0};
    int32_t disabled_stride[4] = {0, 0, 0, 0};
    int32_t chan_a[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape)};
    int32_t chan_b[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape)};
    int32_t chan_d[1] = {1};
    int32_t chan_off[1] = {0};
    moe_set_dual_versacore_streamer_csr(
        A_addr, Asl, Atb, Ats, 0, chan_a,
        B0_addr, Bsl, Btb, Bts, 0, chan_b,
        B1_addr, Bsl, Btb, Bts, 0, chan_b,
        D_addr, Dsl, Dtb, Dts, 0, chan_d,
        D_addr, Dsl, disabled_bound, disabled_stride, 0, chan_off);
    moe_set_dual_versacore_mode(0u);
    moe_set_dual_versacore_csr(1u, K, N, 0u, array_shape, 0u);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale_mul(0, rscl_mult, 0, rscl_shift);
}

static inline void __moe_bank_configure_mode1(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr,
    uint32_t D0_addr, uint32_t D1_addr,
    uint32_t K, uint32_t N, uint32_t array_shape,
    uint32_t rscl_mult, uint32_t rscl_shift)
{
    uint32_t meshRow = __moe_dyn_shape_m(array_shape);
    uint32_t meshCol = __moe_dyn_meshcol(array_shape);
    uint32_t panel_span = (K / 4u) * MOE_BANK_TCDM_ROW_BYTES;
    int32_t Asl[2] = {8, 16};
    int32_t Atb[6] = {(int32_t)K, (int32_t)N, 1, 1, 1, 1};
    int32_t Ats[6] = {MOE_BANK_TCDM_ROW_BYTES, 0, 0, 0, 0, 0};
    int32_t Bsl[2] = {8, (int32_t)panel_span};
    int32_t Btb[4] = {4, (int32_t)(K / 4u), (int32_t)N, 1};
    int32_t Bts[4] = {
        16, MOE_BANK_TCDM_ROW_BYTES,
        (int32_t)((1u << array_shape) * panel_span), 0};
    int32_t Dsl[1] = {8};
    int32_t Dtb[4] = {
        (int32_t)(meshCol / 4u), (int32_t)meshRow, (int32_t)N, 1};
    int32_t Dts[4] = {
        MOE_BANK_TCDM_ROW_BYTES, 8,
        (int32_t)((meshCol / 4u) * MOE_BANK_TCDM_ROW_BYTES), 0};
    int32_t chan_a[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape)};
    int32_t chan_b[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape)};
    int32_t chan_d[1] = {1};
    moe_set_dual_versacore_streamer_csr(
        A_addr, Asl, Atb, Ats, 0, chan_a,
        B0_addr, Bsl, Btb, Bts, 0, chan_b,
        B1_addr, Bsl, Btb, Bts, 0, chan_b,
        D0_addr, Dsl, Dtb, Dts, 0, chan_d,
        D1_addr, Dsl, Dtb, Dts, 0, chan_d);
    moe_set_dual_versacore_mode(1u);
    moe_set_dual_versacore_csr(1u, K, N, 0u, array_shape, 0u);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
}

static inline void __moe_bank_configure_mode1_phase(
    uint32_t A_addr, uint32_t B0_addr, uint32_t B1_addr,
    uint32_t D0_addr, uint32_t D1_addr,
    uint32_t K, uint32_t N, uint32_t array_shape,
    uint32_t phase_blocks, uint32_t B_phase_stride,
    uint32_t D_phase_stride, uint32_t rscl_mult, uint32_t rscl_shift)
{
    uint32_t meshRow = __moe_dyn_shape_m(array_shape);
    uint32_t meshCol = __moe_dyn_meshcol(array_shape);
    uint32_t panel_span = (K / 4u) * MOE_BANK_TCDM_ROW_BYTES;
    int32_t Asl[2] = {8, 16};
    int32_t Atb[6] = {
        (int32_t)K, (int32_t)N, (int32_t)phase_blocks, 1, 1, 1};
    int32_t Ats[6] = {MOE_BANK_TCDM_ROW_BYTES, 0, 0, 0, 0, 0};
    int32_t Bsl[2] = {8, (int32_t)panel_span};
    int32_t Btb[4] = {
        4, (int32_t)(K / 4u), (int32_t)N, (int32_t)phase_blocks};
    int32_t Bts[4] = {
        16, MOE_BANK_TCDM_ROW_BYTES,
        (int32_t)((1u << array_shape) * panel_span),
        (int32_t)B_phase_stride};
    int32_t Dsl[1] = {8};
    int32_t Dtb[4] = {
        (int32_t)(meshCol / 4u), (int32_t)meshRow,
        (int32_t)N, (int32_t)phase_blocks};
    int32_t Dts[4] = {
        MOE_BANK_TCDM_ROW_BYTES, 8,
        (int32_t)((meshCol / 4u) * MOE_BANK_TCDM_ROW_BYTES),
        (int32_t)D_phase_stride};
    int32_t chan_a[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape)};
    int32_t chan_b[1] = {(int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape)};
    int32_t chan_d[1] = {1};
    moe_set_dual_versacore_streamer_csr(
        A_addr, Asl, Atb, Ats, 0, chan_a,
        B0_addr, Bsl, Btb, Bts, 0, chan_b,
        B1_addr, Bsl, Btb, Bts, 0, chan_b,
        D0_addr, Dsl, Dtb, Dts, 0, chan_d,
        D1_addr, Dsl, Dtb, Dts, 0, chan_d);
    moe_set_dual_versacore_mode(1u);
    moe_set_dual_versacore_csr(
        1u, K, N * phase_blocks, 0u, array_shape, 0u);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
}

static inline uint32_t __moe_bank_weight_block_addr(
    uint32_t ping_base, uint32_t block, uint32_t payload_bytes);

static inline uint32_t __moe_bank_down_weight_block_addr(
    uint32_t ping_base, uint32_t block, uint32_t payload_bytes);

static inline uint32_t __moe_dyn_input_base(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return (MOE_DYN_CTRL_SLOT_ID(cfg->ctrl) & 1u) != 0u ?
        st->l1_d_addr : st->l1_a_addr;
}

static inline uint32_t __moe_dyn_intermediate_base(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return (MOE_DYN_CTRL_SLOT_ID(cfg->ctrl) & 1u) != 0u ?
        st->l1_a_addr : st->l1_d_addr;
}

static inline uint32_t __moe_dyn_output_base(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    return (MOE_DYN_CTRL_SLOT_ID(cfg->ctrl) & 1u) != 0u ?
        st->l1_d_addr : st->l1_down_d_addr;
}

static inline uint32_t __moe_csr_stage_is_prepared(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    uint32_t stage)
{
    asm volatile("fence rw, rw" ::: "memory");
    return __moe_s1_dma_ctrl(blk)->csr_prepared_stage == stage;
}

static inline void __moe_csr_publish_prepared(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    uint32_t stage)
{
    __moe_pipeline_publish(
        &__moe_s1_dma_ctrl(blk)->csr_prepared_stage, stage);
}

static inline uint32_t __moe_prepare_s2_csr(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    const __snax_bingo_moe_dyn_s2_call_args_t *call = &cfg->s2_call;
    if (call->valid == 0u) return 0u;

    uint32_t token = call->token_start;
    uint32_t n_tiles =
        __moe_dyn_stage_block_n(call->N, st->s1_block_count);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    __moe_bank_configure_mode0(
        __moe_bank_a_addr(
            __moe_dyn_input_base(cfg, st), token, st->A_token_bytes),
        __moe_bank_weight_block_addr(
            st->l1_b_gate_addr, 0u, st->indiv_B_block_stride),
        __moe_bank_weight_block_addr(
            st->l1_b_up_addr, 0u, st->indiv_B_block_stride),
        __moe_bank_mode0_output_addr(
            __moe_dyn_intermediate_base(cfg, st), token, 0u,
            st->indiv_N_per_block,
            st->s1_block_count),
        st->indiv_K1, n_tiles, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S2);
    return 1u;
}

static inline uint32_t __moe_prepare_s3_csr(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st)
{
    const __snax_bingo_moe_dyn_s3_call_args_t *call = &cfg->s3_call[0];
    if (call->valid == 0u) return 0u;

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    __moe_bank_configure_mode1(
        __moe_dyn_intermediate_base(cfg, st),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr, 0u, st->indiv_down_B_block_stride),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr + 64u, 0u,
            st->indiv_down_B_block_stride),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st), 0u, 0u,
            st->indiv_down_N_per_block,
            st->s3_block_count),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st) + 64u, 0u, 0u,
            st->indiv_down_N_per_block, st->s3_block_count),
        st->indiv_down_K1, call->N, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S3);
    return 1u;
}

static inline void __moe_configure_s4_phase(
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t token, uint32_t phase)
{
    const __snax_bingo_moe_dyn_s4_call_args_t *call = &cfg->s4_call;
    uint32_t n_tiles = __moe_dyn_stage_block_n(
        call->N, st->s3_block_count);
    uint32_t phase_blocks = __moe_s4_blocks_in_phase(
        st->s3_block_count, phase);
    uint32_t b_phase_stride = st->indiv_down_B_block_stride *
        (MOE_BANK_TCDM_ROW_BYTES / MOE_BANK_WEIGHT_ROW_BYTES);
    uint32_t d_phase_stride =
        2u * (st->indiv_down_N_per_block / 4u) *
        MOE_BANK_TCDM_ROW_BYTES;

    __moe_bank_configure_mode1_phase(
        __moe_bank_mode0_output_addr(
            __moe_dyn_intermediate_base(cfg, st), token, 0u,
            st->indiv_N_per_block, st->s1_block_count),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr, phase, st->indiv_down_B_block_stride),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr + 64u, phase,
            st->indiv_down_B_block_stride),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st), token, phase,
            st->indiv_down_N_per_block, st->s3_block_count),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st) + 64u, token, phase,
            st->indiv_down_N_per_block, st->s3_block_count),
        st->indiv_down_K1, n_tiles, call->array_shape, phase_blocks,
        b_phase_stride, d_phase_stride,
        st->rescale_mult, st->rescale_shift);
}

static inline uint32_t __moe_prepare_s4_csr(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t s4_csr_layout)
{
    const __snax_bingo_moe_dyn_s4_call_args_t *call = &cfg->s4_call;
    if (call->valid == 0u) return 0u;

    uint32_t token = call->token_start;
    uint32_t n_tiles =
        __moe_dyn_stage_block_n(call->N, st->s3_block_count);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    uint32_t block = s4_csr_layout != MOE_S4_CSR_LAYOUT_SEQUENTIAL ?
        __moe_s4_block_initial_phase(st) : 0u;
    if (s4_csr_layout == MOE_S4_CSR_LAYOUT_PHASE_BATCHED) {
        __moe_configure_s4_phase(cfg, st, token, block);
        BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
        __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S4);
        return 1u;
    }
    __moe_bank_configure_mode1(
        __moe_bank_mode0_output_addr(
            __moe_dyn_intermediate_base(cfg, st), token, 0u,
            st->indiv_N_per_block,
            st->s1_block_count),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr, block,
            st->indiv_down_B_block_stride),
        __moe_bank_down_weight_block_addr(
            st->l1_b_down_addr + 64u, block,
            st->indiv_down_B_block_stride),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st), token, block,
            st->indiv_down_N_per_block, st->s3_block_count),
        __moe_bank_mode1_output_addr(
            __moe_dyn_output_base(cfg, st) + 64u, token, block,
            st->indiv_down_N_per_block, st->s3_block_count),
        st->indiv_down_K1, n_tiles, call->array_shape,
        st->rescale_mult, st->rescale_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    __moe_csr_publish_prepared(blk, MOE_CSR_PREPARED_S4);
    return 1u;
}

static inline void __moe_prepare_after_s1(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t s4_csr_layout)
{
    if (__moe_prepare_s2_csr(blk, cfg, st) != 0u) return;
    if (__moe_prepare_s3_csr(blk, cfg, st) != 0u) return;
    (void)__moe_prepare_s4_csr(blk, cfg, st, s4_csr_layout);
}

static inline void __moe_prepare_after_s2(
    const __snax_bingo_kernel_moe_dynamic_expert_block_args_t *blk,
    const __snax_bingo_kernel_moe_dynamic_expert_args_t *cfg,
    const __snax_bingo_moe_dynamic_expert_static_args_t *st,
    uint32_t s4_csr_layout)
{
    if (__moe_prepare_s3_csr(blk, cfg, st) != 0u) return;
    (void)__moe_prepare_s4_csr(blk, cfg, st, s4_csr_layout);
}
