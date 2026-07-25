// Legacy full-L1.5 streamer configuration and fused MoE kernel.
#pragma once

__attribute__((always_inline)) static inline void
__l15_cfg_mode0_streamer(
    const __snax_bingo_moe_l15_shape_cfg_t *cfg,
    uint32_t tcdm_base)
{
    /* Reader 0: A (INT16, 6-dim temporal, 2-dim spatial) */
    csrw_ss(BASE_PTR_READER_0_LOW,    tcdm_base + (uint32_t)cfg->delta_local_a);
    csrw_ss(S_STRIDE_BASE_READER_0+0, (uint32_t)cfg->mode0_A_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_0+1, (uint32_t)cfg->mode0_A_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_0+0,  (uint32_t)cfg->mode0_A_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_0+1,  (uint32_t)cfg->mode0_A_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_0+2,  (uint32_t)cfg->mode0_A_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_0+3,  (uint32_t)cfg->mode0_A_tbound[3]);
    csrw_ss(T_BOUND_BASE_READER_0+4,  (uint32_t)cfg->mode0_A_tbound[4]);
    csrw_ss(T_BOUND_BASE_READER_0+5,  (uint32_t)cfg->mode0_A_tbound[5]);
    csrw_ss(T_STRIDE_BASE_READER_0+0, (uint32_t)cfg->mode0_A_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_0+1, (uint32_t)cfg->mode0_A_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_0+2, (uint32_t)cfg->mode0_A_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_0+3, (uint32_t)cfg->mode0_A_tstride[3]);
    csrw_ss(T_STRIDE_BASE_READER_0+4, (uint32_t)cfg->mode0_A_tstride[4]);
    csrw_ss(T_STRIDE_BASE_READER_0+5, (uint32_t)cfg->mode0_A_tstride[5]);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, (uint32_t)cfg->A_channel_en[0]);
    /* Reader 1: B0 = gate weight (INT4 packed, 4-dim temporal, 2-dim spatial) */
    csrw_ss(BASE_PTR_READER_1_LOW,    tcdm_base + (uint32_t)cfg->delta_local_b0);
    csrw_ss(S_STRIDE_BASE_READER_1+0, (uint32_t)cfg->mode0_B_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_1+1, (uint32_t)cfg->mode0_B_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_1+0,  (uint32_t)cfg->mode0_B_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_1+1,  (uint32_t)cfg->mode0_B_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_1+2,  (uint32_t)cfg->mode0_B_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_1+3,  (uint32_t)cfg->mode0_B_tbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_1+0, (uint32_t)cfg->mode0_B_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_1+1, (uint32_t)cfg->mode0_B_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_1+2, (uint32_t)cfg->mode0_B_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_1+3, (uint32_t)cfg->mode0_B_tstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, (uint32_t)cfg->B_channel_en[0]);
    /* Reader 2: B1 = up weight (same layout as B0) */
    csrw_ss(BASE_PTR_READER_2_LOW,    tcdm_base + (uint32_t)cfg->delta_local_b1);
    csrw_ss(S_STRIDE_BASE_READER_2+0, (uint32_t)cfg->mode0_B_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_2+1, (uint32_t)cfg->mode0_B_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_2+0,  (uint32_t)cfg->mode0_B_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_2+1,  (uint32_t)cfg->mode0_B_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_2+2,  (uint32_t)cfg->mode0_B_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_2+3,  (uint32_t)cfg->mode0_B_tbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_2+0, (uint32_t)cfg->mode0_B_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_2+1, (uint32_t)cfg->mode0_B_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_2+2, (uint32_t)cfg->mode0_B_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_2+3, (uint32_t)cfg->mode0_B_tstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, (uint32_t)cfg->B_channel_en[0]);
    /* Writer 0: D0 = SwiGLU output */
    csrw_ss(BASE_PTR_WRITER_0_LOW,    tcdm_base + (uint32_t)cfg->delta_local_d0);
    csrw_ss(S_STRIDE_BASE_WRITER_0+0, (uint32_t)cfg->D_sstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+0,  (uint32_t)cfg->mode0_D_tbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+1,  (uint32_t)cfg->mode0_D_tbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_0+2,  (uint32_t)cfg->mode0_D_tbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_0+3,  (uint32_t)cfg->mode0_D_tbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, (uint32_t)cfg->mode0_D_tstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, (uint32_t)cfg->mode0_D_tstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, (uint32_t)cfg->mode0_D_tstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, (uint32_t)cfg->mode0_D_tstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, (uint32_t)cfg->D_channel_en[0]);
    /* Writer 1: disabled in Mode-0 (SwiGLU hw only drives Writer-0) */
    csrw_ss(BASE_PTR_WRITER_1_LOW,    tcdm_base + (uint32_t)cfg->delta_local_d0);
    csrw_ss(S_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_1+0,  0u);
    csrw_ss(T_BOUND_BASE_WRITER_1+1,  0u);
    csrw_ss(T_BOUND_BASE_WRITER_1+2,  0u);
    csrw_ss(T_BOUND_BASE_WRITER_1+3,  0u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, 0u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, 0u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, 0u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, 0u);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, 0u);
}

__attribute__((always_inline)) static inline void
__l15_cfg_mode1_streamer(
    const __snax_bingo_moe_l15_shape_cfg_t *cfg,
    uint32_t tcdm_base)
{
    /* Reader 0: A = Mode-0 SwiGLU output D0 (INT16, 6-dim temporal) */
    csrw_ss(BASE_PTR_READER_0_LOW,    tcdm_base + (uint32_t)cfg->delta_local_d0);
    csrw_ss(S_STRIDE_BASE_READER_0+0, (uint32_t)cfg->mode1_A_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_0+1, (uint32_t)cfg->mode1_A_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_0+0,  (uint32_t)cfg->mode1_A_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_0+1,  (uint32_t)cfg->mode1_A_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_0+2,  (uint32_t)cfg->mode1_A_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_0+3,  (uint32_t)cfg->mode1_A_tbound[3]);
    csrw_ss(T_BOUND_BASE_READER_0+4,  (uint32_t)cfg->mode1_A_tbound[4]);
    csrw_ss(T_BOUND_BASE_READER_0+5,  (uint32_t)cfg->mode1_A_tbound[5]);
    csrw_ss(T_STRIDE_BASE_READER_0+0, (uint32_t)cfg->mode1_A_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_0+1, (uint32_t)cfg->mode1_A_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_0+2, (uint32_t)cfg->mode1_A_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_0+3, (uint32_t)cfg->mode1_A_tstride[3]);
    csrw_ss(T_STRIDE_BASE_READER_0+4, (uint32_t)cfg->mode1_A_tstride[4]);
    csrw_ss(T_STRIDE_BASE_READER_0+5, (uint32_t)cfg->mode1_A_tstride[5]);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, (uint32_t)cfg->A_channel_en[0]);
    /* Reader 1: B0 = W2l (left VC partition, INT4 packed, 4-dim temporal) */
    csrw_ss(BASE_PTR_READER_1_LOW,    tcdm_base + (uint32_t)cfg->delta_local_w2l);
    csrw_ss(S_STRIDE_BASE_READER_1+0, (uint32_t)cfg->mode1_B_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_1+1, (uint32_t)cfg->mode1_B_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_1+0,  (uint32_t)cfg->mode1_B_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_1+1,  (uint32_t)cfg->mode1_B_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_1+2,  (uint32_t)cfg->mode1_B_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_1+3,  (uint32_t)cfg->mode1_B_tbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_1+0, (uint32_t)cfg->mode1_B_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_1+1, (uint32_t)cfg->mode1_B_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_1+2, (uint32_t)cfg->mode1_B_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_1+3, (uint32_t)cfg->mode1_B_tstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, (uint32_t)cfg->B_channel_en[0]);
    /* Reader 2: B1 = W2r (right VC partition, same layout as W2l) */
    csrw_ss(BASE_PTR_READER_2_LOW,    tcdm_base + (uint32_t)cfg->delta_local_w2r);
    csrw_ss(S_STRIDE_BASE_READER_2+0, (uint32_t)cfg->mode1_B_sstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_2+1, (uint32_t)cfg->mode1_B_sstride[1]);
    csrw_ss(T_BOUND_BASE_READER_2+0,  (uint32_t)cfg->mode1_B_tbound[0]);
    csrw_ss(T_BOUND_BASE_READER_2+1,  (uint32_t)cfg->mode1_B_tbound[1]);
    csrw_ss(T_BOUND_BASE_READER_2+2,  (uint32_t)cfg->mode1_B_tbound[2]);
    csrw_ss(T_BOUND_BASE_READER_2+3,  (uint32_t)cfg->mode1_B_tbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_2+0, (uint32_t)cfg->mode1_B_tstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_2+1, (uint32_t)cfg->mode1_B_tstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_2+2, (uint32_t)cfg->mode1_B_tstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_2+3, (uint32_t)cfg->mode1_B_tstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, (uint32_t)cfg->B_channel_en[0]);
    /* Writer 0: VC0 down-proj output (mode1_d0) */
    csrw_ss(BASE_PTR_WRITER_0_LOW,    tcdm_base + (uint32_t)cfg->delta_local_mode1_d0);
    csrw_ss(S_STRIDE_BASE_WRITER_0+0, (uint32_t)cfg->D_sstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+0,  (uint32_t)cfg->mode1_D_tbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+1,  (uint32_t)cfg->mode1_D_tbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_0+2,  (uint32_t)cfg->mode1_D_tbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_0+3,  (uint32_t)cfg->mode1_D_tbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, (uint32_t)cfg->mode1_D_tstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, (uint32_t)cfg->mode1_D_tstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, (uint32_t)cfg->mode1_D_tstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, (uint32_t)cfg->mode1_D_tstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, (uint32_t)cfg->D_channel_en[0]);
    /* Writer 1: VC1 down-proj output (mode1_d1) */
    csrw_ss(BASE_PTR_WRITER_1_LOW,    tcdm_base + (uint32_t)cfg->delta_local_mode1_d1);
    csrw_ss(S_STRIDE_BASE_WRITER_1+0, (uint32_t)cfg->D_sstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_1+0,  (uint32_t)cfg->mode1_D_tbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_1+1,  (uint32_t)cfg->mode1_D_tbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_1+2,  (uint32_t)cfg->mode1_D_tbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_1+3,  (uint32_t)cfg->mode1_D_tbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, (uint32_t)cfg->mode1_D_tstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, (uint32_t)cfg->mode1_D_tstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, (uint32_t)cfg->mode1_D_tstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, (uint32_t)cfg->mode1_D_tstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, (uint32_t)cfg->D_channel_en[0]);
}

/* Full-size legacy L1.5 path: Mode 0 consumes A/gate/up and produces the
 * SwiGLU intermediate; Mode 1 consumes that intermediate and down weights. */
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_l15_moe_full(void *arg)
{
    const __snax_bingo_moe_l15_shape_cfg_t *cfg =
        (const __snax_bingo_moe_l15_shape_cfg_t *)(uintptr_t)
        ((uint32_t *)arg)[0];
    uint32_t tcdm_base  = ((uint32_t *)arg)[1];
    uint32_t rscl_mult  = ((uint32_t *)arg)[2];
    uint32_t rscl_shift = ((uint32_t *)arg)[3];

    /* Configure and start Mode 0. */
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG_START);
    __l15_cfg_mode0_streamer(cfg, tcdm_base);
    moe_set_dual_versacore_mode(0);
    moe_set_dual_versacore_csr(
        1, cfg->K_tiles, cfg->N_tiles * cfg->M_tiles, 0, cfg->array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale_mul(0, rscl_mult, 0, rscl_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE0_START);
    moe_start_dual_vc_and_streamer();

    /* Non-START writes fill the staging CSR bank while Mode 0 is active. */
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG1_START);
    __l15_cfg_mode1_streamer(cfg, tcdm_base);
    moe_set_dual_versacore_mode(1);
    uint32_t mode1_output_tiles =
        (uint32_t)cfg->mode1_D_tbound[2] * (uint32_t)cfg->mode1_D_tbound[3];
    moe_set_dual_versacore_csr(
        1, cfg->K1, mode1_output_tiles, 0, cfg->array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_CFG1_END);

    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE0_END);

    /* Publish the preloaded Mode 1 bank. */
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE1_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_L15_FULL_MODE1_END);

    return BINGO_RET_SUCC;
}
