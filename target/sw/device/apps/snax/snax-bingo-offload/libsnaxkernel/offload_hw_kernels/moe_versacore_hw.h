// Low-level VersaCore and streamer CSR operations.
#pragma once

static inline void moe_set_minimal_streamer_cfg_with_silu(
    uint32_t A_addr, uint32_t B_addr, uint32_t C_addr,
    uint32_t D_addr, uint32_t silu)
{
    set_minimal_streamer_cfg(A_addr, B_addr, C_addr, D_addr);
#ifdef READER_WRITER_EXTENSION_1_CSR_BASE
    csrw_ss(READER_WRITER_EXTENSION_1_CSR_BASE, (silu << 1));
#else
    (void)silu;
#endif
}
__attribute__((always_inline)) static inline void
moe_set_dual_versacore_streamer_csr(
    uint32_t A_addr,  int32_t *Aslstride,  int32_t *Atlbound,  int32_t *Atlstride,
    int32_t remap_A,  int32_t *chan_en_A,
    uint32_t B0_addr, int32_t *B0slstride, int32_t *B0tlbound, int32_t *B0tlstride,
    int32_t remap_B0, int32_t *chan_en_B0,
    uint32_t B1_addr, int32_t *B1slstride, int32_t *B1tlbound, int32_t *B1tlstride,
    int32_t remap_B1, int32_t *chan_en_B1,
    uint32_t D0_addr, int32_t *D0slstride, int32_t *D0tlbound, int32_t *D0tlstride,
    int32_t remap_D0, int32_t *chan_en_D0,
    uint32_t D1_addr, int32_t *D1slstride, int32_t *D1tlbound, int32_t *D1tlstride,
    int32_t remap_D1, int32_t *chan_en_D1)
{
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    csrw_ss(S_STRIDE_BASE_READER_0+0, Aslstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_0+1, Aslstride[1]);
    csrw_ss(T_BOUND_BASE_READER_0+0, Atlbound[0]);
    csrw_ss(T_BOUND_BASE_READER_0+1, Atlbound[1]);
    csrw_ss(T_BOUND_BASE_READER_0+2, Atlbound[2]);
    csrw_ss(T_BOUND_BASE_READER_0+3, Atlbound[3]);
    csrw_ss(T_BOUND_BASE_READER_0+4, Atlbound[4]);
    csrw_ss(T_BOUND_BASE_READER_0+5, Atlbound[5]);
    csrw_ss(T_STRIDE_BASE_READER_0+0, Atlstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_0+1, Atlstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_0+2, Atlstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_0+3, Atlstride[3]);
    csrw_ss(T_STRIDE_BASE_READER_0+4, Atlstride[4]);
    csrw_ss(T_STRIDE_BASE_READER_0+5, Atlstride[5]);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, remap_A);
#else
    (void)remap_A;
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, chan_en_A[0]);

    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(S_STRIDE_BASE_READER_1+0, B0slstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_1+1, B0slstride[1]);
    csrw_ss(T_BOUND_BASE_READER_1+0, B0tlbound[0]);
    csrw_ss(T_BOUND_BASE_READER_1+1, B0tlbound[1]);
    csrw_ss(T_BOUND_BASE_READER_1+2, B0tlbound[2]);
    csrw_ss(T_BOUND_BASE_READER_1+3, B0tlbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_1+0, B0tlstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_1+1, B0tlstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_1+2, B0tlstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_1+3, B0tlstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, remap_B0);
#else
    (void)remap_B0;
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, chan_en_B0[0]);

    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(S_STRIDE_BASE_READER_2+0, B1slstride[0]);
    csrw_ss(S_STRIDE_BASE_READER_2+1, B1slstride[1]);
    csrw_ss(T_BOUND_BASE_READER_2+0, B1tlbound[0]);
    csrw_ss(T_BOUND_BASE_READER_2+1, B1tlbound[1]);
    csrw_ss(T_BOUND_BASE_READER_2+2, B1tlbound[2]);
    csrw_ss(T_BOUND_BASE_READER_2+3, B1tlbound[3]);
    csrw_ss(T_STRIDE_BASE_READER_2+0, B1tlstride[0]);
    csrw_ss(T_STRIDE_BASE_READER_2+1, B1tlstride[1]);
    csrw_ss(T_STRIDE_BASE_READER_2+2, B1tlstride[2]);
    csrw_ss(T_STRIDE_BASE_READER_2+3, B1tlstride[3]);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, remap_B1);
#else
    (void)remap_B1;
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, chan_en_B1[0]);

    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_0, D0slstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+0, D0tlbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_0+1, D0tlbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_0+2, D0tlbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_0+3, D0tlbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, D0tlstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, D0tlstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, D0tlstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, D0tlstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, remap_D0);
#else
    (void)remap_D0;
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, chan_en_D0[0]);

    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_1, D1slstride[0]);
    csrw_ss(T_BOUND_BASE_WRITER_1+0, D1tlbound[0]);
    csrw_ss(T_BOUND_BASE_WRITER_1+1, D1tlbound[1]);
    csrw_ss(T_BOUND_BASE_WRITER_1+2, D1tlbound[2]);
    csrw_ss(T_BOUND_BASE_WRITER_1+3, D1tlbound[3]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, D1tlstride[0]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, D1tlstride[1]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, D1tlstride[2]);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, D1tlstride[3]);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, remap_D1);
#else
    (void)remap_D1;
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, chan_en_D1[0]);
}

static inline void moe_set_dual_versacore_csr(
    uint32_t take_in_new_c, uint32_t a_b_input_times_one_output,
    uint32_t output_times, uint32_t subtractions,
    uint32_t array_shape, uint32_t data_type)
{
    csrw_ss(MOE_DUAL_VC_OVERWRITE_ACCUM, take_in_new_c);
    csrw_ss(MOE_DUAL_VC_ACCUM_BOUND, a_b_input_times_one_output);
    csrw_ss(MOE_DUAL_VC_OUTPUT_BOUND, output_times);
    csrw_ss(MOE_DUAL_VC_SUBTRACTIONS, subtractions);
    csrw_ss(MOE_DUAL_VC_ARRAY_SHAPE_CFG, array_shape);
    csrw_ss(MOE_DUAL_VC_DATA_TYPE_CFG, data_type);
}

static inline void moe_set_dual_versacore_mode(uint32_t mode)
{
    csrw_ss(MOE_DUAL_VC_MODE, mode);
}

static inline void moe_set_dual_versacore_rescale0(
    int32_t input_zp, uint32_t multiplier, int32_t output_zp, uint32_t shift)
{
    csrw_ss(MOE_DUAL_VC_RESCALE0_INPUT_ZP, (uint32_t)input_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE0_MULTIPLIER, multiplier);
    csrw_ss(MOE_DUAL_VC_RESCALE0_OUTPUT_ZP, (uint32_t)output_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE0_SHIFT, shift);
}

static inline void moe_set_dual_versacore_rescale1(
    int32_t input_zp, uint32_t multiplier, int32_t output_zp, uint32_t shift)
{
    csrw_ss(MOE_DUAL_VC_RESCALE1_INPUT_ZP, (uint32_t)input_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE1_MULTIPLIER, multiplier);
    csrw_ss(MOE_DUAL_VC_RESCALE1_OUTPUT_ZP, (uint32_t)output_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE1_SHIFT, shift);
}

static inline void moe_set_dual_versacore_rescale_mul(
    int32_t input_zp, uint32_t multiplier, int32_t output_zp, uint32_t shift)
{
    csrw_ss(MOE_DUAL_VC_RESCALE_MUL_INPUT_ZP, (uint32_t)input_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE_MUL_MULTIPLIER, multiplier);
    csrw_ss(MOE_DUAL_VC_RESCALE_MUL_OUTPUT_ZP, (uint32_t)output_zp);
    csrw_ss(MOE_DUAL_VC_RESCALE_MUL_SHIFT, shift);
}

static inline void moe_start_dual_vc_and_streamer(void)
{
    csrw_ss(STREAMER_START_CSR, 1);
    csrw_ss(MOE_DUAL_VC_START, 1);
}

static inline void moe_wait_dual_vc_and_streamer(void)
{
    // Match the standalone VersaCore protocol: drop each start signal before
    // waiting for its engine to become idle.  In particular, Mode-0 must
    // release STREAMER_START_CSR before fused L15 code reprograms and restarts
    // the streamer for Mode-1.
    csrw_ss(MOE_DUAL_VC_START, 0);
    csrw_ss(MOE_DUAL_VC_START, 0);
    while (csrr_ss(MOE_DUAL_VC_BUSY)) {}
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    while (csrr_ss(STREAMER_BUSY_CSR)) {}
}
