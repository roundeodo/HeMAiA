// Router and generic dual-VersaCore device kernels.
#pragma once

// gemm_minimal_silu: like gemm_minimal but also updates the SiLU extension CSR.
// Saves ~45 CSR writes compared to gemm_full when shape (M,K,N,strides) is unchanged.
// Use when: same shape as a preceding gemm_full, only addresses + silu_enable differ.
// arg[0]=A_addr, arg[1]=B_addr, arg[2]=C_addr, arg[3]=D_addr, arg[4]=silu_enable
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_gemm_minimal_silu(void *arg)
{
    if (snrt_cluster_core_idx() != 0){
        printf_safe("[Cluster %d Core %d]: Error! Bingo GEMM minimal_silu should be called from core 0!\r\n", snrt_cluster_idx(), snrt_cluster_core_idx());
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t A_addr    = ((uint32_t *)arg)[0];
    uint32_t B_addr    = ((uint32_t *)arg)[1];
    uint32_t C_addr    = ((uint32_t *)arg)[2];
    uint32_t D_addr    = ((uint32_t *)arg)[3];
    uint32_t silu_enable = ((uint32_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_START);
    moe_set_minimal_streamer_cfg_with_silu(A_addr, B_addr, C_addr, D_addr, silu_enable);
    start_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_START);
    wait_versacore_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_MIN_RUN_END);
    VERSACORE_DEBUG_PRINT("Bingo GEMM Minimal SiLU Kernel Compute Done!\r\n");
    return BINGO_RET_SUCC;
}
// ============================================================
// Dual-VersaCore GEMM kernel (Mode 1: A@B0/B1 -> D0/D1)
// INT16 A x INT4 packed B -> INT16 D
// Args (uint32_t array, 11 fields):
//   [0] A_addr  [1] B0_addr  [2] B1_addr  [3] D0_addr  [4] D1_addr
//   [5] M  [6] K  [7] N
//   [8] array_shape  [9] rescale_mult  [10] rescale_shift
// ============================================================
__attribute__((always_inline)) static inline uint32_t
__moe_dual_vc_gemm_full_params(
    uint32_t A_addr,
    uint32_t B0_addr,
    uint32_t B1_addr,
    uint32_t D0_addr,
    uint32_t D1_addr,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t array_shape,
    uint32_t rscl_mult,
    uint32_t rscl_shift)
{
    uint32_t meshRow = MOE_DUAL_VC_MESH_ROW_0 >> array_shape;
    uint32_t tileSize = MOE_DUAL_VC_TILE_SIZE_0;
    uint32_t meshCol = MOE_DUAL_VC_MESH_COL_0 << array_shape;

    // A: INT16 token-contiguous rows.  multi_cluster_MoE routes the same
    // input_A buffer used by the dynamic path: each token has K*tileSize*2
    // payload bytes in a larger physical row. The GEMM reader streams only the
    // payload and uses the L15 row stride to skip the trailing address gap.
    uint32_t a_row_stride = K * tileSize * 2u +
        BINGO_MOE_L15_ROW_GAP_BYTES;
    int32_t Asl[2]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)a_row_stride };
    int32_t Atb[6]       = { (int32_t)K, (int32_t)N, (int32_t)M, 1, 1, 1 };
    int32_t Ats[6]       = { (int32_t)(tileSize * 2), 0,
                              (int32_t)(meshRow * a_row_stride), 0, 0, 0 };
    int32_t chan_en_A[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape) };

    // B0: INT4 packed, 4-dim temporal, 2-dim spatial.  Use the same canonical
    // mode-1 B reader layout as the L15/reference kernels: per K tile the
    // stream advances by 16B, while the second spatial stride covers the full
    // K section.  A zero second spatial stride aliases active B channels and
    // produces router scores that are self-consistent for TopK but not equal
    // to the datagen GEMM.
    uint32_t n_b_chan      = (array_shape == 0u) ? 2u : (array_shape == 1u) ? 4u : 8u;
    int32_t B_stream_bytes = (int32_t)(n_b_chan * (MOE_DUAL_VC_BANK_WIDTH / 8));
    uint32_t b_k_section   = K * tileSize * 2u;
    int32_t B0sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B0tb[4]        = { (int32_t)K, (int32_t)N, (int32_t)M, 1 };
    int32_t B0ts[4]        = { (int32_t)(tileSize * 2u), (int32_t)(K * B_stream_bytes), 0, 0 };
    int32_t chan_en_B0[1]  = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // B1: INT4 packed, same canonical mode-1 layout as B0.
    int32_t B1sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B1tb[4]        = { (int32_t)K, (int32_t)N, (int32_t)M, 1 };
    int32_t B1ts[4]        = { (int32_t)(tileSize * 2u), (int32_t)(K * B_stream_bytes), 0, 0 };
    int32_t chan_en_B1[1]  = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // D0: INT16, 4-dim temporal (VC0 output, 4-lane postproc: 1ch, 8 bytes/beat)
    // NOTE: For L15 layout mode-1 down-proj, use __snax_bingo_kernel_dual_vc_l15_moe_full.
    int32_t D0sl[1]       = { 8 };
    int32_t D0tb[4]       = { 8, (int32_t)N, (int32_t)M, 1 };
    int32_t D0ts[4]       = { 8, 64, (int32_t)(N * 64), 0 };
    int32_t chan_en_D0[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_D };

    // D1: INT16, 4-dim temporal (VC1 output, 4-lane postproc: 1ch, 8 bytes/beat)
    int32_t D1sl[1]       = { 8 };
    int32_t D1tb[4]       = { 8, (int32_t)N, (int32_t)M, 1 };
    int32_t D1ts[4]       = { 8, 64, (int32_t)(N * 64), 0 };
    int32_t chan_en_D1[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_D };

#if BINGO_DEBUG_LEVEL >= 1
    if (M == 4u && K == 128u && N == 1u && array_shape == 0u) {
        volatile int16_t *A_dbg = (volatile int16_t *)(uintptr_t)A_addr;
        volatile uint8_t *B0_dbg = (volatile uint8_t *)(uintptr_t)B0_addr;
        volatile uint8_t *B1_dbg = (volatile uint8_t *)(uintptr_t)B1_addr;
        printf_safe("[ROUTER_GEMM_DBG] args A=0x%08x B0=0x%08x B1=0x%08x D0=0x%08x D1=0x%08x M=%u K=%u N=%u shape=%u\r\n",
                    A_addr, B0_addr, B1_addr, D0_addr, D1_addr, M, K, N, array_shape);
        printf_safe("[ROUTER_GEMM_DBG] D_layout D0_row_stride=64 D1_minus_D0=%u D_tile_bytes=%u\r\n",
                    (uint32_t)(D1_addr - D0_addr), (uint32_t)(M * N * 64u));
        printf_safe("[ROUTER_GEMM_DBG] A16[0..7]=%d %d %d %d %d %d %d %d\r\n",
                    A_dbg[0], A_dbg[1], A_dbg[2], A_dbg[3],
                    A_dbg[4], A_dbg[5], A_dbg[6], A_dbg[7]);
        printf_safe("[ROUTER_GEMM_DBG] B0u8[0..15]=%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\r\n",
                    B0_dbg[0], B0_dbg[1], B0_dbg[2], B0_dbg[3],
                    B0_dbg[4], B0_dbg[5], B0_dbg[6], B0_dbg[7],
                    B0_dbg[8], B0_dbg[9], B0_dbg[10], B0_dbg[11],
                    B0_dbg[12], B0_dbg[13], B0_dbg[14], B0_dbg[15]);
        printf_safe("[ROUTER_GEMM_DBG] B1u8[0..15]=%u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\r\n",
                    B1_dbg[0], B1_dbg[1], B1_dbg[2], B1_dbg[3],
                    B1_dbg[4], B1_dbg[5], B1_dbg[6], B1_dbg[7],
                    B1_dbg[8], B1_dbg[9], B1_dbg[10], B1_dbg[11],
                    B1_dbg[12], B1_dbg[13], B1_dbg[14], B1_dbg[15]);
    }
#endif

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    moe_set_dual_versacore_streamer_csr(
        A_addr,  Asl,  Atb,  Ats,  0, chan_en_A,
        B0_addr, B0sl, B0tb, B0ts, 0, chan_en_B0,
        B1_addr, B1sl, B1tb, B1ts, 0, chan_en_B1,
        D0_addr, D0sl, D0tb, D0ts, 0, chan_en_D0,
        D1_addr, D1sl, D1tb, D1ts, 0, chan_en_D1);

    moe_set_dual_versacore_mode(1);   // Mode 1 = GEMM

    moe_set_dual_versacore_csr(
        1,       // take_in_new_c = start fresh
        K,       // accum_bound
        N * M,   // output_times
        0,       // subtractions
        array_shape,
        0);      // data_type

    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
#if BINGO_DEBUG_LEVEL >= 1
    if (M == 4u && K == 128u && N == 1u && array_shape == 0u) {
        volatile int16_t *D0_dbg = (volatile int16_t *)(uintptr_t)D0_addr;
        volatile int16_t *D1_dbg = (volatile int16_t *)(uintptr_t)D1_addr;
        printf_safe("[ROUTER_GEMM_DBG] D0_16[0..7]=%d %d %d %d %d %d %d %d\r\n",
                    D0_dbg[0], D0_dbg[1], D0_dbg[2], D0_dbg[3],
                    D0_dbg[4], D0_dbg[5], D0_dbg[6], D0_dbg[7]);
        printf_safe("[ROUTER_GEMM_DBG] D1_16[0..7]=%d %d %d %d %d %d %d %d\r\n",
                    D1_dbg[0], D1_dbg[1], D1_dbg[2], D1_dbg[3],
                    D1_dbg[4], D1_dbg[5], D1_dbg[6], D1_dbg[7]);
    }
#endif
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_dual_vc_gemm_full(void *arg)
{
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_dual_vc_gemm_full_args_t *cfg =
        (const __snax_bingo_kernel_dual_vc_gemm_full_args_t *)arg;
    if (cfg->array_shape > 2u) return BINGO_RET_FAIL;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    return __moe_dual_vc_gemm_full_params(
        cfg->input_A_addr, cfg->input_B0_addr, cfg->input_B1_addr,
        cfg->output_D0_addr, cfg->output_D1_addr,
        cfg->M, cfg->K, cfg->N, cfg->array_shape,
        cfg->rescale_mult, cfg->rescale_shift);
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_router_gemm_s0(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_router_gemm_s0_args_t *cfg =
        (const __snax_bingo_kernel_moe_router_gemm_s0_args_t *)arg;
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    return __moe_dual_vc_gemm_full_params(
        cfg->input_A_addr, cfg->input_B0_addr, cfg->input_B1_addr,
        cfg->output_D0_addr, cfg->output_D1_addr,
        cfg->M, cfg->K, cfg->N, 0u,
        cfg->rescale_mult, cfg->rescale_shift);
}
