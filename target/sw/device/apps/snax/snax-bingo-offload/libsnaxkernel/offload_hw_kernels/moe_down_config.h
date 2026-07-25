// Shared down-projection VersaCore configuration.
#pragma once

__attribute__((always_inline)) static inline void
__moe_configure_down(
    uint32_t A_addr,
    uint32_t B0_addr,
    uint32_t B1_addr,
    uint32_t D0_addr,
    uint32_t D1_addr,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t array_shape,
    uint32_t b_k_section,
    uint32_t b_n_stride,
    uint32_t a_m_stride,
    uint32_t d_m_stride,
    uint32_t d_row_stride,
    uint32_t rscl_mult,
    uint32_t rscl_shift)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    uint32_t meshRow = __moe_dyn_shape_m(array_shape);
    uint32_t meshCol = __moe_dyn_meshcol(array_shape);
    uint32_t d_beats_per_row = meshCol / 4u;

    uint32_t a_sstride0 = __moe_dyn_mode1_a_sstride0(array_shape);
    uint32_t a_sstride1 = __moe_dyn_mode1_a_sstride1(array_shape);
    uint32_t a_k_stride = __moe_dyn_mode1_a_k_stride(array_shape);
    uint32_t chan_en_A  = MOE_DUAL_VC_CHAN_EN_A(array_shape);
    uint32_t chan_en_B  = MOE_DUAL_VC_CHAN_EN_B(array_shape);

    /* Dynamic down reads SwiGLU's Mode-0 D0 layout.  Keep this path expanded
     * like the L15 mode1 helper: no local CSR arrays, no loop-based streamer
     * setup, and only runtime-dependent values remain as register operands. */
    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    csrw_ss(S_STRIDE_BASE_READER_0+0, a_sstride0);
    csrw_ss(S_STRIDE_BASE_READER_0+1, a_sstride1);
    csrw_ss(T_BOUND_BASE_READER_0+0, K);
    csrw_ss(T_BOUND_BASE_READER_0+1, N);
    csrw_ss(T_BOUND_BASE_READER_0+2, M);
    csrw_ss(T_BOUND_BASE_READER_0+3, 1u);
    csrw_ss(T_BOUND_BASE_READER_0+4, 1u);
    csrw_ss(T_BOUND_BASE_READER_0+5, 1u);
    csrw_ss(T_STRIDE_BASE_READER_0+0, a_k_stride);
    csrw_ss(T_STRIDE_BASE_READER_0+1, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+2, a_m_stride);
    csrw_ss(T_STRIDE_BASE_READER_0+3, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+4, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+5, 0u);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, chan_en_A);

    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(S_STRIDE_BASE_READER_1+0, MOE_DUAL_VC_BANK_WIDTH / 8u);
    csrw_ss(S_STRIDE_BASE_READER_1+1, b_k_section);
    csrw_ss(T_BOUND_BASE_READER_1+0, K);
    csrw_ss(T_BOUND_BASE_READER_1+1, N);
    csrw_ss(T_BOUND_BASE_READER_1+2, M);
    csrw_ss(T_BOUND_BASE_READER_1+3, 1u);
    csrw_ss(T_STRIDE_BASE_READER_1+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_1+1, b_n_stride);
    csrw_ss(T_STRIDE_BASE_READER_1+2, 0u);
    csrw_ss(T_STRIDE_BASE_READER_1+3, 0u);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, chan_en_B);

    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(S_STRIDE_BASE_READER_2+0, MOE_DUAL_VC_BANK_WIDTH / 8u);
    csrw_ss(S_STRIDE_BASE_READER_2+1, b_k_section);
    csrw_ss(T_BOUND_BASE_READER_2+0, K);
    csrw_ss(T_BOUND_BASE_READER_2+1, N);
    csrw_ss(T_BOUND_BASE_READER_2+2, M);
    csrw_ss(T_BOUND_BASE_READER_2+3, 1u);
    csrw_ss(T_STRIDE_BASE_READER_2+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_2+1, b_n_stride);
    csrw_ss(T_STRIDE_BASE_READER_2+2, 0u);
    csrw_ss(T_STRIDE_BASE_READER_2+3, 0u);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, chan_en_B);

    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_0+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_0+0, d_beats_per_row);
    csrw_ss(T_BOUND_BASE_WRITER_0+1, meshRow);
    csrw_ss(T_BOUND_BASE_WRITER_0+2, N);
    csrw_ss(T_BOUND_BASE_WRITER_0+3, M);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, 8u);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, d_row_stride);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, meshCol * 2u);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, d_m_stride);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, MOE_DUAL_VC_CHAN_EN_D);

    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_1+0, d_beats_per_row);
    csrw_ss(T_BOUND_BASE_WRITER_1+1, meshRow);
    csrw_ss(T_BOUND_BASE_WRITER_1+2, N);
    csrw_ss(T_BOUND_BASE_WRITER_1+3, M);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, d_row_stride);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, meshCol * 2u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, d_m_stride);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, MOE_DUAL_VC_CHAN_EN_D);

    moe_set_dual_versacore_mode(1);
    moe_set_dual_versacore_csr(1, K, N * M, 0, array_shape, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
}
__attribute__((always_inline)) static inline void
__moe_configure_down_shape_c(
    uint32_t A_addr,
    uint32_t B0_addr,
    uint32_t B1_addr,
    uint32_t D0_addr,
    uint32_t D1_addr,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t b_block_count,
    uint32_t b_block_stride,
    uint32_t b_k_section,
    uint32_t b_n_per_block,
    uint32_t b_n_stride,
    uint32_t a_m_stride,
    uint32_t d_m_stride,
    uint32_t d_row_stride,
    uint32_t rscl_mult,
    uint32_t rscl_shift)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);

    csrw_ss(BASE_PTR_READER_0_LOW, A_addr);
    csrw_ss(S_STRIDE_BASE_READER_0+0, 8u);
    csrw_ss(S_STRIDE_BASE_READER_0+1, 32u);
    csrw_ss(T_BOUND_BASE_READER_0+0, K);
    csrw_ss(T_BOUND_BASE_READER_0+1, N);
    csrw_ss(T_BOUND_BASE_READER_0+2, M);
    csrw_ss(T_BOUND_BASE_READER_0+3, 1u);
    csrw_ss(T_BOUND_BASE_READER_0+4, 1u);
    csrw_ss(T_BOUND_BASE_READER_0+5, 1u);
    csrw_ss(T_STRIDE_BASE_READER_0+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_0+1, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+2, a_m_stride);
    csrw_ss(T_STRIDE_BASE_READER_0+3, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+4, 0u);
    csrw_ss(T_STRIDE_BASE_READER_0+5, 0u);
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_0, MOE_DUAL_VC_CHAN_EN_A(2u));

    csrw_ss(BASE_PTR_READER_1_LOW, B0_addr);
    csrw_ss(S_STRIDE_BASE_READER_1+0, MOE_DUAL_VC_BANK_WIDTH / 8u);
    csrw_ss(S_STRIDE_BASE_READER_1+1, b_k_section);
    csrw_ss(T_BOUND_BASE_READER_1+0, K);
    csrw_ss(T_BOUND_BASE_READER_1+1, b_n_per_block);
    csrw_ss(T_BOUND_BASE_READER_1+2, b_block_count);
    csrw_ss(T_BOUND_BASE_READER_1+3, M);
    csrw_ss(T_STRIDE_BASE_READER_1+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_1+1, b_n_stride);
    csrw_ss(T_STRIDE_BASE_READER_1+2, b_block_stride);
    csrw_ss(T_STRIDE_BASE_READER_1+3, 0u);
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_1, MOE_DUAL_VC_CHAN_EN_B(2u));

    csrw_ss(BASE_PTR_READER_2_LOW, B1_addr);
    csrw_ss(S_STRIDE_BASE_READER_2+0, MOE_DUAL_VC_BANK_WIDTH / 8u);
    csrw_ss(S_STRIDE_BASE_READER_2+1, b_k_section);
    csrw_ss(T_BOUND_BASE_READER_2+0, K);
    csrw_ss(T_BOUND_BASE_READER_2+1, b_n_per_block);
    csrw_ss(T_BOUND_BASE_READER_2+2, b_block_count);
    csrw_ss(T_BOUND_BASE_READER_2+3, M);
    csrw_ss(T_STRIDE_BASE_READER_2+0, 16u);
    csrw_ss(T_STRIDE_BASE_READER_2+1, b_n_stride);
    csrw_ss(T_STRIDE_BASE_READER_2+2, b_block_stride);
    csrw_ss(T_STRIDE_BASE_READER_2+3, 0u);
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_READER_2, MOE_DUAL_VC_CHAN_EN_B(2u));

    csrw_ss(BASE_PTR_WRITER_0_LOW, D0_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_0+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_0+0, MOE_DUAL_VC_MESH_COL_2 / 4u);
    csrw_ss(T_BOUND_BASE_WRITER_0+1, MOE_DUAL_VC_MESH_ROW_2);
    csrw_ss(T_BOUND_BASE_WRITER_0+2, N);
    csrw_ss(T_BOUND_BASE_WRITER_0+3, M);
    csrw_ss(T_STRIDE_BASE_WRITER_0+0, 8u);
    csrw_ss(T_STRIDE_BASE_WRITER_0+1, d_row_stride);
    csrw_ss(T_STRIDE_BASE_WRITER_0+2, MOE_DUAL_VC_MESH_COL_2 * 2u);
    csrw_ss(T_STRIDE_BASE_WRITER_0+3, d_m_stride);
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_0, MOE_DUAL_VC_CHAN_EN_D);

    csrw_ss(BASE_PTR_WRITER_1_LOW, D1_addr);
    csrw_ss(S_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_BOUND_BASE_WRITER_1+0, MOE_DUAL_VC_MESH_COL_2 / 4u);
    csrw_ss(T_BOUND_BASE_WRITER_1+1, MOE_DUAL_VC_MESH_ROW_2);
    csrw_ss(T_BOUND_BASE_WRITER_1+2, N);
    csrw_ss(T_BOUND_BASE_WRITER_1+3, M);
    csrw_ss(T_STRIDE_BASE_WRITER_1+0, 8u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+1, d_row_stride);
    csrw_ss(T_STRIDE_BASE_WRITER_1+2, MOE_DUAL_VC_MESH_COL_2 * 2u);
    csrw_ss(T_STRIDE_BASE_WRITER_1+3, d_m_stride);
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, 0u);
#endif
    csrw_ss(ENABLED_CHANNEL_WRITER_1, MOE_DUAL_VC_CHAN_EN_D);

    moe_set_dual_versacore_mode(1);
    moe_set_dual_versacore_csr(1, K, N * M, 0, 2u, 0);
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
}
