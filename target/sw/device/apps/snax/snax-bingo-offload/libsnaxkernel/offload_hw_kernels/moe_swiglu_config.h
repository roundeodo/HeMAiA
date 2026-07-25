// Shared SwiGLU VersaCore configuration.
#pragma once

__attribute__((always_inline)) static inline void
__moe_configure_swiglu(
    uint32_t A_addr,
    uint32_t Bg_addr,
    uint32_t Bu_addr,
    uint32_t D0_addr,
    uint32_t D1_addr,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t b_block_count,
    uint32_t b_block_stride,
    uint32_t array_shape,
    uint32_t rscl_mult,
    uint32_t rscl_shift)
{
    uint32_t meshRow = MOE_DUAL_VC_MESH_ROW_0 >> array_shape;
    uint32_t tileSize = MOE_DUAL_VC_TILE_SIZE_0;
    uint32_t meshCol = MOE_DUAL_VC_MESH_COL_0 << array_shape;

    // A: INT16, 6-dim temporal (shared for gate and up), 2-dim spatial [2, 8] (hw bounds).
    // Dynamic gather copies only each token payload into row starts separated
    // by the L15 stride. The streamer never accesses the 32-byte row gaps.
    // Ats[0] = tileSize*2 (K stride: advance one K-tile within token row).
    // Ats[2] = meshRow * a_row_stride (M stride: advance by physical rows).
    uint32_t a_row_stride = (uint32_t)K * (uint32_t)tileSize * 2u +
        BINGO_MOE_L15_ROW_GAP_BYTES;
    int32_t Asl[2]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)a_row_stride };
    int32_t Atb[6]       = { (int32_t)K, (int32_t)N, (int32_t)M, 1, 1, 1 };
    int32_t Ats[6]       = { (int32_t)(tileSize * 2), 0,
                              (int32_t)(meshRow * a_row_stride), 0, 0, 0 };
    int32_t chan_en_A[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_A(array_shape) };

    // B0 (gate): INT4 packed, 4-dim temporal, 2-dim spatial (hw req.)
    // L15 B canonical layout: [N_tiles_S0][K_tiles][16 bytes].
    // B_sl[1] = K*tileSize*2 (K-section stride between outer spatial j-groups).
    // B_ts[0] = 16 (fixed k-stride for all shapes; NOT B_stream_bytes).
    // B_ts[1] = K * B_stream_bytes (correct N-tile stride, same for all shapes).
    uint32_t n_b_chan      = (array_shape == 0u) ? 2u : (array_shape == 1u) ? 4u : 8u;
    int32_t B_stream_bytes = (int32_t)(n_b_chan * (MOE_DUAL_VC_BANK_WIDTH / 8));
    uint32_t b_k_section = (uint32_t)K * (uint32_t)tileSize * 2u;
    uint32_t b_n_per_block = N / b_block_count;
    int32_t B0sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B0tb[4]        = { (int32_t)K, (int32_t)b_n_per_block,
                               (int32_t)b_block_count, (int32_t)M };
    int32_t B0ts[4]        = { (int32_t)(tileSize * 2u),
                               (int32_t)(K * B_stream_bytes),
                               (int32_t)b_block_stride, 0 };
    int32_t chan_en_B0[1]  = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // B1 (up): INT4 packed, same layout as B0, 2-dim spatial (hw req.)
    int32_t B1sl[2]        = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8), (int32_t)b_k_section };
    int32_t B1tb[4]        = { (int32_t)K, (int32_t)b_n_per_block,
                               (int32_t)b_block_count, (int32_t)M };
    int32_t B1ts[4]        = { (int32_t)(tileSize * 2u),
                               (int32_t)(K * B_stream_bytes),
                               (int32_t)b_block_stride, 0 };
    int32_t chan_en_B1[1]  = { (int32_t)MOE_DUAL_VC_CHAN_EN_B(array_shape) };

    // D0: INT16, 4-dim temporal (4-lane postproc: 1ch, 4×int16/beat=8 bytes/beat)
    // 8 beats per N-tile (constant for all shapes: meshRow*meshCol/4=8)
    int32_t D0sl[1]       = { 8 };
    int32_t D0tb[4]       = { 8, (int32_t)N, (int32_t)M, 1 };
    int32_t D0ts[4]       = { 8, 64, (int32_t)(N * 64), 0 };
    int32_t chan_en_D0[1] = { (int32_t)MOE_DUAL_VC_CHAN_EN_D };

    // D1: NOT used in Mode 0 (SwiGLU). The hardware NEVER sends data to Writer 1
    // in Mode 0 (oa1_in_valid = 1'b0 in DualVersaCoreSwigluGen). We must zero
    // the temporal bounds and channel enable so Streamer Writer 1 completes
    // immediately (0 iterations) instead of waiting for data that never arrives.
    int32_t D1sl[1]       = { (int32_t)(MOE_DUAL_VC_BANK_WIDTH / 8) };
    int32_t D1tb[4]       = { 0, 0, 0, 0 };   // 0 iterations → Writer 1 done instantly
    int32_t D1ts[4]       = { 0, 0, 0, 0 };   // irrelevant (no iterations)
    int32_t chan_en_D1[1] = { 0 };             // no channels in Mode 0

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_START);
    moe_set_dual_versacore_streamer_csr(
        A_addr,  Asl,  Atb,  Ats,  0, chan_en_A,
        Bg_addr, B0sl, B0tb, B0ts, 0, chan_en_B0,
        Bu_addr, B1sl, B1tb, B1ts, 0, chan_en_B1,
        D0_addr, D0sl, D0tb, D0ts, 0, chan_en_D0,
        D1_addr, D1sl, D1tb, D1ts, 0, chan_en_D1);

    moe_set_dual_versacore_mode(0);   // Mode 0 = SwiGLU

    moe_set_dual_versacore_csr(
        1,       // take_in_new_c
        K,       // accum_bound
        N * M,   // output_times
        0,       // subtractions
        array_shape,
        0);      // data_type

    // Rescale for all 3 paths (gate, up, and element-wise multiply)
    moe_set_dual_versacore_rescale0(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale1(0, rscl_mult, 0, rscl_shift);
    moe_set_dual_versacore_rescale_mul(0, rscl_mult, 0, rscl_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_CFG_END);
}
