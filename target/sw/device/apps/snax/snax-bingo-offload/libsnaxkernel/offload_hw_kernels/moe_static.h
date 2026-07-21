// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// These wrappers expose the already validated MoE streamer helpers to static
// DFGs. They intentionally contain no scheduler state, slot checks, prefetch
// policy, or runtime lowering.

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_swiglu(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_moe_swiglu_args_t);
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_swiglu_args_t *cfg =
        (const __snax_bingo_kernel_moe_swiglu_args_t *)arg;
    if (cfg->array_shape > 2u || cfg->b_block_count == 0u ||
        cfg->N % cfg->b_block_count != 0u) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    __moe_configure_swiglu(
        cfg->input_A_addr,
        cfg->input_gate_B_addr,
        cfg->input_up_B_addr,
        cfg->output_D0_addr,
        cfg->output_D1_addr,
        cfg->M,
        cfg->K,
        cfg->N,
        cfg->b_block_count,
        cfg->b_block_stride,
        cfg->array_shape,
        cfg->rescale_mult,
        cfg->rescale_shift);
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    return BINGO_RET_SUCC;
}

static inline uint32_t __moe_active_cfg_test_args_valid(
    const __snax_bingo_kernel_moe_active_cfg_preload_test_args_t *cfg)
{
    return cfg->s1_array_shape <= 2u && cfg->s2_array_shape <= 2u &&
        cfg->s2_b_block_count != 0u &&
        cfg->s2_N % cfg->s2_b_block_count == 0u &&
        cfg->down_array_shape <= 2u;
}

// Match the production binding=2 S2 down prefetch: queue both tensor halves
// on xDMA's committed-task FIFO while the GEMM core runs S2 in parallel.
SNAX_LIB_DEFINE uint32_t
__snax_bingo_kernel_moe_active_cfg_prefetch_down_xdma_pair_test(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_dual_dma_args_t);
    if (!snrt_is_dm_core()) return BINGO_RET_FAIL;

    const uint32_t *a = (const uint32_t *)arg;
    uint64_t src0 = make_u64(a[0], a[1]);
    uint64_t dst0 = make_u64(a[2], a[3]);
    uint32_t bytes0 = a[4];
    uint64_t src1 = make_u64(a[5], a[6]);
    uint64_t dst1 = make_u64(a[7], a[8]);
    uint32_t bytes1 = a[9];
    if (bytes0 == 0u || bytes0 != bytes1) return BINGO_RET_FAIL;

    return __moe_dyn_copy_pair(2u, dst0, src0, dst1, src1, bytes0);
}

// Exact predecessor of the production S2 task: run the last shape-1 S1 block
// while staging the complete two-block shape-2 S2 configuration.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_active_cfg_s1_preload_s2_test(void *arg)
{
    BINGO_SW_GUARD_CHECK(
        arg, __snax_bingo_kernel_moe_active_cfg_preload_test_args_t);
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;

    const __snax_bingo_kernel_moe_active_cfg_preload_test_args_t *cfg =
        (const __snax_bingo_kernel_moe_active_cfg_preload_test_args_t *)arg;
    if (!__moe_active_cfg_test_args_valid(cfg)) return BINGO_RET_FAIL;

    uint32_t last_block_offset =
        (cfg->s2_b_block_count - 1u) * cfg->swiglu_b_block_stride;

    __moe_configure_swiglu(
        cfg->input_s1_A_addr,
        cfg->input_gate_B_addr + last_block_offset,
        cfg->input_up_B_addr + last_block_offset,
        cfg->s1_output_D0_addr,
        cfg->swiglu_output_D1_addr,
        cfg->s1_M,
        cfg->swiglu_K,
        cfg->s1_N,
        1u,
        cfg->swiglu_b_block_stride,
        cfg->s1_array_shape,
        cfg->rescale_mult,
        cfg->rescale_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();

    __moe_configure_swiglu(
        cfg->input_s2_A_addr,
        cfg->input_gate_B_addr,
        cfg->input_up_B_addr,
        cfg->s2_output_D0_addr,
        cfg->swiglu_output_D1_addr,
        cfg->s2_M,
        cfg->swiglu_K,
        cfg->s2_N,
        cfg->s2_b_block_count,
        cfg->swiglu_b_block_stride,
        cfg->s2_array_shape,
        cfg->rescale_mult,
        cfg->rescale_shift);

    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    return BINGO_RET_SUCC;
}

// Consume the S2 image staged by the previous task without reconfiguration,
// then reproduce production's S2-busy / S3-preload overlap.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_active_cfg_s2_preload_s3_test(void *arg)
{
    BINGO_SW_GUARD_CHECK(
        arg, __snax_bingo_kernel_moe_active_cfg_preload_test_args_t);
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;

    const __snax_bingo_kernel_moe_active_cfg_preload_test_args_t *cfg =
        (const __snax_bingo_kernel_moe_active_cfg_preload_test_args_t *)arg;
    if (!__moe_active_cfg_test_args_valid(cfg)) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();

    __moe_configure_down(
        cfg->down_input_A_addr,
        cfg->input_down_B0_addr,
        cfg->input_down_B1_addr,
        cfg->down_output_D0_addr,
        cfg->down_output_D1_addr,
        cfg->down_M,
        cfg->down_K,
        cfg->down_N,
        cfg->down_array_shape,
        cfg->down_K * 16u,
        cfg->down_K * (16u << cfg->down_array_shape),
        ((cfg->down_K * MOE_DUAL_VC_TILE_SIZE_0) /
         (MOE_DUAL_VC_MESH_COL_0 << cfg->down_array_shape)) * 64u,
        (MOE_DUAL_VC_MESH_ROW_0 >> cfg->down_array_shape) *
            cfg->down_output_row_stride,
        cfg->down_output_row_stride,
        cfg->rescale_mult,
        cfg->rescale_shift);

    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    return BINGO_RET_SUCC;
}

// Reproduce a cache-hit second slot: S1 is skipped, so S2 is configured and
// started directly with all token rows, then stages the full shape-2 Mode-1 S4.
SNAX_LIB_DEFINE uint32_t
__snax_bingo_kernel_moe_active_cfg_cache_hit_s2_preload_s4_test(void *arg)
{
    BINGO_SW_GUARD_CHECK(
        arg, __snax_bingo_kernel_moe_active_cfg_preload_test_args_t);
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;

    const __snax_bingo_kernel_moe_active_cfg_preload_test_args_t *cfg =
        (const __snax_bingo_kernel_moe_active_cfg_preload_test_args_t *)arg;
    if (!__moe_active_cfg_test_args_valid(cfg) ||
        cfg->down_N % cfg->s2_b_block_count != 0u) {
        return BINGO_RET_FAIL;
    }

    __moe_configure_swiglu(
        cfg->input_s2_A_addr,
        cfg->input_gate_B_addr,
        cfg->input_up_B_addr,
        cfg->s2_output_D0_addr,
        cfg->swiglu_output_D1_addr,
        cfg->s2_M,
        cfg->swiglu_K,
        cfg->s2_N,
        cfg->s2_b_block_count,
        cfg->swiglu_b_block_stride,
        cfg->s2_array_shape,
        cfg->rescale_mult,
        cfg->rescale_shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();

    uint32_t down_half_weight_bytes =
        cfg->input_down_B1_addr - cfg->input_down_B0_addr;
    __moe_configure_down_shape_c(
        cfg->down_input_A_addr,
        cfg->input_down_B0_addr,
        cfg->input_down_B1_addr,
        cfg->down_output_D0_addr,
        cfg->down_output_D1_addr,
        cfg->down_M,
        cfg->down_K,
        cfg->down_N,
        cfg->s2_b_block_count,
        down_half_weight_bytes / cfg->s2_b_block_count,
        cfg->down_K * 16u,
        cfg->down_N / cfg->s2_b_block_count,
        cfg->down_K * (16u << cfg->down_array_shape),
        ((cfg->down_K * MOE_DUAL_VC_TILE_SIZE_0) /
         (MOE_DUAL_VC_MESH_COL_0 << cfg->down_array_shape)) * 64u,
        (MOE_DUAL_VC_MESH_ROW_0 >> cfg->down_array_shape) *
            cfg->down_output_row_stride,
        cfg->down_output_row_stride,
        cfg->rescale_mult,
        cfg->rescale_shift);

    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    return BINGO_RET_SUCC;
}

// Consume the Mode-1 image staged by S2 and prove that the next START also
// completes. Keeping this as a separate task makes the failing stage visible
// directly in NODE_TIMING output.
SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_active_cfg_run_s3_test(void *arg)
{
    BINGO_SW_GUARD_CHECK(
        arg, __snax_bingo_kernel_moe_active_cfg_preload_test_args_t);
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    return BINGO_RET_SUCC;
}

SNAX_LIB_DEFINE uint32_t __snax_bingo_kernel_moe_down(void *arg)
{
    BINGO_SW_GUARD_CHECK(arg, __snax_bingo_kernel_moe_down_args_t);
    if (snrt_cluster_core_idx() != 0) return BINGO_RET_FAIL;

    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    const __snax_bingo_kernel_moe_down_args_t *cfg =
        (const __snax_bingo_kernel_moe_down_args_t *)arg;
    if (cfg->array_shape > 2u || cfg->b_block_count == 0u ||
        cfg->N % cfg->b_block_count != 0u ||
        (cfg->array_shape != 2u && cfg->b_block_count != 1u)) {
        BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
        return BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    if (cfg->array_shape == 2u) {
        __moe_configure_down_shape_c(
            cfg->input_A_addr,
            cfg->input_B0_addr,
            cfg->input_B1_addr,
            cfg->output_D0_addr,
            cfg->output_D1_addr,
            cfg->M,
            cfg->K,
            cfg->N,
            cfg->b_block_count,
            cfg->b_block_stride,
            cfg->K * 16u,
            cfg->N / cfg->b_block_count,
            cfg->K * 64u,
            ((cfg->K * MOE_DUAL_VC_TILE_SIZE_2) /
             MOE_DUAL_VC_MESH_COL_2) * 64u,
            MOE_DUAL_VC_MESH_ROW_2 * cfg->output_row_stride,
            cfg->output_row_stride,
            cfg->rescale_mult,
            cfg->rescale_shift);
    } else {
        __moe_configure_down(
            cfg->input_A_addr,
            cfg->input_B0_addr,
            cfg->input_B1_addr,
            cfg->output_D0_addr,
            cfg->output_D1_addr,
            cfg->M,
            cfg->K,
            cfg->N,
            cfg->array_shape,
            cfg->K * 16u,
            cfg->K * (16u << cfg->array_shape),
            ((cfg->K * MOE_DUAL_VC_TILE_SIZE_0) /
             (MOE_DUAL_VC_MESH_COL_0 << cfg->array_shape)) * 64u,
            (MOE_DUAL_VC_MESH_ROW_0 >> cfg->array_shape) *
                cfg->output_row_stride,
            cfg->output_row_stride,
            cfg->rescale_mult,
            cfg->rescale_shift);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_START);
    moe_start_dual_vc_and_streamer();
    moe_wait_dual_vc_and_streamer();
    BINGO_TRACE_MARKER(BINGO_TRACE_GEMM_FULL_RUN_END);
    return BINGO_RET_SUCC;
}
