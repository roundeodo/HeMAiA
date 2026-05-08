// Auto-generated test workload for MoE host API verification
// This file defines kernel_execution() which tests each host-side MoE
// operator function against golden data produced by Python.
#pragma once
#include "libbingo/bingo_api.h"
#include "host.h"
#include "host_api_test_data.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline float u32_to_f32(uint32_t u) {
    union { uint32_t u; float f; } conv;
    conv.u = u;
    return conv.f;
}

static inline int f32_close(float a, float b, float tol) {
    float d = a - b;
    if (d < 0.0f) d = -d;
    return d <= tol;
}

// ---------------------------------------------------------------------------
// kernel_execution — called from offload_bingo_hw.c after system init
// ---------------------------------------------------------------------------
int kernel_execution() {
    check_kernel_tab_ready();
    printf_safe("Chip(%x, %x): [Host] Preparing host_api_test Workload\r\n",
                get_current_chip_loc_x(), get_current_chip_loc_y());

    uint32_t total_errors = 0;

    // ================================================================
    // TEST 1 — moe_router_global_schedule (find_top_k + extract)
    // ================================================================
    {
        moe_operator_cfg_t cfg = {
            .input_dimension    = TEST_INPUT_DIM,
            .expert_number_each_layer = TEST_EXPERT_NUM,
            .individual_expert_number_k = TEST_TOP_K,
            .mesh_row   = TEST_MESH_ROW,
            .mesh_col   = TEST_MESH_COL,
            .router_m1  = TEST_ROUTER_M1,
            .router_n1  = TEST_ROUTER_N1,
            .softmax_scale = 0.0f,
        };

        uint16_t out_idx[TEST_TOTAL_TOKENS * TEST_TOP_K];
        int32_t  out_scores[TEST_TOTAL_TOKENS * TEST_TOP_K];

        moe_router_global_schedule(TEST_TOTAL_TOKENS,
                                   test_sram_scores,
                                   out_idx, out_scores, &cfg);

        uint32_t err = 0;
        for (uint32_t i = 0; i < (uint32_t)(TEST_TOTAL_TOKENS * TEST_TOP_K); i++) {
            if (out_idx[i] != golden_topk_idx[i]) {
                printf_safe("  [T1] idx[%u]: got %u exp %u\r\n",
                            i, (uint32_t)out_idx[i],
                            (uint32_t)golden_topk_idx[i]);
                err++;
            }
            if (out_scores[i] != golden_topk_scores[i]) {
                printf_safe("  [T1] score[%u]: got %d exp %d\r\n",
                            i, out_scores[i], golden_topk_scores[i]);
                err++;
            }
        }
        if (err == 0) printf_safe("[TEST 1] moe_router_global_schedule: PASSED\r\n");
        else          printf_safe("[TEST 1] moe_router_global_schedule: FAILED (%u)\r\n", err);
        total_errors += err;
    }

    // ================================================================
    // TEST 2 — compute_delayed_softmax
    // ================================================================
    {
        moe_operator_cfg_t cfg = {
            .input_dimension = 0,
            .expert_number_each_layer = 0,
            .individual_expert_number_k = TEST_TOP_K,
            .mesh_row = 0, .mesh_col = 0,
            .router_m1 = 0, .router_n1 = 0,
            .softmax_scale = u32_to_f32(TEST_SOFTMAX_HEX),
        };

        int32_t  scores_in[TEST_TOTAL_TOKENS * TEST_TOP_K];
        uint32_t out_prob[TEST_TOTAL_TOKENS * TEST_TOP_K];
        for (uint32_t i = 0; i < (uint32_t)(TEST_TOTAL_TOKENS * TEST_TOP_K); i++)
            scores_in[i] = golden_topk_scores[i];

        compute_delayed_softmax(scores_in, out_prob, TEST_TOTAL_TOKENS, &cfg);

        uint32_t err = 0;
        for (uint32_t i = 0; i < (uint32_t)(TEST_TOTAL_TOKENS * TEST_TOP_K); i++) {
            int32_t diff = (int32_t)out_prob[i] - (int32_t)golden_softmax[i];
            if (diff < 0) diff = -diff;
            if (diff > 2) {
                printf_safe("  [T2] prob[%u]: got %u exp %u\r\n",
                            i, out_prob[i], golden_softmax[i]);
                err++;
            }
        }
        if (err == 0) printf_safe("[TEST 2] compute_delayed_softmax: PASSED\r\n");
        else          printf_safe("[TEST 2] compute_delayed_softmax: FAILED (%u)\r\n", err);
        total_errors += err;
    }

    // ================================================================
    // TEST 3 — build_scatter_metadata
    // ================================================================
    {
        moe_operator_cfg_t cfg = {
            .input_dimension = 0,
            .expert_number_each_layer = TEST_EXPERT_NUM,
            .individual_expert_number_k = TEST_TOP_K,
            .mesh_row = 0, .mesh_col = 0,
            .router_m1 = 0, .router_n1 = 0,
            .softmax_scale = 0.0f,
        };

        uint16_t idx_in[TEST_TOTAL_TOKENS * TEST_TOP_K];
        for (uint32_t i = 0; i < (uint32_t)(TEST_TOTAL_TOKENS * TEST_TOP_K); i++)
            idx_in[i] = golden_topk_idx[i];

        uint32_t out_counts[TEST_EXPERT_NUM];
        uint32_t out_offsets[TEST_EXPERT_NUM];
        uint32_t out_rev[TEST_TOTAL_TOKENS * TEST_TOP_K];

        build_scatter_metadata(idx_in, TEST_TOTAL_TOKENS,
                               out_counts, out_offsets, out_rev, &cfg);

        uint32_t err = 0;
        for (uint32_t i = 0; i < (uint32_t)TEST_EXPERT_NUM; i++) {
            if (out_counts[i] != golden_token_counts[i]) {
                printf_safe("  [T3] counts[%u]: got %u exp %u\r\n",
                            i, out_counts[i], golden_token_counts[i]);
                err++;
            }
            if (out_offsets[i] != golden_mem_offsets[i]) {
                printf_safe("  [T3] offsets[%u]: got %u exp %u\r\n",
                            i, out_offsets[i], golden_mem_offsets[i]);
                err++;
            }
        }
        for (uint32_t i = 0; i < (uint32_t)(TEST_TOTAL_TOKENS * TEST_TOP_K); i++) {
            if (out_rev[i] != golden_rev_flat[i]) {
                printf_safe("  [T3] rev[%u]: got %u exp %u\r\n",
                            i, out_rev[i], golden_rev_flat[i]);
                err++;
            }
        }
        if (err == 0) printf_safe("[TEST 3] build_scatter_metadata: PASSED\r\n");
        else          printf_safe("[TEST 3] build_scatter_metadata: FAILED (%u)\r\n", err);
        total_errors += err;
    }

    // ================================================================
    // TEST 4 — compute_swish_activation_tile
    // ================================================================
    {
        float scale_in = u32_to_f32(TEST_SCALE_IN_HEX);

        int32_t gate_in[TEST_NUM_SWISH_ELEMS];
        for (uint32_t i = 0; i < (uint32_t)TEST_NUM_SWISH_ELEMS; i++)
            gate_in[i] = test_gate_data[i];

        float out_swish[TEST_NUM_SWISH_ELEMS];
        compute_swish_activation_tile(gate_in, out_swish,
                                      TEST_NUM_SWISH_ELEMS, scale_in);

        uint32_t err = 0;
        for (uint32_t i = 0; i < (uint32_t)TEST_NUM_SWISH_ELEMS; i++) {
            float golden = u32_to_f32(golden_swish_hex[i]);
            if (!f32_close(out_swish[i], golden, 1e-4f)) {
                printf_safe("  [T4] swish[%u]: mismatch\r\n", i);
                err++;
            }
        }
        if (err == 0) printf_safe("[TEST 4] compute_swish_activation_tile: PASSED\r\n");
        else          printf_safe("[TEST 4] compute_swish_activation_tile: FAILED (%u)\r\n", err);
        total_errors += err;
    }

    // ================================================================
    // TEST 5 — compute_glu_multiplication_tile
    // ================================================================
    {
        float scale_in  = u32_to_f32(TEST_SCALE_IN_HEX);
        float scale_out = u32_to_f32(TEST_SCALE_OUT_HEX);

        // Use Python golden swish as input (test independence)
        float swish_in[TEST_NUM_SWISH_ELEMS];
        for (uint32_t i = 0; i < (uint32_t)TEST_NUM_SWISH_ELEMS; i++)
            swish_in[i] = u32_to_f32(golden_swish_hex[i]);

        int32_t up_in[TEST_NUM_SWISH_ELEMS];
        for (uint32_t i = 0; i < (uint32_t)TEST_NUM_SWISH_ELEMS; i++)
            up_in[i] = test_up_data[i];

        int8_t out_glu[TEST_NUM_SWISH_ELEMS];
        compute_glu_multiplication_tile(swish_in, up_in, out_glu,
                                         TEST_NUM_SWISH_ELEMS,
                                         scale_in, scale_out);

        uint32_t err = 0;
        for (uint32_t i = 0; i < (uint32_t)TEST_NUM_SWISH_ELEMS; i++) {
            int32_t diff = (int32_t)out_glu[i] - (int32_t)golden_glu[i];
            if (diff < 0) diff = -diff;
            if (diff > 1) {
                printf_safe("  [T5] glu[%u]: got %d exp %d\r\n",
                            i, (int32_t)out_glu[i], (int32_t)golden_glu[i]);
                err++;
            }
        }
        if (err == 0) printf_safe("[TEST 5] compute_glu_multiplication_tile: PASSED\r\n");
        else          printf_safe("[TEST 5] compute_glu_multiplication_tile: FAILED (%u)\r\n", err);
        total_errors += err;
    }

    // ================================================================
    // TEST 6 — scatter_and_pad_input_for_expert
    // ================================================================
    {
        int8_t input_A[TEST_TOTAL_TOKENS * TEST_INPUT_DIM];
        for (uint32_t i = 0; i < (uint32_t)(TEST_TOTAL_TOKENS * TEST_INPUT_DIM); i++)
            input_A[i] = test_global_input_A[i];

        uint32_t counts[TEST_EXPERT_NUM];
        uint32_t offsets[TEST_EXPERT_NUM];
        uint32_t rev[TEST_TOTAL_TOKENS * TEST_TOP_K];
        for (uint32_t i = 0; i < (uint32_t)TEST_EXPERT_NUM; i++) {
            counts[i]  = golden_token_counts[i];
            offsets[i] = golden_mem_offsets[i];
        }
        for (uint32_t i = 0; i < (uint32_t)(TEST_TOTAL_TOKENS * TEST_TOP_K); i++)
            rev[i] = golden_rev_flat[i];

        int8_t out_scatter[TEST_MAX_PAD_TOKENS * TEST_INPUT_DIM];
        scatter_and_pad_input_for_expert(
            input_A, out_scatter, TEST_EXPERT_ID,
            counts, offsets, rev,
            TEST_INPUT_DIM, TEST_MAX_PAD_TOKENS, TEST_TOP_K);

        uint32_t err = 0;
        for (uint32_t i = 0; i < (uint32_t)(TEST_MAX_PAD_TOKENS * TEST_INPUT_DIM); i++) {
            if (out_scatter[i] != golden_scatter[i]) {
                printf_safe("  [T6] scatter[%u]: got %d exp %d\r\n",
                            i, (int32_t)out_scatter[i], (int32_t)golden_scatter[i]);
                err++;
            }
        }
        if (err == 0) printf_safe("[TEST 6] scatter_and_pad_input: PASSED\r\n");
        else          printf_safe("[TEST 6] scatter_and_pad_input: FAILED (%u)\r\n", err);
        total_errors += err;
    }

    // ================================================================
    // Summary
    // ================================================================
    if (total_errors == 0)
        printf_safe("=== ALL HOST API TESTS PASSED ===\r\n");
    else
        printf_safe("=== HOST API TESTS FAILED: %u total errors ===\r\n", total_errors);

    return 0;
}
