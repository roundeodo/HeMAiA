#pragma once

#include <stdint.h>
#include <math.h>
#include "heterogeneous_runtime.h"
#include "sys_dma.h"
#include "perf_tracing.h"

typedef struct
{
    uint32_t input_dimension;
    uint32_t expert_number_each_layer;
    uint32_t individual_expert_number_k;
    uint32_t mesh_row;
    uint32_t mesh_col;
    uint32_t router_m1;
    uint32_t router_n1;
    float softmax_scale;
} moe_operator_cfg_t;

// MoE_common_variable.h is NOT included here — its macros (input_dimension,
// softmax_scale, etc.) clash with struct field names and local variables.
// Those macros are only used by datagen.py and main_bingo.py (Python side).

// ============================================================================
// 这个头文件的角色
// ============================================================================
// 当前 workload 使用的主机侧 MoE 算子：
//   - TopK 提取：find_top_k_expert / extract_top_k_indices_and_scores
//   - 路由全局调度：moe_router_global_schedule（Phase 2，CVA6 调用）
//   - Tiled 累加：experts_result_accumulate_tiled（Phase 5，CVA6 调用）
//   - 累加 host kernel：__host_bingo_kernel_prefill_accumulate（DFG 末尾节点）
// ============================================================================

void find_top_k_expert(
    const int32_t *scores,
    uint16_t *top_k_indices,
    const moe_operator_cfg_t *cfg)
{
    uint16_t temp_indices[cfg->expert_number_each_layer];
    uint16_t temp_index = 0;
    for (uint32_t i = 0; i < cfg->expert_number_each_layer; i++)
    {
        temp_indices[i] = (uint16_t)i;
    }
    for (uint32_t i = 0; i < cfg->individual_expert_number_k; i++)
    {
        for (uint32_t j = cfg->expert_number_each_layer - 1; j > i; j--)
        {
            if (scores[temp_indices[j]] > scores[temp_indices[j - 1]])
            {
                temp_index = temp_indices[j];
                temp_indices[j] = temp_indices[j - 1];
                temp_indices[j - 1] = temp_index;
            }
        }
        top_k_indices[i] = temp_indices[i];
    }
}

// ==============================================
// 拆分 API 1: 仅提取 Top-K 索引和对应的原始分数
// ==============================================
void extract_top_k_indices_and_scores(
    int32_t *sram_raw_score_buffer,
    uint16_t *global_top_k_indices_ptr,
    int32_t *global_top_k_scores_ptr,
    uint32_t valid_tokens_in_block,
    const moe_operator_cfg_t *cfg)
{
    int32_t local_score[cfg->expert_number_each_layer];
    uint16_t local_top_k_indices[cfg->individual_expert_number_k];

    // 预计算 SRAM 索引参数: 用位移/掩码替代除法/取模 (mesh 维度为 2 的幂)
    uint32_t mr_shift = __builtin_ctz(cfg->mesh_row);
    uint32_t mc_shift = __builtin_ctz(cfg->mesh_col);
    uint32_t mr_mask = cfg->mesh_row - 1;
    uint32_t mc_mask = cfg->mesh_col - 1;
    uint32_t row_stride = cfg->router_n1 * cfg->mesh_row * cfg->mesh_col;
    uint32_t col_stride = cfg->mesh_row * cfg->mesh_col;
    uint32_t k = cfg->individual_expert_number_k;

    for (uint32_t r = 0; r < valid_tokens_in_block; r++)
    {
        for (uint32_t c = 0; c < cfg->expert_number_each_layer; c++)
        {
            uint32_t mem_idx = (r >> mr_shift) * row_stride +
                               (c >> mc_shift) * col_stride +
                               (r & mr_mask) * cfg->mesh_col +
                               (c & mc_mask);
            local_score[c] = sram_raw_score_buffer[mem_idx];
        }

        find_top_k_expert(local_score, local_top_k_indices, cfg);

        uint32_t out_base = r * k;
        for (uint32_t i = 0; i < k; i++)
        {
            global_top_k_indices_ptr[out_base + i] = local_top_k_indices[i];
            global_top_k_scores_ptr[out_base + i] = local_score[local_top_k_indices[i]];
        }
    }
}

void moe_router_global_schedule(
    uint32_t total_tokens,
    int32_t *hardware_output_buffer,
    uint16_t *global_indices_out,
    int32_t *global_scores_out,
    const moe_operator_cfg_t *cfg) // 【修正】这里输出的是 Scores，不是 Prob
{
    uint32_t tokens_per_hw_tile = cfg->router_m1 * cfg->mesh_row;
    uint32_t M2_loops = (total_tokens + tokens_per_hw_tile - 1) / tokens_per_hw_tile;
    uint32_t tokens_processed = 0;

    for (uint32_t m2 = 0; m2 < M2_loops; m2++)
    {
        uint32_t remaining = total_tokens > tokens_processed ? (total_tokens - tokens_processed) : 0;
        uint32_t valid_tokens_in_this_step = (remaining > tokens_per_hw_tile) ? tokens_per_hw_tile : remaining;
        if (valid_tokens_in_this_step == 0)
            break;

        uint32_t hw_buffer_offset = m2 * (cfg->router_m1 * cfg->router_n1 * cfg->mesh_row * cfg->mesh_col);
        int32_t *current_sram_hw_buffer = hardware_output_buffer + hw_buffer_offset;
        uint32_t write_offset = tokens_processed * cfg->individual_expert_number_k;

        // 【修正】调用拆分后的纯提取函数
        extract_top_k_indices_and_scores(
            current_sram_hw_buffer,
            &global_indices_out[write_offset],
            &global_scores_out[write_offset],
            valid_tokens_in_this_step,
            cfg);

        tokens_processed += valid_tokens_in_this_step;
    }
}

// ============================================================================
// Prefill Tiled Accumulate
// Handles: tiled GEMM output format [N2][M1][N1][meshRow][meshCol]
//          dynamic per-expert token counts
//          multiple shared experts
//          output in same tiled format (for golden comparison)
// ============================================================================
void experts_result_accumulate_tiled(
    int32_t *shared_hw_out, // all shared experts' down outputs, tiled
    int32_t *indiv_hw_out,  // all individual experts' down outputs, tiled
    uint32_t *expert_token_counts,
    uint32_t *expert_memory_offsets,
    uint32_t *reverse_original_token_flat_idx,
    uint32_t *global_calculated_probability,
    int32_t *final_layer_output,  // tiled output: (M2, N2, M1, N1, meshRow, meshCol)
    uint32_t actual_total_tokens, // M_total
    uint32_t N2_out,              // N2 tiles in down projection output
    uint32_t ptc,                 // per_tile_cols = N1 × meshRow × meshCol
    uint32_t max_tok,             // max_tokens_per_expert (= M1 × meshRow × M2)
    uint32_t E,                   // expert_number_each_layer
    uint32_t S,                   // shared_expert_number_k
    uint32_t k,                   // individual_expert_number_k
    uint32_t softmax_scale_int,
    uint32_t shift_step)
{
    uint32_t dim = N2_out * ptc;                  // total output dimension per token
    uint32_t tile_elems = max_tok * ptc;          // elements per tile per expert
    uint32_t expert_region = N2_out * tile_elems; // total elements per expert

    // ---- Phase 1: shared experts → initialize final_layer_output ----
    // Output tiled format: [N2][M_total][meshCol]
    // (M2=1 so outer M2 loop omitted)
    uint32_t out_tile_elems = actual_total_tokens * ptc;

    for (uint32_t s = 0; s < S; s++)
    {
        int32_t *se_base = &shared_hw_out[s * expert_region];
        for (uint32_t n = 0; n < N2_out; n++)
        {
            int32_t *src_tile = &se_base[n * tile_elems];
            int32_t *dst_tile = &final_layer_output[n * out_tile_elems];
            for (uint32_t t = 0; t < actual_total_tokens; t++)
            {
                int32_t *src_row = &src_tile[t * ptc];
                int32_t *dst_row = &dst_tile[t * ptc];
                for (uint32_t c = 0; c < ptc; c++)
                {
                    int64_t val = (int64_t)src_row[c] * (int64_t)softmax_scale_int;
                    if (s == 0)
                        dst_row[c] = (int32_t)(val >> shift_step);
                    else
                        dst_row[c] += (int32_t)(val >> shift_step);
                }
            }
        }
    }

    // ---- Phase 2: individual experts → gather-accumulate ----
    uint32_t k_shift = __builtin_ctz(k);

    for (uint32_t e = 0; e < E; e++)
    {
        uint32_t actual_tok_e = expert_token_counts[e];
        if (actual_tok_e == 0)
            continue;
        if (actual_tok_e > max_tok)
            actual_tok_e = max_tok;

        int32_t *ie_base = &indiv_hw_out[e * expert_region];
        uint32_t mem_offset_e = expert_memory_offsets[e];

        for (uint32_t slot = 0; slot < actual_tok_e; slot++)
        {
            uint32_t flat_idx = reverse_original_token_flat_idx[mem_offset_e + slot];
            uint32_t original_t = flat_idx >> k_shift;
            uint32_t prob = global_calculated_probability[flat_idx];

            for (uint32_t n = 0; n < N2_out; n++)
            {
                int32_t *src_row = &ie_base[n * tile_elems + slot * ptc];
                int32_t *dst_row = &final_layer_output[n * out_tile_elems + original_t * ptc];
                for (uint32_t c = 0; c < ptc; c++)
                {
                    int64_t val = (int64_t)src_row[c] * (int64_t)prob;
                    dst_row[c] += (int32_t)(val >> shift_step);
                }
            }
        }
    }
}

// ============================================================================
// Prefill accumulate host kernel（DFG 末尾节点，调用 experts_result_accumulate_tiled）
// ============================================================================
static inline uint64_t __host_bingo_kernel_prefill_accumulate(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_ACCUMULATE_START);
    int32_t *shared_hw = (int32_t *)(((uint64_t *)arg)[0]);
    int32_t *indiv_hw = (int32_t *)(((uint64_t *)arg)[1]);
    uint32_t *rev_idx = (uint32_t *)(((uint64_t *)arg)[2]);
    uint32_t *prob = (uint32_t *)(((uint64_t *)arg)[3]);
    uint32_t *exp_counts = (uint32_t *)(((uint64_t *)arg)[4]);
    uint32_t *exp_offsets = (uint32_t *)(((uint64_t *)arg)[5]);
    int32_t *final_out = (int32_t *)(((uint64_t *)arg)[6]);
    uint32_t total_tokens = (uint32_t)(((uint64_t *)arg)[7]);
    uint32_t N2_out = (uint32_t)(((uint64_t *)arg)[8]);
    uint32_t per_tile_cols = (uint32_t)(((uint64_t *)arg)[9]);
    uint32_t max_tok = (uint32_t)(((uint64_t *)arg)[10]);
    uint32_t E_val = (uint32_t)(((uint64_t *)arg)[11]);
    uint32_t S_val = (uint32_t)(((uint64_t *)arg)[12]);
    uint32_t k_val = (uint32_t)(((uint64_t *)arg)[13]);

    // Unpack float softmax_scale from raw bits
    union
    {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = (uint32_t)(((uint64_t *)arg)[14]);
    uint32_t scale_int = (uint32_t)cvt.f;

    uint32_t shift = (uint32_t)(((uint64_t *)arg)[15]);

    experts_result_accumulate_tiled(
        shared_hw, indiv_hw,
        exp_counts, exp_offsets, rev_idx, prob,
        final_out, total_tokens,
        N2_out, per_tile_cols, max_tok,
        E_val, S_val, k_val,
        scale_int, shift);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_ACCUMULATE_END);
    return 0;
}
