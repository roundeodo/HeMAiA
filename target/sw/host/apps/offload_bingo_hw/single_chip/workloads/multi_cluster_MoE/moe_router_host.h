#pragma once

#include <stdint.h>

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

static inline uint32_t moe_ctz32_no_libgcc(uint32_t x)
{
    if (x == 0u)
        return 32u;

    uint32_t n = 0;
    while ((x & 1u) == 0u) {
        n++;
        x >>= 1;
    }
    return n;
}

// Software Router Top-K extraction for the multi-cluster workload.
// The pure-HW build uses the direct, table-producing path in host_kernel_lib.h;
// these generic routines remain only for the pure-software comparison build.

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
    int16_t *sram_raw_score_buffer,  // INT16 output from dual_vc_gemm_full (8 bytes/beat = 4×INT16)
    uint16_t *global_top_k_indices_ptr,
    int32_t *global_top_k_scores_ptr,
    uint32_t *expert_token_counts_out,
    uint32_t valid_tokens_in_block,
    const moe_operator_cfg_t *cfg)
{
    int32_t local_score[cfg->expert_number_each_layer];
    uint16_t local_top_k_indices[cfg->individual_expert_number_k];

    // 预计算 SRAM 索引参数: 用位移/掩码替代除法/取模 (mesh 维度为 2 的幂)
    uint32_t mr_shift = moe_ctz32_no_libgcc(cfg->mesh_row);
    uint32_t mc_shift = moe_ctz32_no_libgcc(cfg->mesh_col);
    uint32_t mr_mask = cfg->mesh_row - 1;
    uint32_t mc_mask = cfg->mesh_col - 1;
    uint32_t row_stride = cfg->router_n1 * cfg->mesh_row * cfg->mesh_col;
    uint32_t col_stride = cfg->mesh_row * cfg->mesh_col;
    uint32_t tile_stride = cfg->router_m1 * row_stride;
    uint32_t experts_per_n2_tile = cfg->router_n1 * cfg->mesh_col;
    uint32_t k = cfg->individual_expert_number_k;

    for (uint32_t r = 0; r < valid_tokens_in_block; r++)
    {
        for (uint32_t c = 0; c < cfg->expert_number_each_layer; c++)
        {
            uint32_t n2 = c / experts_per_n2_tile;
            uint32_t n1 = (c >> mc_shift) - n2 * cfg->router_n1;
            uint32_t mem_idx = n2 * tile_stride +
                               (r >> mr_shift) * row_stride +
                               n1 * col_stride +
                               (r & mr_mask) * cfg->mesh_col +
                               (c & mc_mask);
            local_score[c] = (int32_t)sram_raw_score_buffer[mem_idx];  // sign-extend INT16→INT32
        }

        find_top_k_expert(local_score, local_top_k_indices, cfg);

        uint32_t out_base = r * k;
        for (uint32_t i = 0; i < k; i++)
        {
            uint16_t expert_id = local_top_k_indices[i];
            global_top_k_indices_ptr[out_base + i] = expert_id;
            global_top_k_scores_ptr[out_base + i] = local_score[expert_id];
            if (expert_token_counts_out != 0) {
                expert_token_counts_out[expert_id]++;
            }
        }
    }
}

static inline int moe_router_top2_fast_supported(const moe_operator_cfg_t *cfg)
{
    if (cfg == 0) {
        return 0;
    }
    return cfg->individual_expert_number_k == 2u &&
           cfg->expert_number_each_layer >= 2u &&
           cfg->router_n1 == 1u &&
           cfg->mesh_col != 0u &&
           (cfg->expert_number_each_layer % cfg->mesh_col) == 0u &&
           cfg->mesh_row != 0u &&
           ((cfg->mesh_row & (cfg->mesh_row - 1u)) == 0u);
}

static inline void moe_router_update_top2(
    int32_t score,
    uint16_t eid,
    int32_t *best0_score,
    int32_t *best1_score,
    uint16_t *best0_eid,
    uint16_t *best1_eid)
{
    if (score > *best0_score) {
        *best1_score = *best0_score;
        *best1_eid = *best0_eid;
        *best0_score = score;
        *best0_eid = eid;
    } else if (score > *best1_score) {
        *best1_score = score;
        *best1_eid = eid;
    }
}

static inline void extract_top2_indices_and_scores_fast(
    const int16_t *sram_raw_score_buffer,
    uint16_t *global_top_k_indices_ptr,
    int32_t *global_top_k_scores_ptr,
    uint32_t *expert_token_counts_out,
    uint32_t valid_tokens_in_block,
    const moe_operator_cfg_t *cfg)
{
    uint32_t mr_shift = moe_ctz32_no_libgcc(cfg->mesh_row);
    uint32_t mr_mask = cfg->mesh_row - 1u;
    uint32_t row_stride = cfg->mesh_row * cfg->mesh_col;
    uint32_t tile_stride = cfg->router_m1 * row_stride;
    uint32_t n2_count = cfg->expert_number_each_layer / cfg->mesh_col;

    for (uint32_t r = 0; r < valid_tokens_in_block; r++) {
        uint32_t row_base = (r >> mr_shift) * row_stride +
                            (r & mr_mask) * cfg->mesh_col;
        int32_t best0_score = INT32_MIN;
        int32_t best1_score = INT32_MIN;
        uint16_t best0_eid = 0u;
        uint16_t best1_eid = 1u;

        for (uint32_t n2 = 0; n2 < n2_count; n2++) {
            const int16_t *score_tile =
                sram_raw_score_buffer + n2 * tile_stride + row_base;
            uint32_t eid_base = n2 * cfg->mesh_col;

            for (uint32_t col = 0; col < cfg->mesh_col; col++) {
                moe_router_update_top2((int32_t)score_tile[col],
                                       (uint16_t)(eid_base + col),
                                       &best0_score, &best1_score,
                                       &best0_eid, &best1_eid);
            }
        }

        uint32_t out_base = r * 2u;
        global_top_k_indices_ptr[out_base + 0u] = best0_eid;
        global_top_k_indices_ptr[out_base + 1u] = best1_eid;
        global_top_k_scores_ptr[out_base + 0u] = best0_score;
        global_top_k_scores_ptr[out_base + 1u] = best1_score;
        if (expert_token_counts_out != 0) {
            expert_token_counts_out[best0_eid]++;
            expert_token_counts_out[best1_eid]++;
        }
    }
}

void moe_router_global_schedule(
    uint32_t total_tokens,
    int32_t *hardware_output_buffer,
    uint16_t *global_indices_out,
    int32_t *global_scores_out,
    uint32_t *expert_token_counts_out,
    const moe_operator_cfg_t *cfg) // 【修正】这里输出的是 Scores，不是 Prob
{
    uint32_t tokens_per_hw_tile = cfg->router_m1 * cfg->mesh_row;
    uint32_t M2_loops = (total_tokens + tokens_per_hw_tile - 1) / tokens_per_hw_tile;
    uint32_t tokens_processed = 0;

    if (expert_token_counts_out != 0) {
        for (uint32_t e = 0; e < cfg->expert_number_each_layer; e++) {
            expert_token_counts_out[e] = 0u;
        }
    }

    for (uint32_t m2 = 0; m2 < M2_loops; m2++)
    {
        uint32_t remaining = total_tokens > tokens_processed ? (total_tokens - tokens_processed) : 0;
        uint32_t valid_tokens_in_this_step = (remaining > tokens_per_hw_tile) ? tokens_per_hw_tile : remaining;
        if (valid_tokens_in_this_step == 0)
            break;

        uint32_t hw_buffer_offset = m2 * (cfg->router_m1 * cfg->router_n1 * cfg->mesh_row * cfg->mesh_col);
        // Cast to int16_t*: hw_buffer_offset elements × 2 bytes = correct byte skip for INT16 M2-groups
        int16_t *current_sram_hw_buffer = (int16_t *)hardware_output_buffer + hw_buffer_offset;
        uint32_t write_offset = tokens_processed * cfg->individual_expert_number_k;

        if (moe_router_top2_fast_supported(cfg)) {
            extract_top2_indices_and_scores_fast(
                current_sram_hw_buffer,
                &global_indices_out[write_offset],
                &global_scores_out[write_offset],
                expert_token_counts_out,
                valid_tokens_in_this_step,
                cfg);
        } else {
            // Generic path for non-top2 or non-router_n1=1 layouts.
            extract_top_k_indices_and_scores(
                current_sram_hw_buffer,
                &global_indices_out[write_offset],
                &global_scores_out[write_offset],
                expert_token_counts_out,
                valid_tokens_in_this_step,
                cfg);
        }

        tokens_processed += valid_tokens_in_this_step;
    }
}
