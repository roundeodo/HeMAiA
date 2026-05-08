#pragma once

#include <stdint.h>
#include <math.h>

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

static inline uint32_t moe_get_sram_block_index(
    uint32_t r,
    uint32_t c,
    const moe_operator_cfg_t *cfg)
{
    return ((r / cfg->mesh_row) * (cfg->router_n1 * cfg->mesh_row * cfg->mesh_col) +
            (c / cfg->mesh_col) * (cfg->mesh_row * cfg->mesh_col) +
            (r % cfg->mesh_row) * cfg->mesh_col +
            (c % cfg->mesh_col));
}

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

void build_scatter_metadata(
    uint16_t *global_top_k_indices,
    uint32_t actual_total_tokens,
    uint32_t *expert_token_counts,
    uint32_t *expert_memory_offsets,
    uint32_t *reverse_original_token_flat_idx,
    const moe_operator_cfg_t *cfg) // 【升级】存储 1D 绝对坐标
{
    for (uint32_t i = 0; i < cfg->expert_number_each_layer; i++)
    {
        expert_token_counts[i] = 0; // 假如不手动初始化，会直接是0吗？ 这里进行初始化，可以用DMA搬进去吗？
    }

    for (uint32_t t = 0; t < actual_total_tokens; t++)
    {
        for (uint32_t k = 0; k < cfg->individual_expert_number_k; k++)
        {
            uint32_t flat_idx = t * cfg->individual_expert_number_k + k;
            uint16_t expert_id = global_top_k_indices[flat_idx];
            expert_token_counts[expert_id]++;
        }
    }

    expert_memory_offsets[0] = 0;
    for (uint32_t i = 1; i < cfg->expert_number_each_layer; i++)
    {
        expert_memory_offsets[i] = expert_memory_offsets[i - 1] + expert_token_counts[i - 1];
    }

    uint32_t current_write_ptr[cfg->expert_number_each_layer];
    for (uint32_t i = 0; i < cfg->expert_number_each_layer; i++)
    {
        current_write_ptr[i] = expert_memory_offsets[i];
    }

    for (uint32_t t = 0; t < actual_total_tokens; t++)
    {
        for (uint32_t k = 0; k < cfg->individual_expert_number_k; k++)
        {
            uint32_t flat_idx = t * cfg->individual_expert_number_k + k;
            uint16_t expert_id = global_top_k_indices[flat_idx];
            uint32_t write_position = current_write_ptr[expert_id];

            // 【升级】将 flat_idx 刻入 Scatter 重组后的物理位置
            reverse_original_token_flat_idx[write_position] = flat_idx;
            current_write_ptr[expert_id]++;
        }
    }
}
// ==============================================
// 拆分 API 1: 仅提取 Top-K 索引和对应的原始分数
// ==============================================
void extract_top_k_indices_and_scores(
    int32_t *sram_raw_score_buffer,
    uint16_t *global_top_k_indices_ptr,
    int32_t *global_top_k_scores_ptr, // 【新增】保存原始分数，供后续延期算 Softmax
    uint32_t valid_tokens_in_block,
    const moe_operator_cfg_t *cfg)
{
    int32_t local_score[cfg->expert_number_each_layer];
    uint16_t local_top_k_indices[cfg->individual_expert_number_k];

    for (uint32_t r = 0; r < valid_tokens_in_block; r++)
    {
        for (uint32_t c = 0; c < cfg->expert_number_each_layer; c++)
        {
            uint32_t mem_idx = moe_get_sram_block_index(r, c, cfg);
            local_score[c] = sram_raw_score_buffer[mem_idx];
        }

        find_top_k_expert(local_score, local_top_k_indices, cfg);

        for (uint32_t i = 0; i < cfg->individual_expert_number_k; i++)
        {
            uint32_t out_idx = r * cfg->individual_expert_number_k + i;
            global_top_k_indices_ptr[out_idx] = local_top_k_indices[i];
            // 保存选中专家的原始分数
            global_top_k_scores_ptr[out_idx] = local_score[local_top_k_indices[i]];
        }
    }
}

// ==============================================
// 拆分 API 2: 延期的 Softmax 计算 (可以在硬件干活时在后台跑)
// ==============================================
void compute_delayed_softmax(
    int32_t *global_top_k_scores_ptr,
    uint32_t *global_calculated_probability_ptr,
    uint32_t actual_total_tokens,
    const moe_operator_cfg_t *cfg)
{
    const uint32_t K = cfg->individual_expert_number_k;
    const float scale = cfg->softmax_scale;
    int32_t *score_ptr = global_top_k_scores_ptr;
    uint32_t *prob_ptr = global_calculated_probability_ptr;

    for (uint32_t r = 0; r < actual_total_tokens; r++)
    {
        int32_t max_score = score_ptr[0];
        float exp_sum = 0.0f;
        float exp_value[K];

        for (uint32_t i = 0; i < K; i++)
        {
            exp_value[i] = expf((float)(score_ptr[i] - max_score));
            exp_sum += exp_value[i];
        }

        float inv_sum = 1.0f / exp_sum;
        for (uint32_t i = 0; i < K; i++)
        {
            prob_ptr[i] = (uint32_t)(exp_value[i] * inv_sum * scale);
        }

        score_ptr += K;
        prob_ptr += K;
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

void swishGLU_and_multiplication(
    int32_t *gate_project_data,
    int32_t *up_project_data,
    int8_t *activated_data,
    uint32_t valid_elements,
    float scale_in,
    float scale_out)
{
    for (uint32_t i = 0; i < valid_elements; i++)
    {
        float gated_data = (float)gate_project_data[i] * scale_in;
        float uped_data = (float)up_project_data[i] * scale_in;
        float sigmoid_gated_data = 1.0f / (1.0f + expf(-gated_data));
        float swish_gated_data = gated_data * sigmoid_gated_data;
        float result_float = swish_gated_data * uped_data;
        int32_t result_int = (int32_t)(result_float * scale_out);
        if (result_int > 127)
            result_int = 127;
        else if (result_int < -128)
            result_int = -128;

        activated_data[i] = result_int;
    }
}

void compute_swish_activation_tile(
    int32_t *gate_project_hw_data,
    float *swish_intermediate_buf,
    uint32_t valid_elements,
    float scale_in)
{
    const int32_t *src = gate_project_hw_data;
    float *dst = swish_intermediate_buf;
    for (uint32_t i = 0; i < valid_elements; i++)
    {
        float gated = (float)(*src++) * scale_in;
        float sigmoid = 1.0f / (1.0f + expf(-gated));
        *dst++ = gated * sigmoid;
    }
}

void compute_glu_multiplication_tile(
    float *swish_intermediate_buf,
    int32_t *up_projection_hw_data,
    int8_t *activated_out_data,
    uint32_t valid_elements,
    float scale_in,
    float scale_out)
{
    const float *swish_ptr = swish_intermediate_buf;
    const int32_t *up_ptr = up_projection_hw_data;
    int8_t *out_ptr = activated_out_data;
    for (uint32_t i = 0; i < valid_elements; i++)
    {
        float uped = (float)(*up_ptr++) * scale_in;
        int32_t r = (int32_t)((*swish_ptr++) * uped * scale_out);
        if (r > 127) r = 127;
        else if (r < -128) r = -128;
        *out_ptr++ = (int8_t)r;
    }
}

// scatter_and_pad_input_for_expert 已被移除。
// 该功能现在在 host_kernel_lib.h 的 __host_bingo_kernel_scatter_and_pad_input 中
// 通过 SoC iDMA 实现，直接从 L3 global_input_A 按 Token 粒度 DMA 到 L1 TCDM，
// 不再需要中间 L3 scatter pool，也不做零填充。

void experts_result_accumulate(
    int32_t *shared_expert_hw_output,
    int32_t *individual_experts_hw_output,
    uint32_t *reverse_original_token_flat_idx,
    uint32_t *global_calculated_probability,
    int32_t *final_layer_output,
    uint32_t actual_total_tokens,
    const moe_operator_cfg_t *cfg)
{
    const uint32_t D = cfg->input_dimension;
    const int64_t scale = (int64_t)cfg->softmax_scale;

    // Shared Expert: pointer-based sequential scan
    const int32_t *shared_ptr = shared_expert_hw_output;
    int32_t *final_ptr = final_layer_output;
    uint32_t total_elems = actual_total_tokens * D;
    for (uint32_t i = 0; i < total_elems; i++)
    {
        int64_t r = (int64_t)shared_ptr[i] * scale;
        final_ptr[i] = (int32_t)(r >> 16);
    }

    // Individual Expert: pre-compute row pointers to avoid inner-loop multiply
    const uint32_t K = cfg->individual_expert_number_k;
    uint32_t total_routed = actual_total_tokens * K;
    const int32_t *indiv_row = individual_experts_hw_output;

    for (uint32_t pos = 0; pos < total_routed; pos++)
    {
        uint32_t flat_idx = reverse_original_token_flat_idx[pos];
        uint32_t original_t = flat_idx / K;
        int64_t prob = (int64_t)global_calculated_probability[flat_idx];
        int32_t *out_row = final_ptr + original_t * D;

        for (uint32_t d = 0; d < D; d++)
        {
            int64_t r = (int64_t)indiv_row[d] * prob;
            out_row[d] += (int32_t)(r >> 16);
        }
        indiv_row += D;
    }
}