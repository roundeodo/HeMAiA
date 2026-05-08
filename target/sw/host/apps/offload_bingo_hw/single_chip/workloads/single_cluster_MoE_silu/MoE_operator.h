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

// ==============================================
// 拆分 API 2: 延期的 Softmax 计算 (可以在硬件干活时在后台跑)
// 优化: scores 已降序排列, exp(max-max)=1.0 直接跳过首项 expf 调用
// ==============================================
void compute_delayed_softmax(
    int32_t *global_top_k_scores_ptr,
    uint32_t *global_calculated_probability_ptr,
    uint32_t actual_total_tokens,
    const moe_operator_cfg_t *cfg)
{
    uint32_t k = cfg->individual_expert_number_k;
    float scale = cfg->softmax_scale;

    for (uint32_t r = 0; r < actual_total_tokens; r++)
    {
        uint32_t base = r * k;
        float max_score_f = (float)global_top_k_scores_ptr[base];

        // 首项 exp(0) = 1.0, 跳过 fast_expf 调用
        float exp_sum = 1.0f;
        for (uint32_t i = 1; i < k; i++)
        {
            float delta = (float)global_top_k_scores_ptr[base + i] - max_score_f;
            float ev = expf(delta);
            // 暂存 exp 值到输出缓冲区 (类型双关: float→uint32_t)
            union
            {
                float f;
                uint32_t u;
            } cvt = {.f = ev};
            global_calculated_probability_ptr[base + i] = cvt.u;
            exp_sum += ev;
        }

        float inv_sum = scale / exp_sum;
        global_calculated_probability_ptr[base] = (uint32_t)(1.0f * inv_sum);
        for (uint32_t i = 1; i < k; i++)
        {
            union
            {
                uint32_t u;
                float f;
            } cvt = {.u = global_calculated_probability_ptr[base + i]};
            global_calculated_probability_ptr[base + i] = (uint32_t)(cvt.f * inv_sum);
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

        activated_data[i] = (int8_t)result_int;
    }
}

void compute_swish_activation_tile(
    int32_t *gate_project_hw_data,
    float *swish_intermediate_buf,
    uint32_t valid_elements,
    float scale_in)
{
    for (uint32_t i = 0; i < valid_elements; i++)
    {
        float gated_data = (float)gate_project_hw_data[i] * scale_in;
        float sigmoid_gated_data = 1.0f / (1.0f + expf(-gated_data));
        swish_intermediate_buf[i] = gated_data * sigmoid_gated_data;
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
    for (uint32_t i = 0; i < valid_elements; i++)
    {
        float uped_data = (float)up_projection_hw_data[i] * scale_in;
        float result_float = swish_intermediate_buf[i] * uped_data;
        int32_t result_int = (int32_t)(result_float * scale_out);

        if (result_int > 127)
            result_int = 127;
        else if (result_int < -128)
            result_int = -128;

        activated_out_data[i] = (int8_t)result_int;
    }
}

// ============================================================================
// 阶段 4 预处理: DMA 加速 Scatter (不再零填充)
// 使用 SoC iDMA 逐 Token 搬运, 取代 CVA6 逐字节软件拷贝
// 下游 GEMM 仍按静态 M 维度执行, 未覆盖的行为残留垃圾值,
// 但 accumulate 仅通过 reverse_index 读取有效行, 不影响正确性
// ============================================================================
void scatter_and_pad_input_for_expert(
    int8_t *global_input_A,
    int8_t *scatter_pool_dst,
    uint32_t expert_id,
    uint32_t *expert_token_counts,
    uint32_t *expert_memory_offsets,
    uint32_t *reverse_original_token_flat_idx,
    uint32_t input_dim,
    uint32_t max_padded_tokens,
    uint32_t expert_top_k)
{
    uint32_t actual_token_count = expert_token_counts[expert_id];
    uint32_t physical_offset = expert_memory_offsets[expert_id];

    if (actual_token_count > max_padded_tokens)
        actual_token_count = max_padded_tokens;

    if (actual_token_count == 0)
        return;

    uint8_t chip_id = get_current_chip_id();
    // 用位移替代除法 (expert_top_k 为 2 的幂)
    uint32_t k_shift = __builtin_ctz(expert_top_k);

    // 批量发射 DMA: 全部 token 行的 DMA 请求连续提交到 iDMA 队列,
    // 只在最后一笔传输完成后等待, 避免逐条阻塞
    uint64_t last_tf_id = 0;
    for (uint32_t i = 0; i < actual_token_count; i++)
    {
        uint32_t flat_idx = reverse_original_token_flat_idx[physical_offset + i];
        uint32_t original_token_id = flat_idx >> k_shift;

        uint64_t src = (uint64_t)(uintptr_t)&global_input_A[original_token_id * input_dim];
        uint64_t dst = (uint64_t)(uintptr_t)&scatter_pool_dst[i * input_dim];

        last_tf_id = sys_dma_memcpy(chip_id, dst, src, (uint64_t)input_dim);
    }
    // 等待最后一笔 DMA 完成 (iDMA 保序, last 完成则全部完成)
    while (*(sys_dma_done_ptr(chip_id)) != last_tf_id)
    {
        asm volatile("nop");
    }
}

void experts_result_accumulate(
    int32_t *shared_expert_hw_output,
    int32_t *individual_experts_hw_output,
    uint32_t *reverse_original_token_flat_idx,
    uint32_t *global_calculated_probability,
    int32_t *final_layer_output,
    uint32_t actual_total_tokens,
    const moe_operator_cfg_t *cfg)
{
    uint32_t dim = cfg->input_dimension;
    uint32_t k = cfg->individual_expert_number_k;
    uint32_t scale_int = (uint32_t)cfg->softmax_scale;
    uint32_t k_shift = __builtin_ctz(k);
    uint32_t total_routed = actual_total_tokens * k;

    // ========================================================================
    // Decode 快速路径 (actual_total_tokens == 1):
    // 将 shared expert + 所有 individual expert 的贡献合并到一次内循环中计算。
    // 优势:
    //   1. 消除 final_layer_output 的 read-modify-write (3 遍 → 1 遍 write-only)
    //   2. 截断延后: int64 累加后一次性 >>16, 精度更高 (误差 ≤ ±1 LSB)
    //   3. 减少内存带宽: 5120 次访问 → 1024 次 (对 CVA6 的 L1 cache 压力降低)
    // 溢出安全: 每项 |int32 × uint16| ≤ 2^47, k+1 项之和 < 2^49 ≪ 2^63
    // ========================================================================
    if (actual_total_tokens == 1)
    {
        int32_t *shared_src = shared_expert_hw_output;
        int32_t *dst = final_layer_output;

        if (k == 2)
        {
            uint32_t flat0 = reverse_original_token_flat_idx[0];
            uint32_t flat1 = reverse_original_token_flat_idx[1];
            uint32_t p0 = global_calculated_probability[flat0];
            uint32_t p1 = global_calculated_probability[flat1];
            int32_t *s0 = individual_experts_hw_output;
            int32_t *s1 = individual_experts_hw_output + dim;

            for (uint32_t d = 0; d < dim; d++)
            {
                int64_t acc = (int64_t)shared_src[d] * (int64_t)scale_int + (int64_t)s0[d] * (int64_t)p0 + (int64_t)s1[d] * (int64_t)p1;
                dst[d] = (int32_t)(acc >> 16);
            }
        }
        else
        {
            uint32_t probs[8];
            int32_t *indiv_srcs[8];
            uint32_t actual_k = (k <= 8) ? k : 8;
            for (uint32_t p = 0; p < actual_k; p++)
            {
                uint32_t flat_idx = reverse_original_token_flat_idx[p];
                probs[p] = global_calculated_probability[flat_idx];
                indiv_srcs[p] = &individual_experts_hw_output[p * dim];
            }
            for (uint32_t d = 0; d < dim; d++)
            {
                int64_t acc = (int64_t)shared_src[d] * (int64_t)scale_int;
                for (uint32_t p = 0; p < actual_k; p++)
                {
                    acc += (int64_t)indiv_srcs[p][d] * (int64_t)probs[p];
                }
                dst[d] = (int32_t)(acc >> 16);
            }
        }
        return;
    }

    // ========================================================================
    // 通用多 token 路径 (prefill): 保持原始 3 遍结构
    // ========================================================================
    for (uint32_t t = 0; t < actual_total_tokens; t++)
    {
        int32_t *src = &shared_expert_hw_output[t * dim];
        int32_t *dst = &final_layer_output[t * dim];
        for (uint32_t d = 0; d < dim; d++)
        {
            int64_t result = (int64_t)src[d] * (int64_t)scale_int;
            dst[d] = (int32_t)(result >> 16);
        }
    }

    for (uint32_t p = 0; p < total_routed; p++)
    {
        uint32_t flat_idx = reverse_original_token_flat_idx[p];
        uint32_t original_t = flat_idx >> k_shift;
        uint32_t prob = global_calculated_probability[flat_idx];

        int32_t *hw_src = &individual_experts_hw_output[p * dim];
        int32_t *out_dst = &final_layer_output[original_t * dim];
        for (uint32_t d = 0; d < dim; d++)
        {
            int64_t result = (int64_t)hw_src[d] * (int64_t)prob;
            out_dst[d] += (int32_t)(result >> 16);
        }
    }
}