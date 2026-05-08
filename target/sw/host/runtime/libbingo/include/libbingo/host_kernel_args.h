// Fanchen Kong <fanchen.kong@kuleuven.be>
#pragma once
#include <stdint.h>
#define __HOST_BINGO_KERNEL_ARGS_DEFINE typedef struct __attribute__((packed, aligned(8)))

// NOTE: Every host kernel args struct has `scratchpad_ptr` as its LAST field.
// The kernel reads it to access its own bingo_kernel_scratchpad_t.

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_dummy_args {
    uint64_t dummy_input;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_dummy_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_exit_args {
    uint64_t exit_code;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_exit_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_entry_args {
    uint64_t start_cc_reg_addr;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_entry_args_t;


// Check-mode constants for __host_bingo_kernel_check_result
#define BINGO_CHECK_TYPE_BYTE_EXACT 0
#define BINGO_CHECK_TYPE_FP32_TOL   1
#define BINGO_CHECK_TYPE_FP16_TOL   2

// Precision selector for the runtime-typed Ara kernels (__host_bingo_kernel_<op>
// dispatchers in host_kernel_lib.h). Passed as a plain arg word; the typed
// __host_bingo_kernel_<op>_f32 entry points keep their original FP32-only
// behaviour and ignore precision.
#define BINGO_PREC_FP32  0
#define BINGO_PREC_FP16  1
#define BINGO_PREC_INT8  2
#define BINGO_PREC_INT16 3
#define BINGO_PREC_INT32 4

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_check_result_args {
    uint64_t golden_data_addr;
    uint64_t output_data_addr;
    uint64_t data_size;        // in Bytes (byte-exact) OR fp-array size in Bytes (fp modes)
    uint64_t name_addr;        // const char* label for printf (0 = no label)
    uint64_t check_type;       // 0 = byte-exact, 1 = fp32 absolute-tolerance, 2 = fp16 absolute-tolerance
    uint64_t tolerance_bits;   // FP32 bit pattern of absolute tolerance (used when check_type != 0)
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_check_result_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_idma_args {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t size;        // in Bytes
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_idma_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_xdma_1d_copy_args {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t size;        // in Bytes
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_xdma_1d_copy_args_t;
typedef __host_bingo_kernel_xdma_1d_copy_args_t __host_bingo_kernel_xdma_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_moe_router_schedule_args {
    uint64_t total_tokens;
    uint64_t hardware_output_buffer_addr;
    uint64_t global_indices_out_addr;
    uint64_t global_scores_out_addr;
    uint64_t expert_token_counts_out_addr;
    uint64_t expert_number_each_layer;
    uint64_t individual_expert_number_k;
    uint64_t mesh_row;
    uint64_t mesh_col;
    uint64_t router_m1;
    uint64_t router_n1;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_moe_router_schedule_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_compute_delayed_softmax_args {
    uint64_t global_top_k_scores_ptr_addr;
    uint64_t global_calculated_probability_ptr_addr;
    uint64_t actual_total_tokens;
    uint64_t individual_expert_number_k;
    uint64_t softmax_scale_raw;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_compute_delayed_softmax_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_build_scatter_metadata_args {
    uint64_t global_top_k_indices_addr;
    uint64_t actual_total_tokens;
    uint64_t expert_token_counts_addr;
    uint64_t expert_memory_offsets_addr;
    uint64_t reverse_original_token_flat_idx_addr;
    uint64_t expert_number_each_layer;
    uint64_t individual_expert_number_k;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_build_scatter_metadata_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_compute_swish_activation_args {
    uint64_t gate_project_hw_data_addr;
    uint64_t swish_intermediate_buf_addr;
    uint64_t valid_elements;
    uint64_t swish_glu_scale_in_raw;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_compute_swish_activation_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_compute_glu_multiplication_args {
    uint64_t swish_intermediate_buf_addr;
    uint64_t up_projection_hw_data_addr;
    uint64_t activated_out_data_addr;
    uint64_t valid_elements;
    uint64_t swish_glu_scale_in_raw;
    uint64_t swish_glu_scale_out_raw;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_compute_glu_multiplication_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_experts_result_accumulate_args {
    uint64_t shared_expert_hw_output_addr;
    uint64_t individual_experts_hw_output_addr;
    uint64_t reverse_original_token_flat_idx_addr;
    uint64_t global_calculated_probability_addr;
    uint64_t final_layer_output_addr;
    uint64_t actual_total_tokens;
    uint64_t input_dimension;
    uint64_t individual_expert_number_k;
    uint64_t softmax_scale_raw;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_experts_result_accumulate_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_compute_hw_silu_glu_args {
    uint64_t gate_silu_hw_data_addr;
    uint64_t up_projection_hw_data_addr;
    uint64_t activated_out_data_addr;
    uint64_t valid_elements;
    uint64_t swish_glu_scale_in_raw;
    uint64_t swish_glu_scale_out_raw;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_compute_hw_silu_glu_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_scatter_and_pad_input_args {
    uint64_t expert_id;
    uint64_t global_input_A_addr;
    uint64_t padded_scatter_pool_addr;
    uint64_t expert_token_counts_addr;
    uint64_t expert_memory_offsets_addr;
    uint64_t reverse_original_token_flat_idx_addr;
    uint64_t input_dimension;
    uint64_t max_padded_tokens;
    uint64_t individual_expert_number_k;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_scatter_and_pad_input_args_t;

// MoEPrepare ABI:
//   pure HW fast build: counts/CAM -> RTL scheduler -> direct L3 stage args.
//   SW/check builds: request_out_addr/schedule_out_addr remain available.
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_moe_prepare_request_args {
    uint64_t expert_token_counts_addr;
    uint64_t cam_state_addr;
#if !defined(MOE_ENABLE_HW_SCHEDULER) || defined(MOE_ENABLE_HW_SCHEDULER_CHECK)
    uint64_t request_out_addr;
    uint64_t schedule_out_addr;
#endif
    uint64_t expert_token_offsets_addr;
    uint64_t expert_token_ids_addr;
    uint64_t expert_token_kpos_addr;
    uint64_t n_experts;
    uint64_t topk_indices_l3;
    uint64_t M_total;
    uint64_t top_k;
    uint64_t expert_token_counts_valid;
    uint64_t runtime_state_addr;
    uint64_t c2_stage_base;
    uint64_t c3_stage_base;
    uint64_t dynamic_arg_slot_bytes;
    uint64_t dynamic_num_slots;
    uint64_t c2_l1_a;
    uint64_t c2_l1_d;
    uint64_t c2_l1_down_d;
    uint64_t c3_l1_a;
    uint64_t c3_l1_d;
    uint64_t c3_l1_down_d;
    uint64_t A_token_bytes;
    uint64_t indiv_D_tile_bytes;
    uint64_t indiv_down_D_tile_bytes;
    uint64_t indiv_N_per_block;
    uint64_t indiv_down_N_per_block;
    uint64_t s1_block_count;
    uint64_t s3_block_count;
    uint64_t max_tokens_per_expert;
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_moe_prepare_request_args_t;

// MoEExecute ABI:
//   pure HW fast build: sync runtime_state and flush active stage args to L1.
//   SW/check builds may still consume request_addr/schedule_addr.
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_moe_execute_args {
#if !defined(MOE_ENABLE_HW_SCHEDULER) || defined(MOE_ENABLE_HW_SCHEDULER_CHECK)
    uint64_t request_addr;
    uint64_t schedule_addr;
#endif
    uint64_t expert_token_offsets_addr;
    uint64_t expert_token_ids_addr;
    uint64_t expert_token_kpos_addr;
    uint64_t cam_state_addr;
    uint64_t input_A_l3_base;
    uint64_t topk_indices_l3;
    uint64_t indiv_gate_B_l3;
    uint64_t indiv_up_B_l3;
    uint64_t indiv_down_B_l3;
    uint64_t c2_l1_b_gate;
    uint64_t c2_l1_b_up;
    uint64_t c2_l1_b_down;
    uint64_t c2_l1_a;
    uint64_t c2_l1_d;
    uint64_t c2_l1_down_d;
    uint64_t c2_l1_d1_scratch;
    uint64_t c3_l1_b_gate;
    uint64_t c3_l1_b_up;
    uint64_t c3_l1_b_down;
    uint64_t c3_l1_a;
    uint64_t c3_l1_d;
    uint64_t c3_l1_down_d;
    uint64_t c3_l1_d1_scratch;
    uint64_t output_l3_addr;
    uint64_t runtime_state_addr;
    uint64_t c2_active_state_l1_addr;
    uint64_t c3_active_state_l1_addr;
    uint64_t A_token_bytes;
    uint64_t indiv_B_expert_stride;
    uint64_t indiv_down_B_expert_stride;
    uint64_t down_D_bytes_per_expert;
    uint64_t M_total;
    uint64_t top_k;
    uint64_t indiv_B_tile_bytes;
    uint64_t indiv_D_tile_bytes;
    uint64_t indiv_down_B_tile_bytes;
    uint64_t indiv_down_D_tile_bytes;
    uint64_t indiv_N2;
    uint64_t indiv_down_N2;
    uint64_t s1_block_count;
    uint64_t s3_block_count;
    uint64_t indiv_K1;
    uint64_t indiv_N_per_block;
    uint64_t indiv_down_K1;
    uint64_t indiv_down_N_per_block;
    uint64_t rescale_mult;
    uint64_t rescale_shift;
    uint64_t output_expert_stride_bytes;
    uint64_t max_tokens_per_expert;
    uint64_t c2_static_args_base;
    uint64_t c3_static_args_base;
    uint64_t c2_dynamic_args_base;
    uint64_t c3_dynamic_args_base;
    uint64_t dynamic_arg_slot_bytes;
    uint64_t dynamic_num_slots;
    uint64_t scratchpad_ptr;
    // L3 staging buffers: CVA6 writes args here (fast L3 stores),
    // then sys_dma_blk_memcpy flushes to cluster L1 in one burst.
    // Must be >= dynamic_num_slots * dynamic_arg_slot_bytes each.
    uint64_t c2_stage_base;
    uint64_t c3_stage_base;
} __host_bingo_kernel_moe_execute_args_t;

// NOTE: the per-kernel typed arg structs for quantize / dequantize / int32-add /
// softmax were folded into the unified Ara shapes below. Their kernels
// (__host_bingo_kernel_quantize_f32i8 / dequantize_i32f32 / add_i32, and the
// softmax dispatcher) read the shared ara_convert / ara_binary / ara_softmax
// structs — the kernels only touch the leading address/size slots, so the extra
// trailing `precision` field is a harmless no-op for the single-precision ops.

// ==========================================================================
// Multi-precision Ara kernel args (runtime-typed __host_bingo_kernel_<op>
// dispatchers in host_kernel_lib.h). The `precision` field is a BINGO_PREC_*
// word that the dispatcher reads to pick the fp32/fp16/int8/int16 path. Four
// shapes are shared across the ops (the field order matches the arg indices
// the dispatchers read; scratchpad_ptr stays last per the convention):
//   ara_binary : add/sub/mul/div/max/min, silu_mul        (a, b, out, n)
//   ara_unary  : relu/neg/abs/exp/sigmoid/sqrt/tanh/reciprocal/silu/gelu,
//                reduce_sum/reduce_max/reduce_mean         (in, out, n)
//   ara_softmax: softmax                                   (in, out, rows, len)
//   ara_rmsnorm: rmsnorm                          (in, weight, out, hidden, tokens)
//   ara_convert: quantize_f32i8 / quantize_f16i8 / dequantize_i32f32
//                                                          (in, out, scale, n)
// ==========================================================================
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_ara_binary_args {
    uint64_t input_a_addr;     // operand A (silu_mul: gate)
    uint64_t input_b_addr;     // operand B (silu_mul: up)
    uint64_t output_addr;
    uint64_t num_elements;
    uint64_t precision;        // BINGO_PREC_*
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_ara_binary_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_ara_unary_args {
    uint64_t input_addr;
    uint64_t output_addr;      // elementwise: array; reduce: scalar (float / int32)
    uint64_t num_elements;
    uint64_t precision;        // BINGO_PREC_*
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_ara_unary_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_ara_softmax_args {
    uint64_t input_addr;
    uint64_t output_addr;
    uint64_t num_rows;
    uint64_t row_length;
    uint64_t precision;        // BINGO_PREC_*
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_ara_softmax_args_t;

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_ara_rmsnorm_args {
    uint64_t input_addr;
    uint64_t weight_addr;
    uint64_t output_addr;
    uint64_t hidden_dim;
    uint64_t num_tokens;
    uint64_t precision;        // BINGO_PREC_*
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_ara_rmsnorm_args_t;

// Conversion ops with a scale pointer (quantize: writes scale; dequantize: reads
// scale). The kernels read only {input, output, scale, num_elements}; precision
// is a passthrough no-op (kept for a uniform shape). Consumed by quantize_f32i8,
// quantize_f16i8 (fp16 input), and dequantize_i32f32.
__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_ara_convert_args {
    uint64_t input_addr;
    uint64_t output_addr;
    uint64_t scale_addr;       // quantize: write computed scale; dequantize: read scale
    uint64_t num_elements;
    uint64_t precision;        // BINGO_PREC_* (no-op for the conversions)
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_ara_convert_args_t;

// DARTS Tier 1: Unified CERF Gating kernel
// Supports multiple activation modes via the `mode` field.
// Mode 0 (top_k):    Read logits from predecessor scratchpad, select top-k experts
// Mode 1 (threshold): Read confidence from predecessor scratchpad, activate if below threshold
// Mode 2 (static):   Use compile-time cerf_write_mask directly (no predecessor read)
//
// Two-level gating for >32 experts (CERF group sharing):
//   Level 1 (HW): CERF groups (max 32) — inactive groups skip at zero cost
//   Level 2 (SW): Per-expert activation array — experts in active groups check
//                 their slot in cond_activation_addr before computing.
//                 Written by this gating kernel, read by expert kernels.
#define BINGO_GATING_MODE_TOP_K      0
#define BINGO_GATING_MODE_THRESHOLD  1
#define BINGO_GATING_MODE_STATIC     2

__HOST_BINGO_KERNEL_ARGS_DEFINE __host_bingo_kernel_cerf_gating_args {
    uint64_t mode;                    // BINGO_GATING_MODE_*
    uint64_t pred_scratchpad_addr;    // predecessor's scratchpad (unused for static)
    uint64_t cerf_controlled_mask;    // 32-bit: groups this gating node owns
    uint64_t top_k_or_threshold;      // top_k: k; threshold: FP32 bits; static: cerf_write_mask
    uint64_t cerf_group_ids_addr;    // top_k: &cerf_group_ids[]; others: unused (0)
    uint64_t cond_activation_addr;  // Per-expert activation array (uint8_t[num_experts])
                                      // Gating kernel writes 1 for selected experts, 0 for others.
                                      // Expert kernels read their slot before computing.
                                      // 0 = no activation array (all experts in active groups run).
    uint64_t scratchpad_ptr;
} __host_bingo_kernel_cerf_gating_args_t;
