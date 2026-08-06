#pragma once

#include <stdint.h>
#include <string.h>
#include "uart.h"
#include "heterogeneous_runtime.h"
#include "perf_tracing.h"
#include "libbingo/bingo_utils.h"
#include "libbingo/host_kernel_args.h"  // BINGO_CHECK_TYPE_*, BINGO_GATING_MODE_*
#include "libbingo/device_kernel_args.h"
#include "libbingo/bingo_api.h"         // BINGO_PRINTF, bingo_cerf_update()
#ifdef MOE_ENABLE_MULTI_CLUSTER_MOE
#include "moe_scheduler.h"
#ifdef MOE_ENABLE_HW_SCHEDULER
#include "moe_scheduler_hw_mmio.h"
#endif
#endif

/* ── Host-side scheduler/router debug prints are guarded by
 *    BINGO_DEBUG_LEVEL >= 1 so release timing is not dominated by UART. ── */
#ifndef BINGO_DEBUG_LEVEL
#define BINGO_DEBUG_LEVEL 0
#endif
#define MOE_SCHED_DEBUG_PRINT (BINGO_DEBUG_LEVEL >= 1)

#ifndef MOE_MCYCLE_DETAIL
#define MOE_MCYCLE_DETAIL 0
#endif

#define MOE_HOST_TIMING_ROUTER_SCHED 0u
#define MOE_HOST_TIMING_PREPARE      1u
#define MOE_HOST_TIMING_EXECUTE      2u
#define MOE_HOST_TIMING_HW_SCHED     3u
#define MOE_HOST_TIMING_HW_SORT      4u
#define MOE_HOST_TIMING_HW_INIT      5u
#define MOE_HOST_TIMING_HW_WAIT      6u
#define MOE_HOST_TIMING_HW_STATUS    7u
#define MOE_HOST_TIMING_HW_DRAIN     8u
#define MOE_HOST_TIMING_HW_CONTROL   9u
#define MOE_HOST_TIMING_HW_LOWER     10u
#define MOE_HOST_TIMING_LOWER_S4PF   11u
#define MOE_HOST_TIMING_LOWER_SLOT   12u
#define MOE_HOST_TIMING_LOWER_ARG    13u
#define MOE_HOST_TIMING_LOWER_DMA    14u
#define MOE_HOST_TIMING_LOWER_FINAL  15u
#define MOE_HOST_TIMING_LOWER_PRE    16u
#define MOE_HOST_TIMING_SW_SCHED     17u
#define MOE_HOST_TIMING_COUNT        18u

typedef struct {
    uint64_t start;
    uint64_t end;
    uint64_t total;
    uint32_t count;
} moe_host_timing_entry_t;

static volatile moe_host_timing_entry_t __moe_host_timing[MOE_HOST_TIMING_COUNT];

static inline void __moe_host_timing_start(uint32_t id)
{
    __moe_host_timing[id].start = bingo_mcycle();
}

static inline void __moe_host_timing_end(uint32_t id)
{
    uint64_t end = bingo_mcycle();
    uint64_t start = __moe_host_timing[id].start;
    __moe_host_timing[id].end = end;
    __moe_host_timing[id].total += end - start;
    __moe_host_timing[id].count++;
}

#define MOE_HOST_TIMING_PRINT(_label, _id)                                      \
    printf_safe("[MOE_PHASE_MCYCLE] " _label " count=%u last=%lu total=%lu\r\n", \
                (unsigned)__moe_host_timing[(_id)].count,                       \
                (unsigned long)(__moe_host_timing[(_id)].end -                  \
                                __moe_host_timing[(_id)].start),                \
                (unsigned long)__moe_host_timing[(_id)].total)

static inline void __host_bingo_moe_print_phase_timing(void)
{
    MOE_HOST_TIMING_PRINT("RouterSched", MOE_HOST_TIMING_ROUTER_SCHED);
    MOE_HOST_TIMING_PRINT("MoEPrepare", MOE_HOST_TIMING_PREPARE);
    MOE_HOST_TIMING_PRINT("MoEExecute", MOE_HOST_TIMING_EXECUTE);
    MOE_HOST_TIMING_PRINT("SW_SCHED", MOE_HOST_TIMING_SW_SCHED);
    MOE_HOST_TIMING_PRINT("HW_SCHED", MOE_HOST_TIMING_HW_SCHED);
#if MOE_MCYCLE_DETAIL
    MOE_HOST_TIMING_PRINT("HW_SORT", MOE_HOST_TIMING_HW_SORT);
    MOE_HOST_TIMING_PRINT("HW_INIT", MOE_HOST_TIMING_HW_INIT);
    MOE_HOST_TIMING_PRINT("HW_WAIT", MOE_HOST_TIMING_HW_WAIT);
    MOE_HOST_TIMING_PRINT("HW_STATUS", MOE_HOST_TIMING_HW_STATUS);
    MOE_HOST_TIMING_PRINT("HW_DRAIN", MOE_HOST_TIMING_HW_DRAIN);
    MOE_HOST_TIMING_PRINT("HW_CONTROL", MOE_HOST_TIMING_HW_CONTROL);
    MOE_HOST_TIMING_PRINT("HW_LOWER", MOE_HOST_TIMING_HW_LOWER);
    MOE_HOST_TIMING_PRINT("LOWER_S4PF", MOE_HOST_TIMING_LOWER_S4PF);
    MOE_HOST_TIMING_PRINT("LOWER_SLOT", MOE_HOST_TIMING_LOWER_SLOT);
    MOE_HOST_TIMING_PRINT("LOWER_ARG", MOE_HOST_TIMING_LOWER_ARG);
    MOE_HOST_TIMING_PRINT("LOWER_DMA", MOE_HOST_TIMING_LOWER_DMA);
    MOE_HOST_TIMING_PRINT("LOWER_FINAL", MOE_HOST_TIMING_LOWER_FINAL);
    MOE_HOST_TIMING_PRINT("LOWER_PRE", MOE_HOST_TIMING_LOWER_PRE);
#endif
}

#ifdef MOE_ENABLE_MULTI_CLUSTER_MOE
#define MOE_PACK_U32_PAIR(lo, hi) \
    ((uint64_t)(uint32_t)(lo) | ((uint64_t)(uint32_t)(hi) << 32u))

#ifndef MOE_ENABLE_HW_SCHEDULER
static inline void __moe_dbg_print_schedule(const moe_schedule_t *s)
{
    if (!MOE_SCHED_DEBUG_PRINT) {
        return;
    }
    printf_safe("[SCHED] n_tasks=%u n_dma_ops=%u\r\n",
                (unsigned)s->n_tasks, (unsigned)s->n_dma_ops);
    for (uint32_t _ti = 0; _ti < s->n_tasks; _ti++) {
        const moe_task_t *_t = &s->tasks[_ti];
        printf_safe("[SCHED] T%u C%u e%u n%u sh1=%u sh3=%u dm1=%u dm3=%u sk1=%u sk2=%u sk3=%u sk4=%u\r\n",
                    _ti, (unsigned)_t->cluster + 2u, (unsigned)_t->expert_id,
                    (unsigned)_t->ntokens,
                    (unsigned)_t->shape_s1, (unsigned)_t->shape_s3,
                    (unsigned)_t->dma_s1,   (unsigned)_t->dma_s3,
                    (unsigned)_t->skip_s1,  (unsigned)_t->skip_s2,
                    (unsigned)_t->skip_s3,  (unsigned)_t->skip_s4);
    }
    for (uint32_t _oi = 0; _oi < s->n_dma_ops; _oi++) {
        const moe_dma_op_t *_op = &s->dma_ops[_oi];
        printf_safe("[SCHED] D%u t%u k%u dm%u e%d\r\n",
                    _oi, (unsigned)_op->task_idx, (unsigned)_op->kind,
                    (unsigned)_op->dma, (int)_op->expert_id);
    }
}
#endif

#endif

#define EXIT_CODE_SUCC 1
#define EXIT_CODE_FAIL 2
// Host Bingo Kernel Implementations
// Normally the functions ret with 0
// Only the exit kernel returns the exit code defined by EXIT_CODE_SUCC, for now it is 1
static inline uint64_t __host_bingo_kernel_dummy(void *arg){
    // Arg[0]: dummy_input, Arg[1]: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t dummy_input = ((uint64_t *)arg)[0];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[1];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("Chip(%x, %x): [Host] Kernel Dummy: %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), dummy_input);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = 0;
    sp->num_return_values = 0;
    return 0;
}

static inline uint64_t __host_bingo_kernel_exit(void *arg){
    // Arg[0]: exit_code, Arg[1]: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t exit_code = ((uint64_t *)arg)[0];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[1];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    printf_safe("Chip(%x, %x): [Host] Kernel Exit called with exit code %d\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), exit_code);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = EXIT_CODE_SUCC;
    sp->num_return_values = 0;
    return EXIT_CODE_SUCC;
}

static inline uint64_t __host_bingo_kernel_entry(void *arg){
    // Arg[0]: start_cc_reg_addr, Arg[1]: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[1];
    uint64_t start_cc;
    asm volatile("csrr %0, mcycle" : "=r"(start_cc));
    printf_safe("Chip(%x, %x): [Host] Start at %d CC\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), start_cc);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    sp->return_value = (uint32_t)start_cc;
    sp->num_return_values = 0;
    return 0;
}
// Union-based type punning for fp32 ↔ uint32 (no strict-aliasing UB, portable, zero-cost).
typedef union { float f; uint32_t u; } __bingo_f32_u32_t;

static inline float __host_bingo_kernel_unpack_f32(uint64_t raw) {
    __bingo_f32_u32_t bits = {.u = (uint32_t)raw};
    return bits.f;
}

// Convert IEEE 754 binary16 (half) to binary32 (single) in software.
// Handles zero, subnormal, normal, inf, NaN. No libm / no _Float16 dependency.
static inline float __bingo_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)(h & 0x3FF);
    uint32_t f_bits;
    if (exp == 0) {
        if (mant == 0) {
            // +/- zero
            f_bits = sign << 31;
        } else {
            // Subnormal: normalize by shifting mantissa left until bit 10 is 1
            int shift = 0;
            while ((mant & 0x400) == 0) { mant <<= 1; shift++; }
            mant &= 0x3FF;  // clear the implicit bit after normalization
            uint32_t e = (uint32_t)(127 - 15 - shift);
            f_bits = (sign << 31) | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        f_bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        // Normal
        uint32_t e = exp - 15 + 127;
        f_bits = (sign << 31) | (e << 23) | (mant << 13);
    }
    __bingo_f32_u32_t u;
    u.u = f_bits;
    return u.f;
}

static inline uint64_t __host_bingo_kernel_check_result(void *arg){
    // Arg0-5: golden, output, size, name, check_type, tolerance_bits; Arg6: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint8_t* golden_data_addr = (uint8_t*)(((uint64_t *)arg)[0]);
    uint8_t* output_data_addr = (uint8_t*)(((uint64_t *)arg)[1]);
    uint64_t data_size        = ((uint64_t *)arg)[2];
    const char* name          = (const char*)(((uint64_t *)arg)[3]);
    uint64_t check_type       = ((uint64_t *)arg)[4];
    uint32_t tolerance_bits   = (uint32_t)((uint64_t *)arg)[5];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[6];
    if (!name) name = "?";
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);

    // Reinterpret tolerance uint32 bits as fp32 via union.
    __bingo_f32_u32_t tol_u;
    tol_u.u = tolerance_bits;
    const float tolerance = tol_u.f;

    uint32_t err = 0;

    if (check_type == BINGO_CHECK_TYPE_BYTE_EXACT) {
        uint64_t first_mismatch = data_size;
        if ((((uintptr_t)output_data_addr | (uintptr_t)golden_data_addr) &
             (sizeof(uint64_t) - 1u)) == 0u) {
            typedef uint64_t bingo_check_word_t __attribute__((may_alias));
            const bingo_check_word_t *output_words =
                (const bingo_check_word_t *)(const void *)output_data_addr;
            const bingo_check_word_t *golden_words =
                (const bingo_check_word_t *)(const void *)golden_data_addr;
            uint64_t word_count = data_size / sizeof(uint64_t);
            for (uint64_t i = 0; i < word_count; i++) {
                if (output_words[i] != golden_words[i]) {
                    first_mismatch = i * sizeof(uint64_t);
                    break;
                }
            }
            if (first_mismatch == data_size) {
                for (uint64_t i = word_count * sizeof(uint64_t);
                     i < data_size; i++) {
                    if (output_data_addr[i] != golden_data_addr[i]) {
                        first_mismatch = i;
                        break;
                    }
                }
            }
        } else {
            for (uint64_t i = 0; i < data_size; i++) {
                if (output_data_addr[i] != golden_data_addr[i]) {
                    first_mismatch = i;
                    break;
                }
            }
        }

        if (first_mismatch == data_size) {
            BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
            sp->return_value = 0;
            sp->num_return_values = 0;
            return 0;
        }

        enum {
            CHECK_DIAG_MAX_SAMPLES = 8,
            CHECK_DIAG_CHUNK_BYTES = 4096,
            CHECK_DIAG_MAX_CHUNKS = 8,
        };
        uint64_t sample_idx[CHECK_DIAG_MAX_SAMPLES];
        uint8_t sample_output[CHECK_DIAG_MAX_SAMPLES];
        uint8_t sample_golden[CHECK_DIAG_MAX_SAMPLES];
        uint32_t sample_count = 0;
        uint32_t output_nonzero = 0;
        uint32_t golden_nonzero = 0;
        uint32_t output_hash = 2166136261u;
        uint32_t golden_hash = 2166136261u;

        for (uint64_t i = 0; i < data_size; i++) {
            uint8_t output_byte = output_data_addr[i];
            uint8_t golden_byte = golden_data_addr[i];
            output_nonzero += output_byte != 0;
            golden_nonzero += golden_byte != 0;
            output_hash = (output_hash ^ output_byte) * 16777619u;
            golden_hash = (golden_hash ^ golden_byte) * 16777619u;
            if (output_byte != golden_byte) {
                if (sample_count < CHECK_DIAG_MAX_SAMPLES) {
                    sample_idx[sample_count] = i;
                    sample_output[sample_count] = output_byte;
                    sample_golden[sample_count] = golden_byte;
                    sample_count++;
                }
                err++;
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
        sp->return_value = err;
        sp->num_return_values = 0;
        printf_safe("[CHECK_DIAG] %s bytes=%lu err=%u out_nz=%u golden_nz=%u out_hash=0x%08x golden_hash=0x%08x\r\n",
                    name, data_size, err, output_nonzero, golden_nonzero,
                    output_hash, golden_hash);
        for (uint32_t i = 0; i < sample_count; i++) {
            printf_safe("[CHECK_SAMPLE] %s idx=%lu out=0x%02x golden=0x%02x\r\n",
                        name, sample_idx[i], (unsigned)sample_output[i],
                        (unsigned)sample_golden[i]);
        }

        uint32_t chunk_reports = 0;
        uint32_t chunks_with_errors = 0;
        for (uint64_t begin = 0; begin < data_size;
             begin += CHECK_DIAG_CHUNK_BYTES) {
            uint64_t end = begin + CHECK_DIAG_CHUNK_BYTES;
            if (end > data_size) end = data_size;
            uint32_t chunk_err = 0;
            uint32_t chunk_output_nonzero = 0;
            uint32_t chunk_golden_nonzero = 0;
            for (uint64_t i = begin; i < end; i++) {
                uint8_t output_byte = output_data_addr[i];
                uint8_t golden_byte = golden_data_addr[i];
                chunk_output_nonzero += output_byte != 0;
                chunk_golden_nonzero += golden_byte != 0;
                chunk_err += output_byte != golden_byte;
            }
            if (chunk_err == 0) continue;
            chunks_with_errors++;
            if (chunk_reports < CHECK_DIAG_MAX_CHUNKS) {
                printf_safe("[CHECK_CHUNK] %s off=%lu size=%lu err=%u out_nz=%u golden_nz=%u\r\n",
                            name, begin, end - begin, chunk_err,
                            chunk_output_nonzero, chunk_golden_nonzero);
                chunk_reports++;
            }
        }
        if (chunks_with_errors > chunk_reports) {
            printf_safe("[CHECK_CHUNK] %s shown=%u total_bad=%u\r\n",
                        name, chunk_reports, chunks_with_errors);
        }
        printf_safe("[Host] Check [%s]: FAIL (%d / %d bytes)\r\n", name,
                    err, data_size);
        return EXIT_CODE_FAIL;
    } else if (check_type == BINGO_CHECK_TYPE_FP32_TOL) {
        // FP32 absolute-tolerance mode
        const float* out_f    = (const float*)output_data_addr;
        const float* golden_f = (const float*)golden_data_addr;
        uint64_t num_elements = data_size / 4;
        for (uint64_t i = 0; i < num_elements; i++) {
            float o = out_f[i];
            float g = golden_f[i];
            float diff = o - g;
            if (diff < 0.0f) diff = -diff;
            if (diff > tolerance) {
                err++;
                // Hex-bits printing to avoid reliance on libc %f support
                __bingo_f32_u32_t uo, ug, ud;
                uo.f = o; ug.f = g; ud.f = diff;
                printf_safe("[%s] idx=%d out=0x%08x golden=0x%08x diff=0x%08x tol=0x%08x\n",
                       name, i, uo.u, ug.u, ud.u, tolerance_bits);
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
        sp->return_value = err;
        sp->num_return_values = 0;
        if (err == 0) {
            return 0;
        }
        printf_safe("[Host] Check [%s]: FAIL (%d / %d fp32 elems, tol_bits=0x%08x)\r\n",
               name, err, num_elements, tolerance_bits);
        return EXIT_CODE_FAIL;
    } else if (check_type == BINGO_CHECK_TYPE_FP16_TOL) {
        // FP16 absolute-tolerance mode — promote each half to fp32, compare against fp32 tolerance
        const uint16_t* out_h    = (const uint16_t*)output_data_addr;
        const uint16_t* golden_h = (const uint16_t*)golden_data_addr;
        uint64_t num_elements = data_size / 2;
        for (uint64_t i = 0; i < num_elements; i++) {
            uint16_t oh = out_h[i];
            uint16_t gh = golden_h[i];
            float o = __bingo_fp16_to_fp32(oh);
            float g = __bingo_fp16_to_fp32(gh);
            float diff = o - g;
            if (diff < 0.0f) diff = -diff;
            if (diff > tolerance) {
                err++;
                __bingo_f32_u32_t uo, ug, ud;
                uo.f = o; ug.f = g; ud.f = diff;
                printf_safe("[%s] idx=%d out_h=0x%04x golden_h=0x%04x out_f=0x%08x golden_f=0x%08x diff=0x%08x tol=0x%08x\n",
                       name, i, oh, gh, uo.u, ug.u, ud.u, tolerance_bits);
            }
        }
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
        sp->return_value = err;
        sp->num_return_values = 0;
        if (err == 0) {
            return 0;
        }
        printf_safe("[Host] Check [%s]: FAIL (%d / %d fp16 elems, tol_bits=0x%08x)\r\n",
               name, err, num_elements, tolerance_bits);
        return EXIT_CODE_FAIL;
    } else {
        // Unknown check_type — fail loudly
        BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
        printf_safe("[Host] Check [%s]: FAIL (unknown check_type=%d)\r\n", name, (int)check_type);
        sp->return_value = 1;
        sp->num_return_values = 0;
        return EXIT_CODE_FAIL;
    }
}

static inline uint64_t __host_bingo_kernel_idma(void *arg){
    // Arg0-2: src, dst, size; Arg3: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t src_addr = ((uint64_t *)arg)[0];
    uint64_t dst_addr = ((uint64_t *)arg)[1];
    uint64_t size = ((uint64_t *)arg)[2];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[3];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_CFG_START);
    uint64_t tf_id = sys_dma_memcpy(get_current_chip_id(), dst_addr, src_addr, size);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_CFG_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_RUN_START);
    while (*(sys_dma_done_ptr(get_current_chip_id())) != tf_id) {
        asm volatile("nop");
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_IDMA_RUN_END);
    return 0;
}

#if !defined(MOE_ENABLE_MULTI_CLUSTER_MOE) && !defined(MOE_OPERATOR_CUSTOM)
#if defined(__has_include)
#if __has_include("../apps/offload_bingo_hw/single_chip/workloads/single_cluster_MoE/MoE_operator.h")
#include "../apps/offload_bingo_hw/single_chip/workloads/single_cluster_MoE/MoE_operator.h"
#define BINGO_HAS_LEGACY_MOE_OPERATOR 1
#endif
#endif
#endif

#ifdef BINGO_HAS_LEGACY_MOE_OPERATOR
static inline uint64_t __host_bingo_kernel_compute_delayed_softmax(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t *global_top_k_scores_ptr = (int32_t *)(((uint64_t *)arg)[0]);
    uint32_t *global_calculated_probability_ptr = (uint32_t *)(((uint64_t *)arg)[1]);
    uint32_t actual_total_tokens = (uint32_t)(((uint64_t *)arg)[2]);
    moe_operator_cfg_t cfg = {
        .input_dimension = 0,
        .expert_number_each_layer = 0,
        .individual_expert_number_k = (uint32_t)(((uint64_t *)arg)[3]),
        .mesh_row = 0,
        .mesh_col = 0,
        .router_m1 = 0,
        .router_n1 = 0,
        .softmax_scale = __host_bingo_kernel_unpack_f32(((uint64_t *)arg)[4]),
    };
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_SOFTMAX_START);
    compute_delayed_softmax(global_top_k_scores_ptr, global_calculated_probability_ptr, actual_total_tokens, &cfg);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_SOFTMAX_END);
    return 0;
}

static inline uint64_t __host_bingo_kernel_build_scatter_metadata(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint16_t *global_top_k_indices = (uint16_t *)(((uint64_t *)arg)[0]);
    uint32_t actual_total_tokens = (uint32_t)(((uint64_t *)arg)[1]);
    uint32_t *expert_token_counts = (uint32_t *)(((uint64_t *)arg)[2]);
    uint32_t *expert_memory_offsets = (uint32_t *)(((uint64_t *)arg)[3]);
    uint32_t *reverse_original_token_flat_idx = (uint32_t *)(((uint64_t *)arg)[4]);
    moe_operator_cfg_t cfg = {
        .input_dimension = 0,
        .expert_number_each_layer = (uint32_t)(((uint64_t *)arg)[5]),
        .individual_expert_number_k = (uint32_t)(((uint64_t *)arg)[6]),
        .mesh_row = 0,
        .mesh_col = 0,
        .router_m1 = 0,
        .router_n1 = 0,
        .softmax_scale = 0.0f,
    };
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_SCATTER_META_START);
    build_scatter_metadata(global_top_k_indices, actual_total_tokens, expert_token_counts, expert_memory_offsets, reverse_original_token_flat_idx, &cfg);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_SCATTER_META_END);

    uint32_t cerf_bitmask = 0;
    for (uint32_t e = 0; e < cfg.expert_number_each_layer; e++) {
        if (expert_token_counts[e] > 0) {
            cerf_bitmask |= (1u << e);
        }
    }
    bingo_cerf_write_mask(cerf_bitmask);
    BINGO_PRINTF(0, "[CERF] Dynamic update: bitmask=0x%x (experts=%u)\r\n",
                 cerf_bitmask, cfg.expert_number_each_layer);
    return 0;
}

static inline uint64_t __host_bingo_kernel_compute_swish_activation(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t *gate_project_hw_data = (int32_t *)(((uint64_t *)arg)[0]);
    float *swish_intermediate_buf = (float *)(((uint64_t *)arg)[1]);
    uint32_t valid_elements = (uint32_t)(((uint64_t *)arg)[2]);
    float swish_glu_scale_in = __host_bingo_kernel_unpack_f32(((uint64_t *)arg)[3]);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_SWISH_START);
    compute_swish_activation_tile(gate_project_hw_data, swish_intermediate_buf, valid_elements, swish_glu_scale_in);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_SWISH_END);
    return 0;
}

static inline uint64_t __host_bingo_kernel_compute_glu_multiplication(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float *swish_intermediate_buf = (float *)(((uint64_t *)arg)[0]);
    int32_t *up_projection_hw_data = (int32_t *)(((uint64_t *)arg)[1]);
    int8_t *activated_out_data = (int8_t *)(((uint64_t *)arg)[2]);
    uint32_t valid_elements = (uint32_t)(((uint64_t *)arg)[3]);
    float swish_glu_scale_in = __host_bingo_kernel_unpack_f32(((uint64_t *)arg)[4]);
    float swish_glu_scale_out = __host_bingo_kernel_unpack_f32(((uint64_t *)arg)[5]);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_GLU_START);
    compute_glu_multiplication_tile(swish_intermediate_buf, up_projection_hw_data, activated_out_data, valid_elements, swish_glu_scale_in, swish_glu_scale_out);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_GLU_END);
    return 0;
}

static inline void compute_hw_silu_glu_tile(
    int32_t *gate_silu_hw_data,
    int32_t *up_projection_hw_data,
    int8_t *activated_out_data,
    uint32_t valid_elements,
    float scale_in,
    float scale_out)
{
    for (uint32_t i = 0; i < valid_elements; i++) {
        float gated_silu = (float)gate_silu_hw_data[i] * scale_in;
        float uped_data = (float)up_projection_hw_data[i] * scale_in;
        float result_float = gated_silu * uped_data;
        int32_t result_int = (int32_t)(result_float * scale_out);
        if (result_int > 127) {
            result_int = 127;
        } else if (result_int < -128) {
            result_int = -128;
        }
        activated_out_data[i] = (int8_t)result_int;
    }
}

static inline uint64_t __host_bingo_kernel_compute_hw_silu_glu(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t *gate_silu_hw_data = (int32_t *)(((uint64_t *)arg)[0]);
    int32_t *up_projection_hw_data = (int32_t *)(((uint64_t *)arg)[1]);
    int8_t *activated_out_data = (int8_t *)(((uint64_t *)arg)[2]);
    uint32_t valid_elements = (uint32_t)(((uint64_t *)arg)[3]);
    float swish_glu_scale_in = __host_bingo_kernel_unpack_f32(((uint64_t *)arg)[4]);
    float swish_glu_scale_out = __host_bingo_kernel_unpack_f32(((uint64_t *)arg)[5]);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_GLU_START);
    compute_hw_silu_glu_tile(gate_silu_hw_data, up_projection_hw_data, activated_out_data, valid_elements, swish_glu_scale_in, swish_glu_scale_out);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_GLU_END);
    return 0;
}

static inline uint64_t __host_bingo_kernel_experts_result_accumulate(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t *shared_expert_hw_output = (int32_t *)(((uint64_t *)arg)[0]);
    int32_t *individual_experts_hw_output = (int32_t *)(((uint64_t *)arg)[1]);
    uint32_t *reverse_original_token_flat_idx = (uint32_t *)(((uint64_t *)arg)[2]);
    uint32_t *global_calculated_probability = (uint32_t *)(((uint64_t *)arg)[3]);
    int32_t *final_layer_output = (int32_t *)(((uint64_t *)arg)[4]);
    uint32_t actual_total_tokens = (uint32_t)(((uint64_t *)arg)[5]);
    moe_operator_cfg_t cfg = {
        .input_dimension = (uint32_t)(((uint64_t *)arg)[6]),
        .expert_number_each_layer = 0,
        .individual_expert_number_k = (uint32_t)(((uint64_t *)arg)[7]),
        .mesh_row = 0,
        .mesh_col = 0,
        .router_m1 = 0,
        .router_n1 = 0,
        .softmax_scale = __host_bingo_kernel_unpack_f32(((uint64_t *)arg)[8]),
    };
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_ACCUMULATE_START);
    experts_result_accumulate(shared_expert_hw_output, individual_experts_hw_output, reverse_original_token_flat_idx, global_calculated_probability, final_layer_output, actual_total_tokens, &cfg);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_ACCUMULATE_END);
    return 0;
}

static inline uint64_t __host_bingo_kernel_scatter_and_pad_input(void *arg)
{
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint32_t expert_id = (uint32_t)(((uint64_t *)arg)[0]);
    int8_t *global_input_A = (int8_t *)(((uint64_t *)arg)[1]);
    int8_t *padded_scatter_pool = (int8_t *)(((uint64_t *)arg)[2]);
    uint32_t *expert_token_counts = (uint32_t *)(((uint64_t *)arg)[3]);
    uint32_t *expert_memory_offsets = (uint32_t *)(((uint64_t *)arg)[4]);
    uint32_t *reverse_original_token_flat_idx = (uint32_t *)(((uint64_t *)arg)[5]);
    uint32_t input_dimension = (uint32_t)(((uint64_t *)arg)[6]);
    uint32_t max_padded_tokens = (uint32_t)(((uint64_t *)arg)[7]);
    uint32_t individual_expert_number_k = (uint32_t)(((uint64_t *)arg)[8]);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_SCATTER_PAD_START);
    scatter_and_pad_input_for_expert(
        global_input_A, padded_scatter_pool, expert_id,
        expert_token_counts, expert_memory_offsets, reverse_original_token_flat_idx,
        input_dimension, max_padded_tokens, individual_expert_number_k);
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_SCATTER_PAD_END);
    return 0;
}
#endif

#if defined(MOE_ENABLE_MULTI_CLUSTER_MOE) || defined(MOE_OPERATOR_CUSTOM) || \
    defined(BINGO_HAS_LEGACY_MOE_OPERATOR)
static inline uint64_t __host_bingo_kernel_moe_router_schedule(void *arg)
{
    __moe_host_timing_start(MOE_HOST_TIMING_ROUTER_SCHED);
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    __host_bingo_kernel_moe_router_schedule_args_t *args =
        (__host_bingo_kernel_moe_router_schedule_args_t *)arg;
    uint32_t total_tokens = (uint32_t)args->total_tokens;
    int32_t *hardware_output_buffer =
        (int32_t *)(uintptr_t)args->hardware_output_buffer_addr;
    uint16_t *global_indices_out =
        (uint16_t *)(uintptr_t)args->global_indices_out_addr;
    int32_t *global_scores_out =
        (int32_t *)(uintptr_t)args->global_scores_out_addr;
    uint32_t *expert_token_counts_out =
        (uint32_t *)(uintptr_t)args->expert_token_counts_out_addr;
#ifdef MOE_ENABLE_HW_SCHEDULER
    uint16_t *expert_token_refs =
        (uint16_t *)(uintptr_t)args->expert_token_refs_addr;
    uint32_t token_stride = (uint32_t)args->max_tokens_per_expert;
#endif
    uint32_t n_experts = (uint32_t)args->expert_number_each_layer;
#ifdef MOE_ENABLE_HW_SCHEDULER
#if !defined(MOE_HW_ROUTER_MESH_ROW) || !defined(MOE_HW_ROUTER_MESH_COL) || \
    !defined(MOE_HW_ROUTER_M1) || !defined(MOE_HW_ROUTER_N1) || \
    !defined(MOE_HW_ROUTER_MR_SHIFT) || !defined(MOE_HW_TOP_K)
#error "pure-HW Router path requires compile-time MOE_HW_ROUTER_* constants"
#endif
#if MOE_HW_TOP_K != 2
#error "pure-HW Router path is specialized for top_k=2"
#endif
    const uint32_t top_k = 2u;
    const uint32_t mesh_row = MOE_HW_ROUTER_MESH_ROW;
    const uint32_t mesh_col = MOE_HW_ROUTER_MESH_COL;
    const uint32_t router_m1 = MOE_HW_ROUTER_M1;
    const uint32_t router_n1 = MOE_HW_ROUTER_N1;
#else
    uint32_t top_k = (uint32_t)args->individual_expert_number_k;
    uint32_t mesh_row = (uint32_t)args->mesh_row;
    uint32_t mesh_col = (uint32_t)args->mesh_col;
    uint32_t router_m1 = (uint32_t)args->router_m1;
    uint32_t router_n1 = (uint32_t)args->router_n1;
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    if (MOE_SCHED_DEBUG_PRINT) {
        const int16_t *raw_router_scores = (const int16_t *)hardware_output_buffer;
        uint32_t row_stride = router_n1 * mesh_row * mesh_col;
        uint32_t tile_stride = router_m1 * row_stride;
        uint32_t dump_raw = row_stride < 16u ? row_stride : 16u;
        printf_safe("[ROUTER_TOPK_ADDR] hw=0x%08x_%08x idx=0x%08x_%08x scores=0x%08x_%08x total=%u row_stride=%u tile_stride=%u\r\n",
                    (uint32_t)(((uint64_t)(uintptr_t)hardware_output_buffer) >> 32),
                    (uint32_t)((uint64_t)(uintptr_t)hardware_output_buffer),
                    (uint32_t)(((uint64_t)(uintptr_t)global_indices_out) >> 32),
                    (uint32_t)((uint64_t)(uintptr_t)global_indices_out),
                    (uint32_t)(((uint64_t)(uintptr_t)global_scores_out) >> 32),
                    (uint32_t)((uint64_t)(uintptr_t)global_scores_out),
                    total_tokens, row_stride, tile_stride);
        printf_safe("[ROUTER_TOPK_PRE_RAW16] n2=0");
        for (uint32_t i = 0; i < dump_raw; i++) {
            printf_safe(" %d", (int)raw_router_scores[i]);
        }
        printf_safe("\r\n");
        if (n_experts > router_n1 * mesh_col) {
            printf_safe("[ROUTER_TOPK_PRE_RAW16] n2=1_off%u", (unsigned)tile_stride);
            for (uint32_t i = 0; i < dump_raw; i++) {
                printf_safe(" %d", (int)raw_router_scores[tile_stride + i]);
            }
            printf_safe("\r\n");
        }
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_ROUTER_SCHED_START);
#ifdef MOE_ENABLE_HW_SCHEDULER
    const int16_t *raw_router_scores = (const int16_t *)hardware_output_buffer;
    const uint32_t tokens_per_hw_tile =
        MOE_HW_ROUTER_M1 * MOE_HW_ROUTER_MESH_ROW;
    const uint32_t row_stride = MOE_HW_ROUTER_N1 *
        MOE_HW_ROUTER_MESH_ROW * MOE_HW_ROUTER_MESH_COL;
    const uint32_t col_stride =
        MOE_HW_ROUTER_MESH_ROW * MOE_HW_ROUTER_MESH_COL;
    const uint32_t tile_stride = MOE_HW_ROUTER_M1 * row_stride;
    const uint32_t experts_per_n2_tile =
        MOE_HW_ROUTER_N1 * MOE_HW_ROUTER_MESH_COL;
    uint32_t n2_count = (n_experts + experts_per_n2_tile - 1u) /
                        experts_per_n2_tile;
    uint32_t hw_tile_score_elems = n2_count * tile_stride;
    const uint32_t mr_mask = MOE_HW_ROUTER_MESH_ROW - 1u;
    uint32_t local_counts[MOE_MAX_EXPERTS];

    for (uint32_t e = 0u; e < n_experts; e++) {
        local_counts[e] = 0u;
    }
    for (uint32_t tile = 0u, tokens_done = 0u;
         tokens_done < total_tokens;
         tile++, tokens_done += tokens_per_hw_tile) {
        uint32_t remaining = total_tokens - tokens_done;
        uint32_t valid_tokens =
            (remaining > tokens_per_hw_tile) ? tokens_per_hw_tile : remaining;
        const int16_t *tile_scores =
            raw_router_scores + tile * hw_tile_score_elems;

        for (uint32_t local_t = 0u; local_t < valid_tokens; local_t++) {
            uint32_t row_base = (local_t >> MOE_HW_ROUTER_MR_SHIFT) * row_stride +
                                (local_t & mr_mask) * MOE_HW_ROUTER_MESH_COL;
            int32_t best0_score = INT32_MIN;
            int32_t best1_score = INT32_MIN;
            uint32_t best0_eid = UINT32_MAX;
            uint32_t best1_eid = UINT32_MAX;
            uint32_t eid = 0u;

            for (uint32_t n2 = 0u; n2 < n2_count; n2++) {
                const int16_t *n2_scores =
                    tile_scores + n2 * tile_stride + row_base;
                for (uint32_t n1 = 0u;
                     n1 < MOE_HW_ROUTER_N1 && eid < n_experts;
                     n1++) {
                    const int16_t *row_scores = n2_scores + n1 * col_stride;
                    for (uint32_t col = 0u;
                         col < MOE_HW_ROUTER_MESH_COL && eid < n_experts;
                         col++, eid++) {
                        int32_t score = (int32_t)row_scores[col];
                        if (score > best0_score ||
                            (score == best0_score && eid < best0_eid)) {
                            best1_score = best0_score;
                            best1_eid = best0_eid;
                            best0_score = score;
                            best0_eid = eid;
                        } else if (score > best1_score ||
                                   (score == best1_score && eid < best1_eid)) {
                            best1_score = score;
                            best1_eid = eid;
                        }
                    }
                }
            }

            uint32_t out_base = (tokens_done + local_t) << 1u;
            global_scores_out[out_base + 0u] = best0_score;
            if (MOE_SCHED_DEBUG_PRINT) {
                global_indices_out[out_base + 0u] = (uint16_t)best0_eid;
            }
            uint32_t pos0 = best0_eid * token_stride + local_counts[best0_eid]++;
            expert_token_refs[pos0] =
                BINGO_MOE_TOKEN_REF_PACK(tokens_done + local_t, 0u);
            global_scores_out[out_base + 1u] = best1_score;
            if (MOE_SCHED_DEBUG_PRINT) {
                global_indices_out[out_base + 1u] = (uint16_t)best1_eid;
            }
            uint32_t pos1 = best1_eid * token_stride + local_counts[best1_eid]++;
            expert_token_refs[pos1] =
                BINGO_MOE_TOKEN_REF_PACK(tokens_done + local_t, 1u);
        }
    }
    for (uint32_t e = 0u; e < n_experts; e++) {
        expert_token_counts_out[e] = local_counts[e];
    }
#else
    moe_operator_cfg_t cfg = {
        .input_dimension = 0,
        .expert_number_each_layer = n_experts,
        .individual_expert_number_k = top_k,
        .mesh_row = mesh_row,
        .mesh_col = mesh_col,
        .router_m1 = router_m1,
        .router_n1 = router_n1,
        .softmax_scale = 0.0f,
    };
    moe_router_global_schedule(total_tokens, hardware_output_buffer,
                               global_indices_out, global_scores_out,
                               expert_token_counts_out, &cfg);
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_HOST_ROUTER_SCHED_END);
    if (MOE_SCHED_DEBUG_PRINT) {
        const int16_t *raw_router_scores = (const int16_t *)hardware_output_buffer;
        uint32_t row_stride = router_n1 * mesh_row * mesh_col;
        uint32_t tile_stride = router_m1 * row_stride;
        uint32_t experts_per_n2_tile = router_n1 * mesh_col;
        uint32_t dump_tokens = total_tokens < 8u ? total_tokens : 8u;
        uint32_t dump_raw = row_stride < 16u ? row_stride : 16u;

        printf_safe("[ROUTER_RAW16] n2=0");
        for (uint32_t i = 0; i < dump_raw; i++) {
            printf_safe(" %d", (int)raw_router_scores[i]);
        }
        printf_safe("\r\n");
        if (n_experts > experts_per_n2_tile) {
            printf_safe("[ROUTER_RAW16] n2=1_off%u", (unsigned)tile_stride);
            for (uint32_t i = 0; i < dump_raw; i++) {
                printf_safe(" %d", (int)raw_router_scores[tile_stride + i]);
            }
            printf_safe("\r\n");
        }

        for (uint32_t t = 0; t < dump_tokens; t++) {
            printf_safe("[ROUTER_SCORE] T%u", (unsigned)t);
            for (uint32_t e = 0; e < n_experts; e++) {
                uint32_t n2 = e / experts_per_n2_tile;
                uint32_t n1 = (e / mesh_col) - n2 * router_n1;
                uint32_t idx = n2 * tile_stride +
                               (t / mesh_row) * row_stride +
                               n1 * mesh_row * mesh_col +
                               (t % mesh_row) * mesh_col +
                               (e % mesh_col);
                printf_safe(" e%u=%d", (unsigned)e, (int)raw_router_scores[idx]);
            }
            printf_safe("\r\n");
        }

        for (uint32_t t = 0; t < dump_tokens; t++) {
            uint32_t base = t * top_k;
            printf_safe("[ROUTER_TOPK] T%u", (unsigned)t);
            for (uint32_t k = 0; k < top_k; k++) {
                printf_safe(" k%u=e%u/s%d", (unsigned)k,
                            (unsigned)global_indices_out[base + k],
                            (int)global_scores_out[base + k]);
            }
            printf_safe("\r\n");
        }
    }
    __moe_host_timing_end(MOE_HOST_TIMING_ROUTER_SCHED);
    return 0;
}
#endif

#ifdef MOE_ENABLE_MULTI_CLUSTER_MOE
static inline void __host_moe_initialize_device_args(
    uintptr_t c2_stage_base,
    uintptr_t c3_stage_base,
    uint64_t c2_static_args_base,
    uint64_t c3_static_args_base,
    uint64_t c2_dynamic_args_base,
    uint64_t c3_dynamic_args_base,
    uint32_t slot_bytes,
    uint32_t num_slots,
    const __snax_bingo_moe_dynamic_expert_static_args_t *c2_static_values,
    const __snax_bingo_moe_dynamic_expert_static_args_t *c3_static_values)
{
    uint64_t static_bytes =
        (sizeof(__snax_bingo_moe_dynamic_expert_static_args_t) + 63u) & ~63u;
    uintptr_t c2_stage_slots = c2_stage_base + 64u;
    uintptr_t c3_stage_slots = c3_stage_base + 64u;
    __snax_bingo_moe_dynamic_expert_static_args_t *c2_static =
        (__snax_bingo_moe_dynamic_expert_static_args_t *)c2_stage_slots;
    __snax_bingo_moe_dynamic_expert_static_args_t *c3_static =
        (__snax_bingo_moe_dynamic_expert_static_args_t *)c3_stage_slots;
    memset((void *)c2_stage_base, 0, 64u);
    memset((void *)c3_stage_base, 0, 64u);
    memset((void *)c2_stage_slots, 0, (size_t)static_bytes);
    memset((void *)c3_stage_slots, 0, (size_t)static_bytes);
    *c2_static = *c2_static_values;
    *c3_static = *c3_static_values;
    asm volatile("fence rw, rw" ::: "memory");
    sys_dma_blk_memcpy(get_current_chip_id(),
                       c2_static_args_base,
                       (uint64_t)chiplet_addr_transform_full(get_current_chip_id(),
                                                             c2_stage_slots),
                       static_bytes);
    sys_dma_blk_memcpy(get_current_chip_id(),
                       c3_static_args_base,
                       (uint64_t)chiplet_addr_transform_full(get_current_chip_id(),
                                                             c3_stage_slots),
                       static_bytes);

    uint64_t bytes = (uint64_t)num_slots * (uint64_t)slot_bytes;
    memset((void *)c2_stage_slots, 0, (size_t)bytes);
    memset((void *)c3_stage_slots, 0, (size_t)bytes);
    asm volatile("fence rw, rw" ::: "memory");
    sys_dma_blk_memcpy(get_current_chip_id(),
                       c2_dynamic_args_base,
                       (uint64_t)chiplet_addr_transform_full(get_current_chip_id(),
                                                             c2_stage_slots),
                       bytes);
    sys_dma_blk_memcpy(get_current_chip_id(),
                       c3_dynamic_args_base,
                       (uint64_t)chiplet_addr_transform_full(get_current_chip_id(),
                                                             c3_stage_slots),
                       bytes);
    asm volatile("fence rw, rw" ::: "memory");
}

#ifndef MOE_ENABLE_HW_SCHEDULER
#include "host_moe_sw_path.h"
#endif

#ifdef MOE_ENABLE_HW_SCHEDULER
#include "host_moe_hw_path.h"

#endif

#endif

// ============================================================
// FP32 CVA6+Ara SIMD kernels — operations that VersaCore cannot handle
// These run on the CVA6 host processor with Ara RVV acceleration.
// Uses RISC-V Vector (RVV) intrinsics following the Ara softmax/exp patterns.
// Requires: enable_vec() called once at startup.
// ============================================================

// Disabled in the default HeMAiA host build: the current toolchain/container
// does not provide the Ara/RVV intrinsic types used below, and the offload MoE
// workloads do not use these host-side Ara kernels.
#if 0

#include "riscv_vector.h"

// Cephes polynomial exp approximation for f32 (from Ara exp kernel)
// Matches: ara/apps/exp/kernel/exp.c __exp_2xf32
static inline vfloat32m1_t __bingo_exp_f32(vfloat32m1_t x, size_t vl) {
    // Clamp input to avoid overflow/underflow
    vfloat32m1_t exp_hi = __riscv_vfmv_v_f_f32m1(88.3762626647949f, vl);
    vfloat32m1_t exp_lo = __riscv_vfmv_v_f_f32m1(-88.3762626647949f, vl);
    x = __riscv_vfmin_vv_f32m1(x, exp_hi, vl);
    x = __riscv_vfmax_vv_f32m1(x, exp_lo, vl);

    // Express exp(x) = exp(g + n*log(2))
    vfloat32m1_t cephes_LOG2EF = __riscv_vfmv_v_f_f32m1(1.44269504088896341f, vl);
    vfloat32m1_t cephes_exp_C1 = __riscv_vfmv_v_f_f32m1(0.693359375f, vl);
    vfloat32m1_t cephes_exp_C2 = __riscv_vfmv_v_f_f32m1(-2.12194440e-4f, vl);
    vfloat32m1_t half = __riscv_vfmv_v_f_f32m1(0.5f, vl);

    vfloat32m1_t fx = __riscv_vfmacc_vv_f32m1(half, x, cephes_LOG2EF, vl);
    vint32m1_t tmp = __riscv_vfcvt_x_f_v_i32m1(fx, vl);
    vfloat32m1_t ftmp = __riscv_vfcvt_f_x_v_f32m1(tmp, vl);

    // Correct for floor behavior
    vbool32_t mask = __riscv_vmflt_vv_f32m1_b32(fx, ftmp, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    ftmp = __riscv_vfsub_vv_f32m1(ftmp, __riscv_vmerge_vvm_f32m1(
        __riscv_vfmv_v_f_f32m1(0.0f, vl), one, mask, vl), vl);
    tmp = __riscv_vfcvt_x_f_v_i32m1(ftmp, vl);

    x = __riscv_vfsub_vv_f32m1(x, __riscv_vfmul_vv_f32m1(ftmp, cephes_exp_C1, vl), vl);
    x = __riscv_vfsub_vv_f32m1(x, __riscv_vfmul_vv_f32m1(ftmp, cephes_exp_C2, vl), vl);
    vfloat32m1_t z = __riscv_vfmul_vv_f32m1(x, x, vl);

    // Polynomial approx
    vfloat32m1_t y = __riscv_vfmv_v_f_f32m1(1.9875691500E-4f, vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(1.3981999507E-3f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(8.3334519073E-3f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(4.1665795894E-2f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(1.6666665459E-1f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, x, __riscv_vfmv_v_f_f32m1(5.0000001201E-1f, vl), vl);
    y = __riscv_vfmacc_vv_f32m1(__riscv_vfadd_vv_f32m1(x, one, vl), y, z, vl);

    // Scale by 2^n via integer addition to exponent bits
    vint32m1_t shift = __riscv_vadd_vv_i32m1(tmp, __riscv_vmv_v_x_i32m1(127, vl), vl);
    shift = __riscv_vsll_vx_i32m1(shift, 23, vl);
    y = __riscv_vfmul_vv_f32m1(y, __riscv_vreinterpret_v_i32m1_f32m1(shift), vl);
    return y;
}

static inline uint64_t __host_bingo_kernel_rmsnorm_f32(void *arg){
    // RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
    // Arg0: float* input_addr
    // Arg1: float* weight_addr (gamma)
    // Arg2: float* output_addr
    // Arg3: uint64_t hidden_dim
    // Arg4: uint64_t num_tokens (seq_len)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* weight = (float*)(((uint64_t *)arg)[1]);
    float* output = (float*)(((uint64_t *)arg)[2]);
    uint64_t hidden_dim    = ((uint64_t *)arg)[3];
    uint64_t num_tokens    = ((uint64_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    float eps = 1e-6f;
    for (uint64_t t = 0; t < num_tokens; t++) {
        float *x_row = input + t * hidden_dim;
        float *o_row = output + t * hidden_dim;

        // Pass 1: compute sum(x^2) using RVV
        float ss = 0.0f;
        uint64_t avl = hidden_dim;
        float *ptr = x_row;
        for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
            vl = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
            vfloat32m1_t sq = __riscv_vfmul_vv_f32m1(v, v, vl);
            // Scalar reduction
            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t sum_v = __riscv_vfredosum_vs_f32m1_f32m1(sq, zero, vl);
            ss += __riscv_vfmv_f_s_f32m1_f32(sum_v);
        }
        float rms = 1.0f / __builtin_sqrtf(ss / (float)hidden_dim + eps);

        // Pass 2: normalize and scale using RVV
        avl = hidden_dim;
        float *w_ptr = weight;
        float *i_ptr = x_row;
        float *o_ptr = o_row;
        for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
             avl -= vl, i_ptr += vl, w_ptr += vl, o_ptr += vl) {
            vl = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(i_ptr, vl);
            vfloat32m1_t w = __riscv_vle32_v_f32m1(w_ptr, vl);
            vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(v, rms, vl);
            vfloat32m1_t result = __riscv_vfmul_vv_f32m1(scaled, w, vl);
            __riscv_vse32_v_f32m1(o_ptr, result, vl);
        }
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_softmax_f32(void *arg){
    // Arg0-3: input, output, num_rows, row_length; Arg4: scratchpad_ptr
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_rows      = ((uint64_t *)arg)[2];
    uint64_t row_length    = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[4];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    for (uint64_t r = 0; r < num_rows; r++) {
        float *in_row = input + r * row_length;
        float *out_row = output + r * row_length;
        uint64_t avl = row_length;

        // Pass 1: find max (RVV reduction)
        float max_val = in_row[0];
        float *ptr = in_row;
        uint64_t rem = avl;
        for (size_t vl = __riscv_vsetvl_e32m1(rem); rem > 0; rem -= vl, ptr += vl) {
            vl = __riscv_vsetvl_e32m1(rem);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
            vfloat32m1_t init = __riscv_vfmv_v_f_f32m1(max_val, vl);
            vfloat32m1_t rmax = __riscv_vfredmax_vs_f32m1_f32m1(v, init, vl);
            max_val = __riscv_vfmv_f_s_f32m1_f32(rmax);
        }

        // Pass 2: exp(x - max) and accumulate sum (RVV)
        float sum = 0.0f;
        ptr = in_row;
        float *optr = out_row;
        rem = avl;
        for (size_t vl = __riscv_vsetvl_e32m1(rem); rem > 0;
             rem -= vl, ptr += vl, optr += vl) {
            vl = __riscv_vsetvl_e32m1(rem);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
            vfloat32m1_t shifted = __riscv_vfsub_vf_f32m1(v, max_val, vl);
            vfloat32m1_t expv = __bingo_exp_f32(shifted, vl);
            __riscv_vse32_v_f32m1(optr, expv, vl);
            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            vfloat32m1_t rsum = __riscv_vfredosum_vs_f32m1_f32m1(expv, zero, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(rsum);
        }

        // Pass 3: divide by sum (RVV)
        float inv_sum = 1.0f / sum;
        optr = out_row;
        rem = avl;
        for (size_t vl = __riscv_vsetvl_e32m1(rem); rem > 0; rem -= vl, optr += vl) {
            vl = __riscv_vsetvl_e32m1(rem);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(optr, vl);
            vfloat32m1_t result = __riscv_vfmul_vf_f32m1(v, inv_sum, vl);
            __riscv_vse32_v_f32m1(optr, result, vl);
        }
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    // Write output info to scratchpad for successor gating kernels
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_rows * row_length;
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_silu_mul_f32(void *arg){
    // SiLU(gate) * up: out[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
    // Uses RVV exp intrinsic for vectorized sigmoid
    // Arg0: float* gate_addr
    // Arg1: float* up_addr
    // Arg2: float* output_addr
    // Arg3: uint64_t num_elements
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* gate   = (float*)(((uint64_t *)arg)[0]);
    float* up     = (float*)(((uint64_t *)arg)[1]);
    float* output = (float*)(((uint64_t *)arg)[2]);
    uint64_t num_elements  = ((uint64_t *)arg)[3];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint64_t avl = num_elements;
    float *g_ptr = gate, *u_ptr = up, *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
         avl -= vl, g_ptr += vl, u_ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t gv = __riscv_vle32_v_f32m1(g_ptr, vl);
        vfloat32m1_t uv = __riscv_vle32_v_f32m1(u_ptr, vl);
        // sigmoid(g) = 1 / (1 + exp(-g))
        vfloat32m1_t neg_g = __riscv_vfneg_v_f32m1(gv, vl);
        vfloat32m1_t exp_neg = __bingo_exp_f32(neg_g, vl);
        vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
        vfloat32m1_t denom = __riscv_vfadd_vv_f32m1(one, exp_neg, vl);
        vfloat32m1_t sigmoid = __riscv_vfdiv_vv_f32m1(one, denom, vl);
        // silu(g) * up = g * sigmoid(g) * up
        vfloat32m1_t silu = __riscv_vfmul_vv_f32m1(gv, sigmoid, vl);
        vfloat32m1_t result = __riscv_vfmul_vv_f32m1(silu, uv, vl);
        __riscv_vse32_v_f32m1(o_ptr, result, vl);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

// ---- Generic binary elementwise: out[i] = op(a[i], b[i]) ----
// All share the same arg layout: a_addr, b_addr, output_addr, num_elements

#define DEFINE_FP32_BINARY_KERNEL(name, vec_op)                                 \
static inline uint64_t __host_bingo_kernel_##name##_f32(void *arg){              \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);                     \
    float* a      = (float*)(((uint64_t *)arg)[0]);                             \
    float* b      = (float*)(((uint64_t *)arg)[1]);                             \
    float* output = (float*)(((uint64_t *)arg)[2]);                             \
    uint64_t num_elements  = ((uint64_t *)arg)[3];                              \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);                       \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);                         \
    uint64_t avl = num_elements;                                                \
    float *a_ptr = a, *b_ptr = b, *o_ptr = output;                             \
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;                       \
         avl -= vl, a_ptr += vl, b_ptr += vl, o_ptr += vl) {                   \
        vl = __riscv_vsetvl_e32m1(avl);                                         \
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a_ptr, vl);                    \
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b_ptr, vl);                    \
        vfloat32m1_t result = vec_op(va, vb, vl);                              \
        __riscv_vse32_v_f32m1(o_ptr, result, vl);                              \
    }                                                                           \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);                           \
    return BINGO_RET_SUCC;                                                                   \
}

DEFINE_FP32_BINARY_KERNEL(add, __riscv_vfadd_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(sub, __riscv_vfsub_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(mul, __riscv_vfmul_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(div, __riscv_vfdiv_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(max, __riscv_vfmax_vv_f32m1)
DEFINE_FP32_BINARY_KERNEL(min, __riscv_vfmin_vv_f32m1)

// ---- INT32 elementwise add (for inter-cluster partial-D accumulation) ----
// Used when DSE picks K-split tilings: each cluster produces a partial D
// in INT32, and partial Ds from clusters covering the same (M,N) region
// must be summed with this kernel to yield the final INT32 D.
// Arg layout (same as FP32 binary): a_addr, b_addr, output_addr, num_elements
static inline uint64_t __host_bingo_kernel_add_i32(void *arg){
    // Arg0-3: a, b, output, num_elements; Arg4: precision (ignored); Arg5: scratchpad_ptr
    // (reads the unified ara_binary_args layout; precision is a no-op for int32 add)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t* a      = (int32_t*)(((uint64_t *)arg)[0]);
    int32_t* b      = (int32_t*)(((uint64_t *)arg)[1]);
    int32_t* output = (int32_t*)(((uint64_t *)arg)[2]);
    uint64_t num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[5];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint64_t avl = num_elements;
    int32_t *a_ptr = a, *b_ptr = b, *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
         avl -= vl, a_ptr += vl, b_ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t va = __riscv_vle32_v_i32m1(a_ptr, vl);
        vint32m1_t vb = __riscv_vle32_v_i32m1(b_ptr, vl);
        vint32m1_t result = __riscv_vadd_vv_i32m1(va, vb, vl);
        __riscv_vse32_v_i32m1(o_ptr, result, vl);
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return BINGO_RET_SUCC;
}

// ---- Generic unary elementwise: out[i] = op(x[i]) ----
// Arg layout: input_addr, output_addr, num_elements

#define DEFINE_FP32_UNARY_KERNEL(name, body)                                    \
static inline uint64_t __host_bingo_kernel_##name##_f32(void *arg){              \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);                     \
    float* input  = (float*)(((uint64_t *)arg)[0]);                             \
    float* output = (float*)(((uint64_t *)arg)[1]);                             \
    uint64_t num_elements = ((uint64_t *)arg)[2];                               \
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);                       \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);                         \
    uint64_t avl = num_elements;                                                \
    float *i_ptr = input, *o_ptr = output;                                      \
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;                       \
         avl -= vl, i_ptr += vl, o_ptr += vl) {                                \
        vl = __riscv_vsetvl_e32m1(avl);                                         \
        vfloat32m1_t v = __riscv_vle32_v_f32m1(i_ptr, vl);                     \
        vfloat32m1_t result;                                                    \
        body                                                                    \
        __riscv_vse32_v_f32m1(o_ptr, result, vl);                              \
    }                                                                           \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);                           \
    return BINGO_RET_SUCC;                                                                   \
}

// relu: max(0, x)
DEFINE_FP32_UNARY_KERNEL(relu, {
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    result = __riscv_vfmax_vv_f32m1(v, zero, vl);
})

// neg: -x
DEFINE_FP32_UNARY_KERNEL(neg, {
    result = __riscv_vfneg_v_f32m1(v, vl);
})

// abs: |x|  (uses neg + max)
DEFINE_FP32_UNARY_KERNEL(abs, {
    vfloat32m1_t neg_v = __riscv_vfneg_v_f32m1(v, vl);
    result = __riscv_vfmax_vv_f32m1(v, neg_v, vl);
})

// exp: exp(x) using Cephes polynomial
DEFINE_FP32_UNARY_KERNEL(exp, {
    result = __bingo_exp_f32(v, vl);
})

// sigmoid: 1 / (1 + exp(-x))
DEFINE_FP32_UNARY_KERNEL(sigmoid, {
    vfloat32m1_t neg_v = __riscv_vfneg_v_f32m1(v, vl);
    vfloat32m1_t exp_neg = __bingo_exp_f32(neg_v, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t denom = __riscv_vfadd_vv_f32m1(one, exp_neg, vl);
    result = __riscv_vfdiv_vv_f32m1(one, denom, vl);
})

// tanh: (exp(2x) - 1) / (exp(2x) + 1)
DEFINE_FP32_UNARY_KERNEL(tanh, {
    vfloat32m1_t two = __riscv_vfmv_v_f_f32m1(2.0f, vl);
    vfloat32m1_t two_x = __riscv_vfmul_vv_f32m1(v, two, vl);
    vfloat32m1_t exp_2x = __bingo_exp_f32(two_x, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t num = __riscv_vfsub_vv_f32m1(exp_2x, one, vl);
    vfloat32m1_t den = __riscv_vfadd_vv_f32m1(exp_2x, one, vl);
    result = __riscv_vfdiv_vv_f32m1(num, den, vl);
})

// sqrt: sqrt(x)
DEFINE_FP32_UNARY_KERNEL(sqrt, {
    result = __riscv_vfsqrt_v_f32m1(v, vl);
})

// reciprocal: 1/x
DEFINE_FP32_UNARY_KERNEL(reciprocal, {
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    result = __riscv_vfdiv_vv_f32m1(one, v, vl);
})

// silu: x * sigmoid(x) = x / (1 + exp(-x))
DEFINE_FP32_UNARY_KERNEL(silu, {
    vfloat32m1_t neg_v = __riscv_vfneg_v_f32m1(v, vl);
    vfloat32m1_t exp_neg = __bingo_exp_f32(neg_v, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t denom = __riscv_vfadd_vv_f32m1(one, exp_neg, vl);
    vfloat32m1_t sig = __riscv_vfdiv_vv_f32m1(one, denom, vl);
    result = __riscv_vfmul_vv_f32m1(v, sig, vl);
})

// gelu: GELU(x) ~ x * sigmoid(1.702 * x) (fast approximation)
DEFINE_FP32_UNARY_KERNEL(gelu, {
    vfloat32m1_t coeff = __riscv_vfmv_v_f_f32m1(1.702f, vl);
    vfloat32m1_t scaled = __riscv_vfmul_vv_f32m1(v, coeff, vl);
    vfloat32m1_t neg_s = __riscv_vfneg_v_f32m1(scaled, vl);
    vfloat32m1_t exp_neg = __bingo_exp_f32(neg_s, vl);
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t denom = __riscv_vfadd_vv_f32m1(one, exp_neg, vl);
    vfloat32m1_t sig = __riscv_vfdiv_vv_f32m1(one, denom, vl);
    result = __riscv_vfmul_vv_f32m1(v, sig, vl);
})

// ---- Reduction kernels ----
// out[0] = reduce_op(x[0..n])
// Arg layout: input_addr, output_addr, num_elements

static inline uint64_t __host_bingo_kernel_reduce_sum_f32(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_elements = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    float acc = 0.0f;
    uint64_t avl = num_elements;
    float *ptr = input;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t rsum = __riscv_vfredosum_vs_f32m1_f32m1(v, zero, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(rsum);
    }
    *output = acc;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_reduce_max_f32(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_elements = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    float max_val = input[0];
    uint64_t avl = num_elements;
    float *ptr = input;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t init = __riscv_vfmv_v_f_f32m1(max_val, vl);
        vfloat32m1_t rmax = __riscv_vfredmax_vs_f32m1_f32m1(v, init, vl);
        max_val = __riscv_vfmv_f_s_f32m1_f32(rmax);
    }
    *output = max_val;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_reduce_mean_f32(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float* input  = (float*)(((uint64_t *)arg)[0]);
    float* output = (float*)(((uint64_t *)arg)[1]);
    uint64_t num_elements = ((uint64_t *)arg)[2];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    float acc = 0.0f;
    uint64_t avl = num_elements;
    float *ptr = input;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t rsum = __riscv_vfredosum_vs_f32m1_f32m1(v, zero, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(rsum);
    }
    *output = acc / (float)num_elements;
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return BINGO_RET_SUCC;
}

// ============================================================
// Data type conversion kernels for mixed-precision inference
// FP32 <-> INT8 at the boundary between CVA6 and VersaCore
// (FP16 -> INT8 quantize lives further down, after the BINGO_HAVE_FP16_VEC
//  block: see __host_bingo_kernel_quantize_f16i8.)
// ============================================================

static inline uint64_t __host_bingo_kernel_quantize_f32i8(void *arg){
    // Arg0-3: input, output, scale_out, num_elements; Arg4: precision (ignored); Arg5: scratchpad_ptr
    // (reads the unified ara_convert_args layout; precision is a no-op for the conversion)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    float*   input      = (float*)(((uint64_t *)arg)[0]);
    int8_t*  output     = (int8_t*)(((uint64_t *)arg)[1]);
    float*   scale_out  = (float*)(((uint64_t *)arg)[2]);
    uint64_t num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[5];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);

    // Pass 1: find max(|x|) using RVV
    float abs_max = 0.0f;
    uint64_t avl = num_elements;
    float *ptr = input;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t neg_v = __riscv_vfneg_v_f32m1(v, vl);
        vfloat32m1_t abs_v = __riscv_vfmax_vv_f32m1(v, neg_v, vl);
        vfloat32m1_t init = __riscv_vfmv_v_f_f32m1(abs_max, vl);
        vfloat32m1_t rmax = __riscv_vfredmax_vs_f32m1_f32m1(abs_v, init, vl);
        abs_max = __riscv_vfmv_f_s_f32m1_f32(rmax);
    }

    // Compute scale
    float scale = abs_max / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;  // avoid div-by-zero for near-zero input
    *scale_out = scale;
    float inv_scale = 1.0f / scale;

    // Pass 2: quantize — scale, round, clamp, narrow to int8
    avl = num_elements;
    ptr = input;
    int8_t *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
         avl -= vl, ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(ptr, vl);
        vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(v, inv_scale, vl);
        // Round to nearest integer
        vint32m1_t rounded = __riscv_vfcvt_x_f_v_i32m1(scaled, vl);
        // Clamp to [-128, 127]
        vint32m1_t lo = __riscv_vmv_v_x_i32m1(-128, vl);
        vint32m1_t hi = __riscv_vmv_v_x_i32m1(127, vl);
        rounded = __riscv_vmax_vv_i32m1(rounded, lo, vl);
        rounded = __riscv_vmin_vv_i32m1(rounded, hi, vl);
        // Narrow int32 -> int8 via scalar extract (safe for initial bring-up)
        for (size_t i = 0; i < vl; i++) {
            int32_t val = __riscv_vmv_x_s_i32m1_i32(
                __riscv_vslidedown_vx_i32m1(rounded, i, vl));
            o_ptr[i] = (int8_t)val;
        }
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return BINGO_RET_SUCC;
}

static inline uint64_t __host_bingo_kernel_dequantize_i32f32(void *arg){
    // Dequantize INT32 GEMM accumulator to FP32
    // y[i] = int32_input[i] * combined_scale
    // where combined_scale = scale_a * scale_b (pre-computed, stored at scale_addr)
    // Arg0-3: input, output, scale, num_elements; Arg4: precision (ignored); Arg5: scratchpad_ptr
    // (reads the unified ara_convert_args layout; precision is a no-op for the conversion)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    int32_t* input     = (int32_t*)(((uint64_t *)arg)[0]);
    float*   output    = (float*)(((uint64_t *)arg)[1]);
    float*   scale_ptr = (float*)(((uint64_t *)arg)[2]);
    uint64_t num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[5];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    float combined_scale = *scale_ptr;

    uint64_t avl = num_elements;
    int32_t *i_ptr = input;
    float   *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl > 0;
         avl -= vl, i_ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t vi = __riscv_vle32_v_i32m1(i_ptr, vl);
        vfloat32m1_t vf = __riscv_vfcvt_f_x_v_f32m1(vi, vl);
        vfloat32m1_t result = __riscv_vfmul_vf_f32m1(vf, combined_scale, vl);
        __riscv_vse32_v_f32m1(o_ptr, result, vl);
    }

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return BINGO_RET_SUCC;
}

#endif
// ================================================================
// DARTS Unified CERF Gating Kernel
// ================================================================
// Supports multiple activation modes via args->mode:
//   MODE_TOP_K (0):    Read logits from predecessor scratchpad, select top-k
//   MODE_THRESHOLD (1): Read confidence from predecessor scratchpad, activate if < threshold
//   MODE_STATIC (2):   Use compile-time cerf_write_mask directly
//
// Args layout (__host_bingo_kernel_cerf_gating_args_t):
//   [0] mode
//   [1] pred_scratchpad_addr (unused for static)
//   [2] cerf_controlled_mask
//   [3] top_k_or_threshold: top_k | threshold_bits | cerf_write_mask
//   [4] cerf_group_ids_addr: &cerf_group_ids[] | unused (0)
//   [5] cond_activation_addr: per-expert uint8_t[] (SW guard for group sharing)
//   [6] scratchpad_ptr
//
// Two-level gating (for >32 experts with CERF group sharing):
//   Level 1 (HW CERF): Inactive groups skip entire expert clusters at zero cost.
//   Level 2 (SW guard): Within active groups, only selected experts compute.
//     The gating kernel writes 1/0 per expert to cond_activation_addr[].
//     Expert kernels read their slot and early-return if 0.
//     When experts <= 32 (no sharing), cond_activation_addr can be 0 (skip SW guard).
static inline uint64_t __host_bingo_kernel_cerf_gating(void *arg){
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint64_t *args = (uint64_t *)arg;
    uint32_t mode = (uint32_t)args[0];
    bingo_kernel_scratchpad_t* pred_sp = (bingo_kernel_scratchpad_t*)(uintptr_t)args[1];
    uint32_t cerf_controlled_mask = (uint32_t)args[2];
    uint64_t top_k_or_threshold = args[3];
    uint64_t cerf_group_ids_addr = args[4];
    uint8_t *cond_activation = (uint8_t *)(uintptr_t)args[5];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)args[6];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_START);

    uint32_t cerf_write_mask = 0;

    if (mode == BINGO_GATING_MODE_TOP_K) {
        uint32_t top_k = (uint32_t)top_k_or_threshold;
        uint8_t *cerf_group_ids = (uint8_t *)(uintptr_t)cerf_group_ids_addr;
        float *logits = (float *)(uintptr_t)pred_sp->return_value;
        uint32_t num_experts = pred_sp->num_return_values;
        if (num_experts > 256) num_experts = 256;  // sanity bound

        // Clear per-expert activation array (all experts start as inactive)
        if (cond_activation) {
            for (uint32_t e = 0; e < num_experts; e++)
                cond_activation[e] = 0;
        }

        // Top-k selection: find k experts with highest logit values
        bool used[256] = {false};
        for (uint32_t k = 0; k < top_k; k++) {
            float best = -1e30f;
            uint32_t best_idx = 0;
            for (uint32_t e = 0; e < num_experts; e++) {
                if (!used[e] && logits[e] > best) {
                    best = logits[e];
                    best_idx = e;
                }
            }
            // Level 1: Mark the CERF group of this expert as active (HW skip)
            cerf_write_mask |= (1 << cerf_group_ids[best_idx]);
            // Level 2: Mark this specific expert as active (SW guard)
            if (cond_activation)
                cond_activation[best_idx] = 1;
            used[best_idx] = true;
        }
        BINGO_PRINTF(1, "Chip(%x, %x): [Host] CERF Gating top_k: n=%d, k=%d, ctrl=0x%04x, write=0x%04x\r\n",
               get_current_chip_loc_x(), get_current_chip_loc_y(),
               num_experts, top_k, cerf_controlled_mask, cerf_write_mask);

    } else if (mode == BINGO_GATING_MODE_THRESHOLD) {
        union { uint32_t u; float f; } thresh_conv;
        thresh_conv.u = (uint32_t)top_k_or_threshold;
        float threshold = thresh_conv.f;
        float *confidence = (float *)(uintptr_t)pred_sp->return_value;

        if (*confidence < threshold) {
            cerf_write_mask = cerf_controlled_mask;  // activate all → continue
        }

    } else if (mode == BINGO_GATING_MODE_STATIC) {
        cerf_write_mask = (uint32_t)top_k_or_threshold;
    }

    // Write CERF registers: single bitmask update (read-modify-write)
    bingo_cerf_update(cerf_controlled_mask, cerf_write_mask);

    BINGO_TRACE_MARKER(BINGO_TRACE_DUMMY_KERNEL_END);
    // Store per-expert activation array pointer in scratchpad so expert kernels
    // can find it via the gating node's scratchpad
    sp->return_value = (uint32_t)(uintptr_t)cond_activation;
    sp->num_return_values = 0;
    return BINGO_RET_SUCC;
}

// ============================================================
// Multi-precision Ara kernels (precision passed as a runtime arg)
// ============================================================
// See the FP32 RVV block above: keep the multi-precision Ara dispatchers out of
// the default host build regardless of external user flags.
#if 0
// The typed __host_bingo_kernel_<op>_f32 / _i32 kernels (and quantize_f32i8 /
// dequantize_i32f32) above are the FP32/INT32 entry points used by ci_ara and the
// bingo graph. The dispatchers below add fp16 / int8 / int16 variants for the
// precision sweep. Each dispatcher reads a BINGO_PREC_* word from a fixed arg
// slot; for BINGO_PREC_FP32 it delegates to the matching typed kernel so the
// FP32 path is byte-for-byte identical.
//
// Applicability (see ara_sweep.h for the per-op precision lists the apps use):
//   - fp16: all float ops. Elementwise/native-unary/reduce_max use native f16
//           vectors; transcendentals and the compound ops (softmax/rmsnorm/
//           silu_mul) widen f16->f32 (vfwcvt), reuse the fp32 math, then narrow
//           f32->f16 (vfncvt) so the fp32 polynomial constants are reused.
//   - int8/int16: only integer-meaningful ops (add/sub/mul/max/min, relu/neg/abs,
//           reduce_sum/reduce_max). reduce_sum widens to an int32 accumulator
//           (overflow-safe for any VLEN); reduce_* write their scalar as int32.
// Unsupported (op, precision) combos return BINGO_RET_FAIL (defensive; the sweep
// apps never iterate them).

#if defined(__riscv_zvfh) && !defined(BINGO_HAVE_FP16_VEC)
#define BINGO_HAVE_FP16_VEC 1
#endif
#ifndef BINGO_HAVE_FP16_VEC
#define BINGO_HAVE_FP16_VEC 0
#endif

#if BINGO_HAVE_FP16_VEC
// Narrow one f32m1 vector to f16mf2. WORKAROUND for an Ara HW bug: the in-place
// narrowing fp conversion the compiler emits for __riscv_vfncvt_f_f_w_f16mf2()
// (e.g. "vfncvt.f.f.w v4,v4", dest reg overlapping the lowest part of the wide
// source reg group -- spec-legal) produces a wrong f32->f16 result on Ara.
// Forcing the destination into a register distinct from the source (early-clobber
// "=&vr") computes correctly. Verified: every transcendental/compound fp16 kernel
// (exp/sigmoid/sqrt/tanh/reciprocal/gelu/silu/silu_mul/softmax/rmsnorm) FAILs the
// op-cost-sweep correctness check with the in-place narrow and PASSes with this.
// The vsetvli is inside the asm because the asm is opaque to the compiler's vtype
// tracking; the caller's next vector op re-establishes vtype as needed.
static inline vfloat16mf2_t __bingo_narrow_f32m1_f16mf2(vfloat32m1_t w, size_t vl) {
    vfloat16mf2_t d;
    asm volatile("vsetvli zero, %2, e16, mf2, ta, ma\n\t"
                 "vfncvt.f.f.w %0, %1"
                 : "=&vr"(d) : "vr"(w), "r"(vl));
    return d;
}
#endif

// ============================================================
// FP16 -> INT8 quantize. Sibling of __host_bingo_kernel_quantize_f32i8
// (see the FP32<->INT8 conversion section above); lives here because it
// uses the BINGO_HAVE_FP16_VEC widen path defined just above. Identical
// per-tensor symmetric scheme (scale = max(|x|)/127), only the input load
// differs: FP16 is widened to FP32 (vfwcvt) before reusing the f32 math.
// ============================================================
static inline uint64_t __host_bingo_kernel_quantize_f16i8(void *arg){
    // Arg0-3: input(fp16), output(int8), scale_out, num_elements; Arg4: precision (ignored); Arg5: scratchpad_ptr
    // (reads the unified ara_convert_args layout; precision is a no-op for the conversion)
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_START);
    uint16_t* input      = (uint16_t*)(((uint64_t *)arg)[0]);
    int8_t*   output     = (int8_t*)(((uint64_t *)arg)[1]);
    float*    scale_out  = (float*)(((uint64_t *)arg)[2]);
    uint64_t  num_elements = ((uint64_t *)arg)[3];
    bingo_kernel_scratchpad_t* sp = (bingo_kernel_scratchpad_t*)(uintptr_t)((uint64_t *)arg)[5];
    BINGO_TRACE_MARKER(BINGO_TRACE_KERNEL_ARG_PARSE_END);

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);

#if BINGO_HAVE_FP16_VEC
    // Pass 1: widen fp16->fp32, find max(|x|) using RVV
    float abs_max = 0.0f;
    uint64_t avl = num_elements;
    _Float16 *ptr = (_Float16*)input;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl > 0; avl -= vl, ptr += vl) {
        vl = __riscv_vsetvl_e16mf2(avl);
        vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(ptr, vl), vl);
        vfloat32m1_t neg_v = __riscv_vfneg_v_f32m1(v, vl);
        vfloat32m1_t abs_v = __riscv_vfmax_vv_f32m1(v, neg_v, vl);
        vfloat32m1_t init = __riscv_vfmv_v_f_f32m1(abs_max, vl);
        vfloat32m1_t rmax = __riscv_vfredmax_vs_f32m1_f32m1(abs_v, init, vl);
        abs_max = __riscv_vfmv_f_s_f32m1_f32(rmax);
    }

    // Compute scale
    float scale = abs_max / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;  // avoid div-by-zero for near-zero input
    *scale_out = scale;
    float inv_scale = 1.0f / scale;

    // Pass 2: widen fp16->fp32, scale, round, clamp, narrow to int8
    avl = num_elements;
    ptr = (_Float16*)input;
    int8_t *o_ptr = output;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl > 0;
         avl -= vl, ptr += vl, o_ptr += vl) {
        vl = __riscv_vsetvl_e16mf2(avl);
        vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(ptr, vl), vl);
        vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(v, inv_scale, vl);
        // Round to nearest integer
        vint32m1_t rounded = __riscv_vfcvt_x_f_v_i32m1(scaled, vl);
        // Clamp to [-128, 127]
        vint32m1_t lo = __riscv_vmv_v_x_i32m1(-128, vl);
        vint32m1_t hi = __riscv_vmv_v_x_i32m1(127, vl);
        rounded = __riscv_vmax_vv_i32m1(rounded, lo, vl);
        rounded = __riscv_vmin_vv_i32m1(rounded, hi, vl);
        // Narrow int32 -> int8 via scalar extract (safe for initial bring-up)
        for (size_t i = 0; i < vl; i++) {
            int32_t val = __riscv_vmv_x_s_i32m1_i32(
                __riscv_vslidedown_vx_i32m1(rounded, i, vl));
            o_ptr[i] = (int8_t)val;
        }
    }
#else
    // Scalar path without fp16 vector support: convert each half to fp32 in SW.
    // Pass 1: find max(|x|)
    float abs_max = 0.0f;
    for (uint64_t i = 0; i < num_elements; i++) {
        float x = __bingo_fp16_to_fp32(input[i]);
        float ax = x < 0.0f ? -x : x;
        if (ax > abs_max) abs_max = ax;
    }
    // Compute scale
    float scale = abs_max / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    *scale_out = scale;
    float inv_scale = 1.0f / scale;
    // Pass 2: scale, round, clamp to int8
    for (uint64_t i = 0; i < num_elements; i++) {
        float scaled = __bingo_fp16_to_fp32(input[i]) * inv_scale;
        int32_t r = (int32_t)(scaled < 0.0f ? scaled - 0.5f : scaled + 0.5f);
        if (r < -128) r = -128;
        if (r > 127)  r = 127;
        output[i] = (int8_t)r;
    }
#endif

    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    sp->return_value = (uint32_t)(uintptr_t)output;
    sp->num_return_values = num_elements;
    return BINGO_RET_SUCC;
}

// ---- typed native binary impls: out[i] = vop(a[i], b[i]) ----
#define __BINGO_BINARY_IMPL(op, P, T, ESET, ELD, EST, VT, VOP)                 \
static inline void __bingo_##op##_##P(const T* a, const T* b, T* o, uint64_t n){\
    uint64_t avl = n; const T *ap=a,*bp=b; T *op_=o;                           \
    for (size_t vl = ESET(avl); avl>0; avl-=vl, ap+=vl, bp+=vl, op_+=vl){      \
        vl = ESET(avl);                                                        \
        VT va = ELD(ap, vl), vb = ELD(bp, vl);                                 \
        EST(op_, VOP(va, vb, vl), vl);                                         \
    }                                                                          \
}
#define __BINGO_BINARY_F16(op, VOP)                                            \
    __BINGO_BINARY_IMPL(op, f16, _Float16, __riscv_vsetvl_e16m1,              \
        __riscv_vle16_v_f16m1, __riscv_vse16_v_f16m1, vfloat16m1_t, VOP)
#define __BINGO_BINARY_I8(op, VOP)                                             \
    __BINGO_BINARY_IMPL(op, i8, int8_t, __riscv_vsetvl_e8m1,                 \
        __riscv_vle8_v_i8m1, __riscv_vse8_v_i8m1, vint8m1_t, VOP)
#define __BINGO_BINARY_I16(op, VOP)                                            \
    __BINGO_BINARY_IMPL(op, i16, int16_t, __riscv_vsetvl_e16m1,              \
        __riscv_vle16_v_i16m1, __riscv_vse16_v_i16m1, vint16m1_t, VOP)
#define __BINGO_BINARY_I32(op, VOP)                                            \
    __BINGO_BINARY_IMPL(op, i32, int32_t, __riscv_vsetvl_e32m1,              \
        __riscv_vle32_v_i32m1, __riscv_vse32_v_i32m1, vint32m1_t, VOP)

#if BINGO_HAVE_FP16_VEC
#define __BINGO_BINARY_CASE_FP16(op)                                           \
    case BINGO_PREC_FP16:                                                      \
        __bingo_##op##_f16((const _Float16*)A[0], (const _Float16*)A[1],      \
                            (_Float16*)A[2], n); break;
#else
#define __BINGO_BINARY_CASE_FP16(op) case BINGO_PREC_FP16: ret=BINGO_RET_FAIL; break;
#endif
#define __BINGO_BINARY_CASE_INT_BOTH(op)                                       \
    case BINGO_PREC_INT8:                                                      \
        __bingo_##op##_i8((const int8_t*)A[0], (const int8_t*)A[1],          \
                            (int8_t*)A[2], n); break;                          \
    case BINGO_PREC_INT16:                                                     \
        __bingo_##op##_i16((const int16_t*)A[0], (const int16_t*)A[1],       \
                             (int16_t*)A[2], n); break;
// int8 + int16 + int32 (for ops with a native int32 vector impl)
#define __BINGO_BINARY_CASE_INT_ALL(op)                                        \
    __BINGO_BINARY_CASE_INT_BOTH(op)                                           \
    case BINGO_PREC_INT32:                                                     \
        __bingo_##op##_i32((const int32_t*)A[0], (const int32_t*)A[1],       \
                             (int32_t*)A[2], n); break;
#define __BINGO_BINARY_CASE_INT_NONE(op)

// Binary dispatcher: arg = {a, b, out, n, precision}.
#define __BINGO_BINARY_DISPATCH(op, INTCASES)                                  \
static inline uint64_t __host_bingo_kernel_##op(void *arg){                    \
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[4];                        \
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_##op##_f32(arg);    \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);                            \
    uint64_t n = A[3]; uint64_t ret = BINGO_RET_SUCC;                          \
    switch (prec) {                                                            \
        __BINGO_BINARY_CASE_FP16(op)                                           \
        INTCASES(op)                                                           \
        default: ret = BINGO_RET_FAIL;                                         \
    }                                                                          \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);                             \
    return ret;                                                                \
}

#if BINGO_HAVE_FP16_VEC
__BINGO_BINARY_F16(add, __riscv_vfadd_vv_f16m1)
__BINGO_BINARY_F16(sub, __riscv_vfsub_vv_f16m1)
__BINGO_BINARY_F16(mul, __riscv_vfmul_vv_f16m1)
__BINGO_BINARY_F16(div, __riscv_vfdiv_vv_f16m1)
__BINGO_BINARY_F16(max, __riscv_vfmax_vv_f16m1)
__BINGO_BINARY_F16(min, __riscv_vfmin_vv_f16m1)
#endif
__BINGO_BINARY_I8(add, __riscv_vadd_vv_i8m1)   __BINGO_BINARY_I16(add, __riscv_vadd_vv_i16m1)
__BINGO_BINARY_I8(sub, __riscv_vsub_vv_i8m1)   __BINGO_BINARY_I16(sub, __riscv_vsub_vv_i16m1)   __BINGO_BINARY_I32(sub, __riscv_vsub_vv_i32m1)
__BINGO_BINARY_I8(mul, __riscv_vmul_vv_i8m1)   __BINGO_BINARY_I16(mul, __riscv_vmul_vv_i16m1)   __BINGO_BINARY_I32(mul, __riscv_vmul_vv_i32m1)
__BINGO_BINARY_I8(max, __riscv_vmax_vv_i8m1)   __BINGO_BINARY_I16(max, __riscv_vmax_vv_i16m1)   __BINGO_BINARY_I32(max, __riscv_vmax_vv_i32m1)
__BINGO_BINARY_I8(min, __riscv_vmin_vv_i8m1)   __BINGO_BINARY_I16(min, __riscv_vmin_vv_i16m1)   __BINGO_BINARY_I32(min, __riscv_vmin_vv_i32m1)
// add keeps INT_BOTH: its int32 path is the distinct __host_bingo_kernel_add_i32
// kernel (K-split accumulation, writes scratchpad return-values). sub/mul/max/min
// take int32 through the dispatcher.
__BINGO_BINARY_DISPATCH(add, __BINGO_BINARY_CASE_INT_BOTH)
__BINGO_BINARY_DISPATCH(sub, __BINGO_BINARY_CASE_INT_ALL)
__BINGO_BINARY_DISPATCH(mul, __BINGO_BINARY_CASE_INT_ALL)
__BINGO_BINARY_DISPATCH(max, __BINGO_BINARY_CASE_INT_ALL)
__BINGO_BINARY_DISPATCH(min, __BINGO_BINARY_CASE_INT_ALL)
__BINGO_BINARY_DISPATCH(div, __BINGO_BINARY_CASE_INT_NONE)

// ---- typed native unary impls: out[i] = body(in[i]) ----
#define __BINGO_UNARY_IMPL(op, P, T, ESET, ELD, EST, VT, BODY)                 \
static inline void __bingo_##op##_##P(const T* in, T* o, uint64_t n){          \
    uint64_t avl = n; const T *ip=in; T *op_=o;                               \
    for (size_t vl = ESET(avl); avl>0; avl-=vl, ip+=vl, op_+=vl){              \
        vl = ESET(avl); VT v = ELD(ip, vl); VT result; BODY;                   \
        EST(op_, result, vl);                                                  \
    }                                                                          \
}
#define __BINGO_UNARY_F16_NATIVE(op, BODY)                                     \
    __BINGO_UNARY_IMPL(op, f16, _Float16, __riscv_vsetvl_e16m1,              \
        __riscv_vle16_v_f16m1, __riscv_vse16_v_f16m1, vfloat16m1_t, BODY)
#define __BINGO_UNARY_I8(op, BODY)                                             \
    __BINGO_UNARY_IMPL(op, i8, int8_t, __riscv_vsetvl_e8m1,                 \
        __riscv_vle8_v_i8m1, __riscv_vse8_v_i8m1, vint8m1_t, BODY)
#define __BINGO_UNARY_I16(op, BODY)                                            \
    __BINGO_UNARY_IMPL(op, i16, int16_t, __riscv_vsetvl_e16m1,              \
        __riscv_vle16_v_i16m1, __riscv_vse16_v_i16m1, vint16m1_t, BODY)
#define __BINGO_UNARY_I32(op, BODY)                                            \
    __BINGO_UNARY_IMPL(op, i32, int32_t, __riscv_vsetvl_e32m1,              \
        __riscv_vle32_v_i32m1, __riscv_vse32_v_i32m1, vint32m1_t, BODY)

// fp16 transcendental: widen f16->f32 (mf2->m1 keeps vl in range), reuse the
// fp32 body, narrow f32->f16. BODY32 reads vfloat32m1_t v -> sets result.
#if BINGO_HAVE_FP16_VEC
#define __BINGO_UNARY_F16_VIA_F32(op, BODY32)                                  \
static inline void __bingo_##op##_f16(const _Float16* in, _Float16* o, uint64_t n){ \
    uint64_t avl = n; const _Float16 *ip=in; _Float16 *op_=o;                  \
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl>0; avl-=vl, ip+=vl, op_+=vl){ \
        vl = __riscv_vsetvl_e16mf2(avl);                                       \
        vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(ip,vl), vl); \
        vfloat32m1_t result; BODY32;                                          \
        __riscv_vse16_v_f16mf2(op_, __bingo_narrow_f32m1_f16mf2(result, vl), vl); \
    }                                                                          \
}
#else
#define __BINGO_UNARY_F16_VIA_F32(op, BODY32)
#endif

#if BINGO_HAVE_FP16_VEC
#define __BINGO_UNARY_CASE_FP16(op)                                            \
    case BINGO_PREC_FP16:                                                      \
        __bingo_##op##_f16((const _Float16*)A[0], (_Float16*)A[1], n); break;
#else
#define __BINGO_UNARY_CASE_FP16(op) case BINGO_PREC_FP16: ret=BINGO_RET_FAIL; break;
#endif
#define __BINGO_UNARY_CASE_INT_BOTH(op)                                        \
    case BINGO_PREC_INT8:                                                      \
        __bingo_##op##_i8((const int8_t*)A[0], (int8_t*)A[1], n); break;     \
    case BINGO_PREC_INT16:                                                     \
        __bingo_##op##_i16((const int16_t*)A[0], (int16_t*)A[1], n); break;
// int8 + int16 + int32 (for ops with a native int32 vector impl)
#define __BINGO_UNARY_CASE_INT_ALL(op)                                         \
    __BINGO_UNARY_CASE_INT_BOTH(op)                                            \
    case BINGO_PREC_INT32:                                                     \
        __bingo_##op##_i32((const int32_t*)A[0], (int32_t*)A[1], n); break;
#define __BINGO_UNARY_CASE_INT_NONE(op)

// Unary dispatcher: arg = {in, out, n, precision}.
#define __BINGO_UNARY_DISPATCH(op, INTCASES)                                   \
static inline uint64_t __host_bingo_kernel_##op(void *arg){                    \
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[3];                        \
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_##op##_f32(arg);    \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);                            \
    uint64_t n = A[2]; uint64_t ret = BINGO_RET_SUCC;                          \
    switch (prec) {                                                            \
        __BINGO_UNARY_CASE_FP16(op)                                            \
        INTCASES(op)                                                           \
        default: ret = BINGO_RET_FAIL;                                         \
    }                                                                          \
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);                             \
    return ret;                                                                \
}

// int-capable unary ops (native fp16 + int8/int16).
#if BINGO_HAVE_FP16_VEC
__BINGO_UNARY_F16_NATIVE(relu, { result = __riscv_vfmax_vv_f16m1(v, __riscv_vfmv_v_f_f16m1((_Float16)0.0f, vl), vl); })
__BINGO_UNARY_F16_NATIVE(neg,  { result = __riscv_vfneg_v_f16m1(v, vl); })
__BINGO_UNARY_F16_NATIVE(abs,  { result = __riscv_vfmax_vv_f16m1(v, __riscv_vfneg_v_f16m1(v, vl), vl); })
#endif
__BINGO_UNARY_I8 (relu, { result = __riscv_vmax_vx_i8m1(v, 0, vl); })
__BINGO_UNARY_I16(relu, { result = __riscv_vmax_vx_i16m1(v, 0, vl); })
__BINGO_UNARY_I32(relu, { result = __riscv_vmax_vx_i32m1(v, 0, vl); })
__BINGO_UNARY_I8 (neg,  { result = __riscv_vneg_v_i8m1(v, vl); })
__BINGO_UNARY_I16(neg,  { result = __riscv_vneg_v_i16m1(v, vl); })
__BINGO_UNARY_I32(neg,  { result = __riscv_vneg_v_i32m1(v, vl); })
__BINGO_UNARY_I8 (abs,  { result = __riscv_vmax_vv_i8m1(v, __riscv_vneg_v_i8m1(v, vl), vl); })
__BINGO_UNARY_I16(abs,  { result = __riscv_vmax_vv_i16m1(v, __riscv_vneg_v_i16m1(v, vl), vl); })
__BINGO_UNARY_I32(abs,  { result = __riscv_vmax_vv_i32m1(v, __riscv_vneg_v_i32m1(v, vl), vl); })
__BINGO_UNARY_DISPATCH(relu, __BINGO_UNARY_CASE_INT_ALL)
__BINGO_UNARY_DISPATCH(neg,  __BINGO_UNARY_CASE_INT_ALL)
__BINGO_UNARY_DISPATCH(abs,  __BINGO_UNARY_CASE_INT_ALL)

// float-only unary ops (fp16 via widen/narrow, reusing the fp32 math).
__BINGO_UNARY_F16_VIA_F32(exp, { result = __bingo_exp_f32(v, vl); })
__BINGO_UNARY_F16_VIA_F32(sqrt, { result = __riscv_vfsqrt_v_f32m1(v, vl); })
__BINGO_UNARY_F16_VIA_F32(reciprocal, {
    result = __riscv_vfdiv_vv_f32m1(__riscv_vfmv_v_f_f32m1(1.0f, vl), v, vl);
})
__BINGO_UNARY_F16_VIA_F32(sigmoid, {
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t en  = __bingo_exp_f32(__riscv_vfneg_v_f32m1(v, vl), vl);
    result = __riscv_vfdiv_vv_f32m1(one, __riscv_vfadd_vv_f32m1(one, en, vl), vl);
})
__BINGO_UNARY_F16_VIA_F32(tanh, {
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t e2  = __bingo_exp_f32(__riscv_vfmul_vf_f32m1(v, 2.0f, vl), vl);
    result = __riscv_vfdiv_vv_f32m1(__riscv_vfsub_vv_f32m1(e2, one, vl),
                                    __riscv_vfadd_vv_f32m1(e2, one, vl), vl);
})
__BINGO_UNARY_F16_VIA_F32(silu, {
    vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t en  = __bingo_exp_f32(__riscv_vfneg_v_f32m1(v, vl), vl);
    vfloat32m1_t sig = __riscv_vfdiv_vv_f32m1(one, __riscv_vfadd_vv_f32m1(one, en, vl), vl);
    result = __riscv_vfmul_vv_f32m1(v, sig, vl);
})
__BINGO_UNARY_F16_VIA_F32(gelu, {
    vfloat32m1_t one    = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(v, 1.702f, vl);
    vfloat32m1_t en     = __bingo_exp_f32(__riscv_vfneg_v_f32m1(scaled, vl), vl);
    vfloat32m1_t sig    = __riscv_vfdiv_vv_f32m1(one, __riscv_vfadd_vv_f32m1(one, en, vl), vl);
    result = __riscv_vfmul_vv_f32m1(v, sig, vl);
})
__BINGO_UNARY_DISPATCH(exp,        __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(sqrt,       __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(reciprocal, __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(sigmoid,    __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(tanh,       __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(silu,       __BINGO_UNARY_CASE_INT_NONE)
__BINGO_UNARY_DISPATCH(gelu,       __BINGO_UNARY_CASE_INT_NONE)

// ---- reduction impls (scalar return; reduce_* dispatchers below) ----
#if BINGO_HAVE_FP16_VEC
static inline float __bingo_reduce_sum_f16(const _Float16* in, uint64_t n){
    float acc = 0.0f; uint64_t avl = n; const _Float16 *p = in;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e16mf2(avl);
        vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, vl), vl);
        vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        acc += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(v, z, vl));
    }
    return acc;
}
static inline float __bingo_reduce_max_f16(const _Float16* in, uint64_t n){
    float mx = (float)in[0]; uint64_t avl = n; const _Float16 *p = in;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e16mf2(avl);
        vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, vl), vl);
        vfloat32m1_t i = __riscv_vfmv_v_f_f32m1(mx, vl);
        mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(v, i, vl));
    }
    return mx;
}
#endif
// int reduce_sum: widen i8/i16 to i32 per chunk (overflow-safe at any VLEN).
static inline int32_t __bingo_reduce_sum_i8(const int8_t* in, uint64_t n){
    int32_t acc = 0; uint64_t avl = n; const int8_t *p = in;
    for (size_t vl = __riscv_vsetvl_e8mf4(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e8mf4(avl);
        vint16mf2_t v16 = __riscv_vwcvt_x_x_v_i16mf2(__riscv_vle8_v_i8mf4(p, vl), vl);
        vint32m1_t  v32 = __riscv_vwcvt_x_x_v_i32m1(v16, vl);
        vint32m1_t  z   = __riscv_vmv_v_x_i32m1(0, vl);
        acc += __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(v32, z, vl));
    }
    return acc;
}
static inline int32_t __bingo_reduce_sum_i16(const int16_t* in, uint64_t n){
    int32_t acc = 0; uint64_t avl = n; const int16_t *p = in;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e16mf2(avl);
        vint32m1_t v32 = __riscv_vwcvt_x_x_v_i32m1(__riscv_vle16_v_i16mf2(p, vl), vl);
        vint32m1_t z   = __riscv_vmv_v_x_i32m1(0, vl);
        acc += __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(v32, z, vl));
    }
    return acc;
}
static inline int32_t __bingo_reduce_max_i8(const int8_t* in, uint64_t n){
    int8_t mv = in[0]; uint64_t avl = n; const int8_t *p = in;
    for (size_t vl = __riscv_vsetvl_e8m1(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e8m1(avl);
        vint8m1_t v = __riscv_vle8_v_i8m1(p, vl);
        vint8m1_t i = __riscv_vmv_v_x_i8m1(mv, vl);
        mv = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmax_vs_i8m1_i8m1(v, i, vl));
    }
    return (int32_t)mv;
}
static inline int32_t __bingo_reduce_max_i16(const int16_t* in, uint64_t n){
    int16_t mv = in[0]; uint64_t avl = n; const int16_t *p = in;
    for (size_t vl = __riscv_vsetvl_e16m1(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e16m1(avl);
        vint16m1_t v = __riscv_vle16_v_i16m1(p, vl);
        vint16m1_t i = __riscv_vmv_v_x_i16m1(mv, vl);
        mv = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmax_vs_i16m1_i16m1(v, i, vl));
    }
    return (int32_t)mv;
}
// int32 reduce: int32 in -> int32 scalar out (same out type as the i8/i16 reduce).
// The sum accumulates in int32 and can overflow (same caveat as a chained int32 add).
static inline int32_t __bingo_reduce_sum_i32(const int32_t* in, uint64_t n){
    int32_t acc = 0; uint64_t avl = n; const int32_t *p = in;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t v = __riscv_vle32_v_i32m1(p, vl);
        vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
        acc += __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(v, z, vl));
    }
    return acc;
}
static inline int32_t __bingo_reduce_max_i32(const int32_t* in, uint64_t n){
    int32_t mv = in[0]; uint64_t avl = n; const int32_t *p = in;
    for (size_t vl = __riscv_vsetvl_e32m1(avl); avl>0; avl-=vl, p+=vl){
        vl = __riscv_vsetvl_e32m1(avl);
        vint32m1_t v = __riscv_vle32_v_i32m1(p, vl);
        vint32m1_t i = __riscv_vmv_v_x_i32m1(mv, vl);
        mv = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m1_i32m1(v, i, vl));
    }
    return mv;
}

// reduce dispatchers: arg = {in, out, n, precision}.
// FP32 -> float out (legacy). FP16 -> float out. INT8/INT16/INT32 -> int32 out.
static inline uint64_t __host_bingo_kernel_reduce_sum(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[3];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_reduce_sum_f32(arg);
    uint64_t n = A[2]; BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint64_t ret = BINGO_RET_SUCC;
    switch (prec) {
#if BINGO_HAVE_FP16_VEC
        case BINGO_PREC_FP16: *(float*)A[1]   = __bingo_reduce_sum_f16((const _Float16*)A[0], n); break;
#endif
        case BINGO_PREC_INT8:  *(int32_t*)A[1] = __bingo_reduce_sum_i8((const int8_t*)A[0], n); break;
        case BINGO_PREC_INT16: *(int32_t*)A[1] = __bingo_reduce_sum_i16((const int16_t*)A[0], n); break;
        case BINGO_PREC_INT32: *(int32_t*)A[1] = __bingo_reduce_sum_i32((const int32_t*)A[0], n); break;
        default: ret = BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return ret;
}
static inline uint64_t __host_bingo_kernel_reduce_max(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[3];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_reduce_max_f32(arg);
    uint64_t n = A[2]; BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint64_t ret = BINGO_RET_SUCC;
    switch (prec) {
#if BINGO_HAVE_FP16_VEC
        case BINGO_PREC_FP16: *(float*)A[1]   = __bingo_reduce_max_f16((const _Float16*)A[0], n); break;
#endif
        case BINGO_PREC_INT8:  *(int32_t*)A[1] = __bingo_reduce_max_i8((const int8_t*)A[0], n); break;
        case BINGO_PREC_INT16: *(int32_t*)A[1] = __bingo_reduce_max_i16((const int16_t*)A[0], n); break;
        case BINGO_PREC_INT32: *(int32_t*)A[1] = __bingo_reduce_max_i32((const int32_t*)A[0], n); break;
        default: ret = BINGO_RET_FAIL;
    }
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return ret;
}
static inline uint64_t __host_bingo_kernel_reduce_mean(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[3];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_reduce_mean_f32(arg);
    uint64_t n = A[2]; BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
    uint64_t ret = BINGO_RET_SUCC;
#if BINGO_HAVE_FP16_VEC
    if (prec == BINGO_PREC_FP16) *(float*)A[1] = __bingo_reduce_sum_f16((const _Float16*)A[0], n) / (float)n;
    else ret = BINGO_RET_FAIL;
#else
    ret = BINGO_RET_FAIL;
#endif
    BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
    return ret;
}

// ---- compound fp16 ops (widen f16->f32, compute, narrow f32->f16) ----
#if BINGO_HAVE_FP16_VEC
static inline void __bingo_silu_mul_f16(const _Float16* g, const _Float16* u,
                                         _Float16* o, uint64_t n){
    uint64_t avl = n; const _Float16 *gp=g,*up=u; _Float16 *op_=o;
    for (size_t vl = __riscv_vsetvl_e16mf2(avl); avl>0; avl-=vl, gp+=vl, up+=vl, op_+=vl){
        vl = __riscv_vsetvl_e16mf2(avl);
        vfloat32m1_t gv = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(gp, vl), vl);
        vfloat32m1_t uv = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(up, vl), vl);
        vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
        vfloat32m1_t en  = __bingo_exp_f32(__riscv_vfneg_v_f32m1(gv, vl), vl);
        vfloat32m1_t sig = __riscv_vfdiv_vv_f32m1(one, __riscv_vfadd_vv_f32m1(one, en, vl), vl);
        vfloat32m1_t r   = __riscv_vfmul_vv_f32m1(__riscv_vfmul_vv_f32m1(gv, sig, vl), uv, vl);
        __riscv_vse16_v_f16mf2(op_, __bingo_narrow_f32m1_f16mf2(r, vl), vl);
    }
}
static inline void __bingo_softmax_row_f16(const _Float16* in, _Float16* out, uint64_t len){
    float maxv = (float)in[0];
    { uint64_t rem = len; const _Float16 *p = in;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, p+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, vl), vl);
          vfloat32m1_t i = __riscv_vfmv_v_f_f32m1(maxv, vl);
          maxv = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(v, i, vl)); } }
    float sum = 0.0f;
    { uint64_t rem = len; const _Float16 *p = in; _Float16 *op_ = out;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, p+=vl, op_+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v  = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, vl), vl);
          vfloat32m1_t ev = __bingo_exp_f32(__riscv_vfsub_vf_f32m1(v, maxv, vl), vl);
          __riscv_vse16_v_f16mf2(op_, __bingo_narrow_f32m1_f16mf2(ev, vl), vl);
          vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, vl);
          sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(ev, z, vl)); } }
    float inv = 1.0f / sum;
    { uint64_t rem = len; _Float16 *op_ = out;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, op_+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(op_, vl), vl);
          vfloat32m1_t r = __riscv_vfmul_vf_f32m1(v, inv, vl);
          __riscv_vse16_v_f16mf2(op_, __bingo_narrow_f32m1_f16mf2(r, vl), vl); } }
}
static inline void __bingo_rmsnorm_row_f16(const _Float16* in, const _Float16* w,
                                            _Float16* out, uint64_t hidden){
    float ss = 0.0f;
    { uint64_t rem = hidden; const _Float16 *p = in;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, p+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v  = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(p, vl), vl);
          vfloat32m1_t sq = __riscv_vfmul_vv_f32m1(v, v, vl);
          vfloat32m1_t z  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
          ss += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m1_f32m1(sq, z, vl)); } }
    float rms = 1.0f / __builtin_sqrtf(ss / (float)hidden + 1e-6f);
    { uint64_t rem = hidden; const _Float16 *ip=in,*wp=w; _Float16 *op_=out;
      for (size_t vl = __riscv_vsetvl_e16mf2(rem); rem>0; rem-=vl, ip+=vl, wp+=vl, op_+=vl){
          vl = __riscv_vsetvl_e16mf2(rem);
          vfloat32m1_t v  = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(ip, vl), vl);
          vfloat32m1_t wv = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2(wp, vl), vl);
          vfloat32m1_t r  = __riscv_vfmul_vv_f32m1(__riscv_vfmul_vf_f32m1(v, rms, vl), wv, vl);
          __riscv_vse16_v_f16mf2(op_, __bingo_narrow_f32m1_f16mf2(r, vl), vl); } }
}
#endif // BINGO_HAVE_FP16_VEC

// compound dispatchers.
// silu_mul: arg = {gate, up, out, n, precision}.
static inline uint64_t __host_bingo_kernel_silu_mul(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[4];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_silu_mul_f32(arg);
#if BINGO_HAVE_FP16_VEC
    if (prec == BINGO_PREC_FP16) {
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        __bingo_silu_mul_f16((const _Float16*)A[0], (const _Float16*)A[1], (_Float16*)A[2], A[3]);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        return BINGO_RET_SUCC;
    }
#endif
    return BINGO_RET_FAIL;
}
// softmax: arg = {in, out, num_rows, row_length, precision, scratchpad}.
static inline uint64_t __host_bingo_kernel_softmax(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[4];
    if (prec == BINGO_PREC_FP32) {
        uint64_t a2[5] = { A[0], A[1], A[2], A[3], A[5] };  // drop precision, keep scratchpad last
        return __host_bingo_kernel_softmax_f32(a2);
    }
#if BINGO_HAVE_FP16_VEC
    if (prec == BINGO_PREC_FP16) {
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        uint64_t rows = A[2], len = A[3];
        for (uint64_t r = 0; r < rows; r++)
            __bingo_softmax_row_f16((const _Float16*)A[0] + r*len, (_Float16*)A[1] + r*len, len);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        return BINGO_RET_SUCC;
    }
#endif
    return BINGO_RET_FAIL;
}
// rmsnorm: arg = {in, weight, out, hidden_dim, num_tokens, precision}.
static inline uint64_t __host_bingo_kernel_rmsnorm(void *arg){
    uint64_t *A = (uint64_t*)arg; uint64_t prec = A[5];
    if (prec == BINGO_PREC_FP32) return __host_bingo_kernel_rmsnorm_f32(arg);
#if BINGO_HAVE_FP16_VEC
    if (prec == BINGO_PREC_FP16) {
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_START);
        uint64_t hidden = A[3], tokens = A[4];
        for (uint64_t t = 0; t < tokens; t++)
            __bingo_rmsnorm_row_f16((const _Float16*)A[0] + t*hidden,
                                     (const _Float16*)A[1], (_Float16*)A[2] + t*hidden, hidden);
        BINGO_TRACE_MARKER(BINGO_TRACE_SIMD_RUN_END);
        return BINGO_RET_SUCC;
    }
#endif
    return BINGO_RET_FAIL;
}
#endif // disabled host RVV/Ara kernels
