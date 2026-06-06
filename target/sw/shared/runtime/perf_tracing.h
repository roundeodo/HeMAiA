#pragma once

// ============================================================================
// Performance Tracing for Bingo
// ============================================================================
// This mechanism uses "Magic NOPs" to inject markers into the instruction trace
// without affecting the architectural state of the processor.
//
// The markers are implemented as 'xori x0, x0, IMM' instructions.
// - They execute as a valid NOP (writing to zero register).
// - They carry a 12-bit immediate payload (IMM) visible in the trace dump.
//
// These markers are parsed by post-processing scripts to generate timelines
// (e.g., for Perfetto).

// Enable/Disable Tracing
// Define BINGO_PERF_TRACING before including this header or via compiler flags.
#ifdef BINGO_PERF_TRACING

// The Magic NOP Macro
// Uses the immediate value %0 (limited to 12 bits: 0-4095)
// We use xori instead of ori to avoid spike-dasm decoding it as prefetch
#define BINGO_TRACE_MARKER(id) asm volatile("xori x0, x0, %0" :: "i"(id))

#else

// If tracing is disabled, these compile to nothing.
#define BINGO_TRACE_MARKER(id) ((void)0)

#endif

// ============================================================================
// Trace Marker IDs (Events)
// ============================================================================
// Hierarchical ID scheme (max 12 bits):
// 0x0xx: BINGO SW Manager Events 
// 0x1XX: BINGO HW Manager Events
// 0x2XX: Kernel Configuration Events
// 0x3XX: Accelerator Execution Events


// // --- BINGO SW Manager Events ---
// Marks the lifespan of a task within the SW Manager loop
#define BINGO_TRACE_SW_MGR_INIT_TASK_QUEUE_START     0x010 // Start processing
#define BINGO_TRACE_SW_MGR_INIT_TASK_QUEUE_END       0x011 // End processing
#define BINGO_TRACE_SW_MGR_ENQUEUE_LOCAL_READY_TASKS_START   0x012
#define BINGO_TRACE_SW_MGR_ENQUEUE_LOCAL_READY_TASKS_END     0x013
#define BINGO_TRACE_SW_MGR_ENQUEUE_REMOTE_READY_TASKS_START  0x014
#define BINGO_TRACE_SW_MGR_ENQUEUE_REMOTE_READY_TASKS_END    0x015
#define BINGO_TRACE_SW_MGR_SCHED_READY_TASKS_START          0x016
#define BINGO_TRACE_SW_MGR_SCHED_READY_TASKS_END  0x017

// --- Hardware Manager Events ---
// Marks the lifespan of a task within the HW Manager loop
#define BINGO_TRACE_MGR_GET_READY_START        0x110 // Start reading Ready Queue
#define BINGO_TRACE_MGR_GET_READY_END          0x111 // End reading Ready Queue
#define BINGO_TRACE_MGR_PREP_START             0x112 // Start preparing kernel (get args, ptrs)
#define BINGO_TRACE_MGR_PREP_END               0x113 // End preparing kernel
#define BINGO_TRACE_MGR_RUN_KERNEL_START       0x114 // Start running kernel
#define BINGO_TRACE_MGR_RUN_KERNEL_END         0x115 // End running kernel
#define BINGO_TRACE_MGR_WRITE_DONE_START       0x116 // Start writing Done Queue
#define BINGO_TRACE_MGR_WRITE_DONE_END         0x117 // End writing Done Queue
#define BINGO_TRACE_KERNEL_ARG_PARSE_START     0x118 // Parsing kernel arguments
#define BINGO_TRACE_KERNEL_ARG_PARSE_END       0x119 // Finished parsing kernel arguments

// --- Kernel Internal Phases ---
// These are used inside individual kernels
// Non-computation kernels (Dummy, Exit)
#define BINGO_TRACE_DUMMY_KERNEL_START  0x200
#define BINGO_TRACE_DUMMY_KERNEL_END    0x201

// Computation Kernels: Configuration Phase
// IDMA
#define BINGO_TRACE_IDMA_CFG_START         0x210
#define BINGO_TRACE_IDMA_CFG_END           0x211
// XDMA
#define BINGO_TRACE_XDMA_CFG_START         0x220
#define BINGO_TRACE_XDMA_CFG_END           0x221
// GEMM FULL
#define BINGO_TRACE_GEMM_FULL_CFG_START    0x230
#define BINGO_TRACE_GEMM_FULL_CFG_END      0x231
// Minimal GEMM
#define BINGO_TRACE_GEMM_MIN_CFG_START     0x240
#define BINGO_TRACE_GEMM_MIN_CFG_END       0x241
// SIMD
#define BINGO_TRACE_SIMD_CFG_START         0x250
#define BINGO_TRACE_SIMD_CFG_END           0x251
// HOST IDMA
#define BINGO_TRACE_HOST_IDMA_CFG_START    0x260
#define BINGO_TRACE_HOST_IDMA_CFG_END      0x261
// HOST MoE legacy helpers
#define BINGO_TRACE_HOST_ACCUMULATE_START   0x270
#define BINGO_TRACE_HOST_ACCUMULATE_END     0x271
#define BINGO_TRACE_HOST_SCATTER_PAD_START  0x280
#define BINGO_TRACE_HOST_SCATTER_PAD_END    0x281
#define BINGO_TRACE_HOST_SCATTER_META_START 0x290
#define BINGO_TRACE_HOST_SCATTER_META_END   0x291
#define BINGO_TRACE_HOST_SOFTMAX_START      0x2A0
#define BINGO_TRACE_HOST_SOFTMAX_END        0x2A1
#define BINGO_TRACE_HOST_SWISH_START        0x2B0
#define BINGO_TRACE_HOST_SWISH_END          0x2B1
#define BINGO_TRACE_HOST_GLU_START          0x2C0
#define BINGO_TRACE_HOST_GLU_END            0x2C1
// HOST ROUTER SCHED
#define BINGO_TRACE_HOST_ROUTER_SCHED_START 0x2D0
#define BINGO_TRACE_HOST_ROUTER_SCHED_END   0x2D1
// DUAL DMA (iDMA + xDMA concurrent) — outer wrapper for the entire kernel
#define BINGO_TRACE_DUAL_DMA_CFG_START      0x2E0
#define BINGO_TRACE_DUAL_DMA_CFG_END        0x2E1
// HOST MoE Phase 3: prepare_request (token counting + moe_schedule) + Phase 4: args lowering
// BINGO_TRACE_HOST_MOE_PREPARE      wraps entire __host_bingo_kernel_moe_prepare_request body
// BINGO_TRACE_HOST_MOE_TOKEN_COUNT  wraps zero-init + count + offset build + token_ids scatter
// BINGO_TRACE_HOST_MOE_REQUEST_BUILD wraps CAM state update + request->experts fill
// BINGO_TRACE_HOST_MOE_SCHED        wraps only the moe_schedule() call inside Phase 3
// BINGO_TRACE_HOST_MOE_EXECUTE      wraps entire __host_bingo_kernel_moe_execute body (Phase 4 args lowering)
#define BINGO_TRACE_HOST_MOE_PREPARE_START       0x2F0
#define BINGO_TRACE_HOST_MOE_PREPARE_END         0x2F1
#define BINGO_TRACE_HOST_MOE_SCHED_START         0x2F2
#define BINGO_TRACE_HOST_MOE_SCHED_END           0x2F3
#define BINGO_TRACE_HOST_MOE_EXECUTE_START       0x2F4
#define BINGO_TRACE_HOST_MOE_EXECUTE_END         0x2F5
#define BINGO_TRACE_HOST_MOE_TOKEN_COUNT_START   0x2F6
#define BINGO_TRACE_HOST_MOE_TOKEN_COUNT_END     0x2F7
#define BINGO_TRACE_HOST_MOE_REQUEST_BUILD_START 0x2F8
#define BINGO_TRACE_HOST_MOE_REQUEST_BUILD_END   0x2F9

// Computation Kernels: Compute/Run Phase
// IDMA
#define BINGO_TRACE_IDMA_RUN_START        0x310
#define BINGO_TRACE_IDMA_RUN_END          0x311
// XDMA
#define BINGO_TRACE_XDMA_RUN_START        0x320
#define BINGO_TRACE_XDMA_RUN_END          0x321
// GEMM FULL
#define BINGO_TRACE_GEMM_FULL_RUN_START   0x330
#define BINGO_TRACE_GEMM_FULL_RUN_END     0x331
// Minimal GEMM
#define BINGO_TRACE_GEMM_MIN_RUN_START    0x340
#define BINGO_TRACE_GEMM_MIN_RUN_END      0x341
// SIMD
#define BINGO_TRACE_SIMD_RUN_START        0x350
#define BINGO_TRACE_SIMD_RUN_END          0x351
// HOST IDMA
#define BINGO_TRACE_HOST_IDMA_RUN_START   0x360
#define BINGO_TRACE_HOST_IDMA_RUN_END     0x361
// L15 MoE device kernels (full, swiglu-only, down-only)
// _full: CFG wraps Mode-0 CSR configuration; CFG1 wraps Mode-1 CSR configuration;
//        MODE0/MODE1 wrap each hardware VersaCore execution phase
#define BINGO_TRACE_L15_FULL_CFG_START    0x370
#define BINGO_TRACE_L15_FULL_CFG_END      0x371
#define BINGO_TRACE_L15_FULL_MODE0_START  0x372
#define BINGO_TRACE_L15_FULL_MODE0_END    0x373
#define BINGO_TRACE_L15_FULL_MODE1_START  0x374
#define BINGO_TRACE_L15_FULL_MODE1_END    0x375
#define BINGO_TRACE_L15_FULL_ZEROFILL_START 0x37E  // Mode-1 output zero-fill
#define BINGO_TRACE_L15_FULL_ZEROFILL_END   0x37F
#define BINGO_TRACE_L15_FULL_CFG1_START   0x380  // Mode-1 CSR configuration
#define BINGO_TRACE_L15_FULL_CFG1_END     0x381
#define BINGO_TRACE_L15_SWIGLU_CFG_START  0x376
#define BINGO_TRACE_L15_SWIGLU_CFG_END    0x377
#define BINGO_TRACE_L15_SWIGLU_RUN_START  0x378
#define BINGO_TRACE_L15_SWIGLU_RUN_END    0x379
#define BINGO_TRACE_L15_DOWN_CFG_START    0x37A
#define BINGO_TRACE_L15_DOWN_CFG_END      0x37B
#define BINGO_TRACE_L15_DOWN_RUN_START    0x37C
#define BINGO_TRACE_L15_DOWN_RUN_END      0x37D

// Device MoE dynamic expert per-kernel identification markers (0x382 - 0x395)
// These are emitted inside each __snax_bingo_kernel_moe_dynamic_expert_*
// function to allow classify_kernel to distinguish DMA kernel types.
#define BINGO_TRACE_DEV_MOE_GATHER_S1_START             0x382
#define BINGO_TRACE_DEV_MOE_GATHER_S1_END               0x383
#define BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_START          0x384
#define BINGO_TRACE_DEV_MOE_LOAD_GATE_UP_END            0x385
#define BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_START       0x386
#define BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_END         0x387
#define BINGO_TRACE_DEV_MOE_LOAD_DOWN_START             0x388
#define BINGO_TRACE_DEV_MOE_LOAD_DOWN_END               0x389
#define BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_START          0x38A
#define BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_END            0x38B
#define BINGO_TRACE_DEV_MOE_PREFETCH_S2_START           0x38C
#define BINGO_TRACE_DEV_MOE_PREFETCH_S2_END             0x38D
#define BINGO_TRACE_DEV_MOE_PREFETCH_S4_START           0x38E
#define BINGO_TRACE_DEV_MOE_PREFETCH_S4_END             0x38F
#define BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_START  0x390
#define BINGO_TRACE_DEV_MOE_COMPUTE_GATE_UP_FULL_END    0x391
#define BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_START     0x392
#define BINGO_TRACE_DEV_MOE_COMPUTE_DOWN_FULL_END       0x393
#define BINGO_TRACE_DEV_MOE_STORE_START                 0x394
#define BINGO_TRACE_DEV_MOE_STORE_END                   0x395

// DMA 子阶段计时 marker (0x396 - 0x39B)
// 由 __moe_dyn_xdma_start_copy / __moe_dyn_wait_xdma / copy_one / copy_pair 内部发出。
// 这些是 gather_s1 / load_gate_up / load_down / prefetch_s2/s4 / store 的子事件。
//
//   xDMA_CFG  = xdma_memcpy_1d_fast_full_addr() 的 30-CSR-write 阶段
//   xDMA_WAIT = xdma_wait_task() 等待 xDMA 传输完成的阶段
//   iDMA_WAIT = snrt_dma_wait_all() 等待 iDMA 传输完成的阶段
#define BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_START          0x396
#define BINGO_TRACE_DEV_MOE_DMA_XDMA_CFG_END            0x397
#define BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_START         0x398
#define BINGO_TRACE_DEV_MOE_DMA_XDMA_WAIT_END           0x399
#define BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_START         0x39A
#define BINGO_TRACE_DEV_MOE_DMA_IDMA_WAIT_END           0x39B
