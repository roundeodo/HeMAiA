// Compact scratchpad schema for end-of-DFG MoE runtime timing.
#pragma once

#include <stdint.h>

#ifndef MOE_RUNTIME_TIMING
#define MOE_RUNTIME_TIMING 0
#endif

#define MOE_RUNTIME_TIMING_MAGIC 0x4d4f4550u

enum bingo_moe_runtime_timing_stage {
    MOE_PROFILE_STAGE_GATHER_S1 = 1,
    MOE_PROFILE_STAGE_LOAD_S1,
    MOE_PROFILE_STAGE_COMPUTE_S1,
    MOE_PROFILE_STAGE_PREFETCH_S2,
    MOE_PROFILE_STAGE_COMPUTE_S2,
    MOE_PROFILE_STAGE_LOAD_S3,
    MOE_PROFILE_STAGE_COMPUTE_S3,
    MOE_PROFILE_STAGE_PREFETCH_S4,
    MOE_PROFILE_STAGE_COMPUTE_S4,
    MOE_PROFILE_STAGE_STORE,
    MOE_PROFILE_STAGE_CONFIG_S1,
    MOE_PROFILE_STAGE_CONFIG_S3,
    MOE_PROFILE_STAGE_CLUSTER_BEGIN,
    MOE_PROFILE_STAGE_CLUSTER_END,
    MOE_PROFILE_STAGE_LOAD_S3_PREFETCH_S4,
};

enum bingo_moe_runtime_timing_resource {
    MOE_PROFILE_RESOURCE_NONE = 0,
    MOE_PROFILE_RESOURCE_IDMA,
    MOE_PROFILE_RESOURCE_XDMA,
    MOE_PROFILE_RESOURCE_DMA_BOTH,
    MOE_PROFILE_RESOURCE_VERSACORE,
    MOE_PROFILE_RESOURCE_CONFIG,
};

enum bingo_moe_runtime_timing_flags {
    MOE_PROFILE_FLAG_ACTIVE = 1u << 0,
    MOE_PROFILE_FLAG_SKIPPED = 1u << 1,
    MOE_PROFILE_FLAG_CTRL_SKIP = 1u << 2,
    MOE_PROFILE_FLAG_INVALID_CALL = 1u << 3,
    MOE_PROFILE_FLAG_NO_PREFETCH = 1u << 4,
    MOE_PROFILE_FLAG_NO_STORE = 1u << 5,
};

// bingo_kernel_scratchpad_t::reserved[] layout.
#define MOE_SP_PROFILE_MAGIC_IDX          0u
#define MOE_SP_PROFILE_META_IDX           1u
#define MOE_SP_PROFILE_TASK_IDX           2u
#define MOE_SP_PROFILE_RESOURCE_START_IDX 3u
#define MOE_SP_PROFILE_RESOURCE_END_IDX   4u
#define MOE_SP_PROFILE_PEER_WAIT_IDX      5u
// Resource-dependent completed work: DMA bytes or VersaCore hardware-busy cycles.
#define MOE_SP_PROFILE_UNITS_IDX          6u
#define MOE_SP_PROFILE_RESULT_IDX         7u
#define MOE_SP_PROFILE_FLAGS_IDX          8u

// meta: stage[7:0], resource[11:8], cluster[15:12], core[19:16], block[31:20]
#define MOE_PROFILE_META(stage, resource, cluster, core, block) \
    (((uint32_t)(stage) & 0xffu) | (((uint32_t)(resource) & 0x0fu) << 8u) | \
     (((uint32_t)(cluster) & 0x0fu) << 12u) | \
     (((uint32_t)(core) & 0x0fu) << 16u) | \
     (((uint32_t)(block) & 0x0fffu) << 20u))

#define MOE_PROFILE_META_STAGE(v)    ((uint32_t)(v) & 0xffu)
#define MOE_PROFILE_META_RESOURCE(v) (((uint32_t)(v) >> 8u) & 0x0fu)
#define MOE_PROFILE_META_CLUSTER(v)  (((uint32_t)(v) >> 12u) & 0x0fu)
#define MOE_PROFILE_META_CORE(v)     (((uint32_t)(v) >> 16u) & 0x0fu)
#define MOE_PROFILE_META_BLOCK(v)    (((uint32_t)(v) >> 20u) & 0x0fffu)

// task: slot[5:0], expert[13:6], ntokens[23:14]
#define MOE_PROFILE_TASK(slot, expert, ntokens) \
    (((uint32_t)(slot) & 0x3fu) | (((uint32_t)(expert) & 0xffu) << 6u) | \
     (((uint32_t)(ntokens) & 0x03ffu) << 14u))

#define MOE_PROFILE_TASK_SLOT(v)   ((uint32_t)(v) & 0x3fu)
#define MOE_PROFILE_TASK_EXPERT(v) (((uint32_t)(v) >> 6u) & 0xffu)
#define MOE_PROFILE_TASK_NTOK(v)   (((uint32_t)(v) >> 14u) & 0x03ffu)
