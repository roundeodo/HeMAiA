#pragma once

// Pure-HW scheduler state carried while compact RTL plans are drained directly
// into the per-cluster L3 slot records. This header has no SW schedule ABI,
// validation path, or fallback path.

#if !MOE_SCHED_PURE_HW_PATH
#error "host_moe_hw_fast_path.h is only valid in a pure-HW scheduler build"
#endif

typedef struct {
    uint32_t slot_bytes;
    uint32_t slot_count[2];
    uintptr_t stage_base[2];
    uintptr_t next_arg[2];
    uint32_t l1_a_addr[2];
    uint32_t l1_d_addr[2];
    uint32_t l1_down_d_addr[2];
    int32_t final_cam[2];
} __host_moe_direct_lower_ctx_t;
