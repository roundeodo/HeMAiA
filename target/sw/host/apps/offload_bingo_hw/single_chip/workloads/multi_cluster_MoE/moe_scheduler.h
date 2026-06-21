/* moe_scheduler.h
 * --------------------------------------------------------------------------
 * Analytical greedy scheduler for HeMAiA two-cluster MoE workload.
 * Pure host-side (CVA6) function. No FP, no malloc, no OS deps.
 *
 * Pipeline-stage cycle constants (per 1 token, on a 1-cluster x 2-versacore
 * compute fabric @ 512 MAC/cc) are hard-coded in moe_scheduler.c and MUST
 * be kept in sync with Idea_Model/four_stage_scheduler.py.
 * --------------------------------------------------------------------------
 */
#ifndef MOE_SCHEDULER_H
#define MOE_SCHEDULER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── compile-time limits (64-expert MoE, HeMAiA 2-cluster) ─────────────── */
#ifndef MOE_MAX_EXPERTS
#define MOE_MAX_EXPERTS    64    /* compile-time max experts per request (override via -D) */
#endif
#define MOE_MAX_TASKS     (MOE_MAX_EXPERTS * 2)   /* SPLIT doubles task count at most   */
#define MOE_MAX_DMA_OPS  (MOE_MAX_TASKS * 4)      /* S1+S3+S2PF+S4PF per task max       */
#define MOE_MAX_SLOTS_PER_CLUSTER  (MOE_MAX_EXPERTS) /* max slots per cluster side   */

/* DMA slot kind indices (used by __host_moe_dma_slot_index) */
#define MOE_TASK_DMA_SLOT_S1          0u
#define MOE_TASK_DMA_SLOT_S3          1u
#define MOE_TASK_DMA_SLOT_S2_PREFETCH 2u
#define MOE_TASK_DMA_SLOT_S4_PREFETCH 3u
#define MOE_TASK_DMA_SLOTS            4u

/* ── Shape enum (mirrors ShapeA/B/C in four_stage_scheduler.py) ───────── */
typedef enum {
    MOE_SHAPE_A = 0,   /* M_dim=8, alloc=64,  bw_req=32   */
    MOE_SHAPE_B = 1,   /* M_dim=4, alloc=64,  bw_req=64   */
    MOE_SHAPE_C = 2    /* M_dim=2, alloc=128, bw_req=128  */
} moe_shape_t;

/* ── Cluster id (compile-time mapping) ───────────────────────────────── */
typedef enum {
    MOE_CLUSTER_C2 = 0,
    MOE_CLUSTER_C3 = 1
} moe_cluster_t;

/* ── DMA lane binding (each lane provides 64 B/cc) ───────────────────── */
typedef enum {
    MOE_DMA_NONE = 0,
    MOE_DMA_IDMA = 1,
    MOE_DMA_XDMA = 2,
    MOE_DMA_BOTH = 3
} moe_dma_binding_t;

typedef enum {
    MOE_DMA_OP_S1 = 1,         /* foreground gate/up weight DMA         */
    MOE_DMA_OP_S3 = 3,         /* foreground down weight DMA            */
    MOE_DMA_OP_S2_PREFETCH = 4,/* down weight prefetched during S1/S2   */
    MOE_DMA_OP_S4_PREFETCH = 5 /* next gate/up prefetched during S3/S4  */
} moe_dma_op_kind_t;

/* ── Input: per-expert token count after routing ──────────────────────── */
typedef struct {
    uint16_t expert_id;       /* 0..N_EXPERTS-1                          */
    uint16_t ntokens;         /* number of tokens routed to this expert  */
} moe_expert_load_t;

typedef struct {
    moe_expert_load_t experts[MOE_MAX_EXPERTS];
    uint16_t  n_experts;      /* number of valid entries (<=MOE_MAX_EXPERTS) */
    int16_t   cache_eid_c2;   /* expert resident in C2 weight SRAM, -1 = none */
    int16_t   cache_eid_c3;   /* expert resident in C3 weight SRAM, -1 = none */
} moe_request_t;

/* ── Output: one scheduled expert run (compact vs old 88-byte record) ───── */
typedef struct {
    moe_cluster_t     cluster;          /* MOE_CLUSTER_C2=0, MOE_CLUSTER_C3=1        */
    uint16_t          expert_id;        /* 0..MOE_MAX_EXPERTS-1                      */
    uint16_t          token_start_rank; /* rank in expert's routed-token list        */
    uint16_t          ntokens;          /* token slice (after SPLIT)                 */
    moe_shape_t       shape_s1;         /* gate/up weight DMA shape                  */
    moe_shape_t       shape_s3;         /* down weight DMA shape                     */
    moe_dma_binding_t dma_s1;          /* iDMA / xDMA / BOTH / NONE for S1          */
    moe_dma_binding_t dma_s3;          /* iDMA / xDMA / BOTH / NONE for S3          */
    uint8_t           skip_s1;          /* 1 = S1 cache hit, skip gate/up DMA+GEMM  */
    uint8_t           skip_s3;          /* 1 = S3 cache hit, skip down DMA+GEMM     */
    uint8_t           skip_s2;          /* 1 = S2 tail empty (ntokens<=shape_Mdim)  */
    uint8_t           skip_s4;          /* 1 = S4 tail empty                        */
    uint32_t          m_s2_exec;        /* S2 M-tile count; fixed shape C, 2 tok/tile */
    uint32_t          m_s4_exec;        /* S4 M-tile count; fixed shape C, 2 tok/tile */
    /* REMOVED: bw_s1, bw_s3 (derivable from shape+skip), prefetch_eid (from dma_ops),
     *          dma_slots[] (from dma_ops), est_start_cc/est_end_cc (timing removed) */
} moe_task_t;

/* ── DMA operation record (compact, 12 bytes vs old 34 bytes) ────────── */
typedef struct {
    uint16_t          task_idx;  /* index into tasks[]                        */
    moe_dma_op_kind_t kind;      /* S1/S3/S2_PREFETCH/S4_PREFETCH             */
    moe_dma_binding_t dma;       /* iDMA/xDMA/BOTH lane assignment            */
    int16_t           expert_id; /* expert whose weight is transferred        */
    /* REMOVED: cluster (from tasks[task_idx]), weight/shape/alloc_bw/start_cc/end_cc */
} moe_dma_op_t;

typedef struct {
    moe_task_t   tasks[MOE_MAX_TASKS];      /* ordered task list               */
    moe_dma_op_t dma_ops[MOE_MAX_DMA_OPS];  /* DMA operations in issue order   */
    uint16_t     n_tasks;                   /* valid tasks count               */
    uint16_t     n_dma_ops;                 /* valid DMA ops count             */
    /* REMOVED: est_makespan_cc (timing no longer propagated to L3)            */
} moe_schedule_t;

/* ── RTL compact plan format -------------------------------------------------
 * This mirrors the pure-slave scheduler RTL output after commit_unit:
 * one entry is one actual cluster task.  Timing/snap fields are intentionally
 * absent; S4 prefetch legality is carried by allow_s4pf.
 */
typedef struct {
    moe_cluster_t cluster;
    uint16_t      expert_id;
    uint16_t      token_start_rank;
    uint16_t      ntokens;
    moe_shape_t   shape_s1;
    moe_shape_t   shape_s3;
    uint8_t       skip_s1;
    uint8_t       skip_s3;
    uint8_t       has_s2pf;
} moe_hw_plan_desc_t;

typedef struct {
    uint8_t            valid;
    moe_hw_plan_desc_t desc;
    uint8_t            allow_s4pf;
} moe_hw_plan_entry_t;

/* ── Return codes ─────────────────────────────────────────────────────── */
typedef enum {
    MOE_OK              =  0,
    MOE_ERR_BAD_INPUT   = -1,  /* n_experts=0 or >MAX, or ntokens=0      */
    MOE_ERR_OVERFLOW    = -2,  /* generated more than MOE_MAX_TASKS      */
    MOE_ERR_INTERNAL    = -3   /* unreachable branch, indicates bug      */
} moe_status_t;

/* ─────────────────────────────────────────────────────────────────────────
 *  moe_schedule
 *  ------------
 *  Pure function: given the per-expert token distribution and current cache
 *  residency, fills `out` with an ordered list of cluster tasks.
 *
 *  Caller workflow on host:
 *      moe_request_t req = { ... };           // fill from router output
 *      moe_schedule_t sch;
 *      if (moe_schedule(&req, &sch) != MOE_OK) { handle_error(); }
 *      for (int i = 0; i < sch.n_tasks; i++) {
 *          dispatch_to_cluster(&sch.tasks[i]); // mailbox / SPM write
 *      }
 *
 *  The function is deterministic and reentrant. Runtime depends on how many
 *  candidate branches are enumerated for the current request. No dynamic
 *  memory is used.
 * ───────────────────────────────────────────────────────────────────────── */
moe_status_t moe_schedule(const moe_request_t *req, moe_schedule_t *out);

/* Generate the same compact plan representation that the RTL scheduler emits.
 * This is intended as the software golden model for RTL schedule_core tests.
 */
moe_status_t moe_make_hw_plan(const moe_request_t *req,
                              moe_hw_plan_entry_t *plan,
                              uint16_t *n_plan);

/* Lower an RTL/software compact plan into the API-facing task/DMA schedule.
 * CVA6 should use this after collecting all per-round RTL plan entries.
 */
moe_status_t moe_lower_hw_plan(const moe_request_t *req,
                               const moe_hw_plan_entry_t *plan,
                               uint16_t n_plan,
                               moe_schedule_t *out);

#ifdef __cplusplus
}
#endif
#endif /* MOE_SCHEDULER_H */
