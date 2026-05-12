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

/* ── compile-time limits (sized for HeMAiA 2-cluster, top-K<=2 routing) ── */
#define MOE_MAX_EXPERTS    8     /* per request                        */
#define MOE_MAX_TASKS     32     /* upper bound for schedule length    */
#define MOE_MAX_DMA_OPS  (MOE_MAX_TASKS * 4)
#define MOE_TASK_DMA_SLOTS 4

#define MOE_TASK_DMA_SLOT_S1          0u
#define MOE_TASK_DMA_SLOT_S3          1u
#define MOE_TASK_DMA_SLOT_S2_PREFETCH 2u
#define MOE_TASK_DMA_SLOT_S4_PREFETCH 3u

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

typedef enum {
    MOE_WEIGHT_GATE_UP = 0,
    MOE_WEIGHT_DOWN = 1
} moe_weight_kind_t;

typedef struct {
    uint8_t valid;            /* 0 = hardware ignores this slot          */
    moe_dma_op_kind_t kind;   /* fixed stage/prefetch role               */
    moe_weight_kind_t weight; /* gate/up or down weight                  */
    int16_t expert_id;        /* expert weight to transfer               */
    moe_shape_t shape;        /* transfer shape                          */
    uint16_t alloc_bw;        /* reserved DMA bandwidth in B/cc          */
    moe_dma_binding_t dma;    /* selected DMA lane(s)                    */
    uint32_t start_cc;        /* analytical absolute cycle               */
    uint32_t end_cc;          /* analytical absolute cycle               */
} moe_task_dma_slot_t;

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

/* ── Output: one scheduled expert run = (cluster, expert, token slice) ─── */
typedef struct {
    moe_cluster_t cluster;    /* which cluster runs this task            */
    uint16_t  expert_id;      /* expert to run                           */
    uint16_t  token_start_rank; /* rank in this expert's routed-token list */
    uint16_t  ntokens;        /* routed-token slice after possible SPLIT */
    moe_shape_t shape_s1;     /* analytical S1 initial token-window shape */
    moe_shape_t shape_s3;     /* analytical S3 initial token-window shape */
    uint16_t  bw_s1;          /* allocated S1 DMA BW in B/cc, 0 if skip  */
    uint16_t  bw_s3;          /* allocated S3 DMA BW in B/cc             */
    moe_dma_binding_t dma_s1; /* DMA lane(s) assigned to S1              */
    moe_dma_binding_t dma_s3; /* DMA lane(s) assigned to S3              */
    uint8_t   skip_s1;        /* 1 if S1 load+compute skipped (cache hit) */
    uint8_t   skip_s3;        /* 1 if S3 load+compute skipped (cache hit) */
    uint8_t   skip_s2;        /* 1 if S2 full compute skipped (ntokens<=shape_M, non-hit) */
    uint8_t   skip_s4;        /* 1 if S4 full compute skipped */
    uint32_t  m_s2_exec;      /* token count for S2 (0 when skip_s2=1)  */
    uint32_t  m_s4_exec;      /* token count for S4 (0 when skip_s4=1)  */
    int16_t   prefetch_eid;   /* expert id to prefetch during S4, -1=none */
    moe_task_dma_slot_t dma_slots[MOE_TASK_DMA_SLOTS];
    uint32_t  est_start_cc;   /* analytical estimate, dispatch hint only */
    uint32_t  est_end_cc;     /* analytical estimate, dispatch hint only */
} moe_task_t;

typedef struct {
    uint16_t task_idx;        /* owner task in tasks[], 0xFFFF if none   */
    moe_cluster_t cluster;    /* cluster whose DMA engine issues op      */
    int16_t expert_id;        /* expert weight being transferred         */
    moe_dma_op_kind_t kind;   /* S1/S3 foreground or prefetch op         */
    moe_weight_kind_t weight; /* gate/up or down weight                  */
    moe_shape_t shape;        /* shape that determines transfer size     */
    uint16_t alloc_bw;        /* reserved DMA bandwidth in B/cc          */
    moe_dma_binding_t dma;    /* lane assignment after binding pass      */
    uint32_t start_cc;
    uint32_t end_cc;
} moe_dma_op_t;

typedef struct {
    moe_task_t tasks[MOE_MAX_TASKS];
    moe_dma_op_t dma_ops[MOE_MAX_DMA_OPS];
    uint16_t   n_tasks;       /* number of valid tasks                    */
    uint16_t   n_dma_ops;     /* number of valid DMA operations           */
    uint32_t   est_makespan_cc;  /* total estimated cycles                */
} moe_schedule_t;

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

#ifdef __cplusplus
}
#endif
#endif /* MOE_SCHEDULER_H */
