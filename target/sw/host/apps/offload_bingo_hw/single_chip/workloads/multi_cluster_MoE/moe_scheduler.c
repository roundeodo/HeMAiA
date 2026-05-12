/* moe_scheduler.c
 * --------------------------------------------------------------------------
 * Fast O(E) greedy stub — replaces the O(E^3*N^4) analytical scheduler.
 *
 * Algorithm:
 *   1. Insertion-sort experts descending by ntokens (E <= 8 → trivial cost).
 *   2. Assign each expert to the cluster (C2 or C3) whose free_at time is
 *      smallest (list-scheduling / LPT greedy).
 *   3. Cache-hit detection: if the resident expert in the target cluster
 *      matches the requested expert, skip all DMA (hit = 1).
 *   4. Generate one S1 dma_op (iDMA, gate+up) and one S3 dma_op (xDMA,
 *      down) per task that needs DMA.  The execute kernel assigns
 *      idma_seq / xdma_seq globally from the dma_ops[] array order.
 *
 * Shape: MOE_SHAPE_A (M_dim = 8) for the first GEMM block.  The device
 * kernel handles remaining tokens in shape-C (M_dim = 2) iterations.
 *
 * Cycle estimate (wait_for_peer_slots heuristic only):
 *   T_s1 = 90112 cc, T_s3 = 45056 cc (shape A DMA + compute, 1 token window)
 *   Tail: ceil(max(n-8,0)/2) * 22528 cc (S2) + ceil(max(n-8,0)/2) * 11264 cc (S4)
 * --------------------------------------------------------------------------
 */
#include "moe_scheduler.h"
#include <stdint.h>

/* Shape A (index 0): M_dim=8, T_s1=90112, T_s3=45056 */
#define GREEDY_T_S1       90112u
#define GREEDY_T_S3       45056u
#define GREEDY_T_C_S1     22528u  /* shape-C tail per 2-token chunk, S2 */
#define GREEDY_T_C_S3     11264u  /* shape-C tail per 2-token chunk, S4 */
#define GREEDY_SHAPE_M    8u      /* shape A M_dim */

/* Estimated task duration in cycles. */
static uint32_t greedy_task_cc(uint32_t ntok)
{
    uint32_t tail = (ntok > GREEDY_SHAPE_M) ? (ntok - GREEDY_SHAPE_M) : 0u;
    uint32_t t_s2 = ((tail + 1u) / 2u) * GREEDY_T_C_S1;
    uint32_t t_s4 = ((tail + 1u) / 2u) * GREEDY_T_C_S3;
    return GREEDY_T_S1 + t_s2 + GREEDY_T_S3 + t_s4;
}

moe_status_t moe_schedule(const moe_request_t *req, moe_schedule_t *out)
{
    if (req == (const moe_request_t *)0 || out == (moe_schedule_t *)0)
        return MOE_ERR_BAD_INPUT;

    uint16_t ne = req->n_experts;
    if (ne == 0u || ne > MOE_MAX_EXPERTS)
        return MOE_ERR_BAD_INPUT;

    /* ------------------------------------------------------------------
     * 1. Copy & sort experts descending by ntokens (insertion sort, E<=8)
     * ------------------------------------------------------------------ */
    moe_expert_load_t sorted[MOE_MAX_EXPERTS];
    uint16_t i;
    for (i = 0u; i < ne; i++) sorted[i] = req->experts[i];
    for (i = 1u; i < ne; i++) {
        moe_expert_load_t key = sorted[i];
        int j = (int)i - 1;
        while (j >= 0 && sorted[j].ntokens < key.ntokens) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }

    out->n_tasks         = 0u;
    out->n_dma_ops       = 0u;
    out->est_makespan_cc = 0u;

    uint32_t free_at[2] = {0u, 0u}; /* [0]=C2, [1]=C3 */

    /* ------------------------------------------------------------------
     * 2. Greedy LPT assignment
     * ------------------------------------------------------------------ */
    for (i = 0u; i < ne; i++) {
        if (sorted[i].ntokens == 0u) continue;
        if (out->n_tasks >= MOE_MAX_TASKS) return MOE_ERR_OVERFLOW;

        /* Pick cluster with smaller free_at (ties → C2) */
        int ci = (free_at[0] <= free_at[1]) ? 0 : 1;
        moe_cluster_t cl = (ci == 0) ? MOE_CLUSTER_C2 : MOE_CLUSTER_C3;
        uint32_t t_start = free_at[ci];

        /* Cache-hit: both S1 (gate/up) and S3 (down) already resident */
        int16_t eid = sorted[i].expert_id;
        uint8_t hit = 0u;
        if (ci == 0 && req->cache_eid_c2 == eid) hit = 1u;
        if (ci == 1 && req->cache_eid_c3 == eid) hit = 1u;

        uint32_t dur   = greedy_task_cc((uint32_t)sorted[i].ntokens);
        uint32_t t_end = t_start + dur;

        /* Fill task fields */
        moe_task_t *tk     = &out->tasks[out->n_tasks];
        tk->cluster          = cl;
        tk->expert_id        = eid;
        tk->token_start_rank = 0u;
        tk->ntokens          = sorted[i].ntokens;
        tk->shape_s1         = MOE_SHAPE_A;
        tk->shape_s3         = MOE_SHAPE_A;
        tk->bw_s1            = hit ? 0u   : 32u;
        tk->bw_s3            = hit ? 0u   : 32u;
        tk->dma_s1           = hit ? MOE_DMA_NONE : MOE_DMA_IDMA;
        tk->dma_s3           = hit ? MOE_DMA_NONE : MOE_DMA_XDMA;
        tk->skip_s1          = hit;
        tk->skip_s3          = hit;
        /* S2/S4 skip 逻辑:
         *   cache hit (hit=1): S1/S3 全跳过，S2/S4 处理所有 token
         *   非 cache hit:      S1/S3 处理 ≤shape_M 个 token，ntokens>shape_M 时 S2/S4 处理尾部
         *   当前配置 max_tokens=shape_M=8，所以非 hit 时 skip_s2=1 (ntokens 永远 ≤ shape_M) */
        if (hit) {
            tk->skip_s2   = 0u;
            tk->skip_s4   = 0u;
            tk->m_s2_exec = sorted[i].ntokens;
            tk->m_s4_exec = sorted[i].ntokens;
        } else {
            tk->skip_s2   = (sorted[i].ntokens <= GREEDY_SHAPE_M) ? 1u : 0u;
            tk->skip_s4   = (sorted[i].ntokens <= GREEDY_SHAPE_M) ? 1u : 0u;
            tk->m_s2_exec = (sorted[i].ntokens > GREEDY_SHAPE_M)
                            ? (sorted[i].ntokens - GREEDY_SHAPE_M) : 0u;
            tk->m_s4_exec = (sorted[i].ntokens > GREEDY_SHAPE_M)
                            ? (sorted[i].ntokens - GREEDY_SHAPE_M) : 0u;
        }
        tk->prefetch_eid     = -1;
        tk->est_start_cc     = t_start;
        tk->est_end_cc       = t_end;
        {
            uint32_t s;
            for (s = 0u; s < MOE_TASK_DMA_SLOTS; s++) tk->dma_slots[s].valid = 0u;
        }

        /* S1 dma_op: iDMA, gate+up weights */
        if (!hit) {
            if (out->n_dma_ops >= MOE_MAX_DMA_OPS) return MOE_ERR_OVERFLOW;
            moe_dma_op_t *op = &out->dma_ops[out->n_dma_ops++];
            op->task_idx  = out->n_tasks;
            op->cluster   = cl;
            op->expert_id = eid;
            op->kind      = MOE_DMA_OP_S1;
            op->weight    = MOE_WEIGHT_GATE_UP;
            op->shape     = MOE_SHAPE_A;
            op->alloc_bw  = 32u;
            op->dma       = MOE_DMA_IDMA;
            op->start_cc  = t_start;
            op->end_cc    = t_start + GREEDY_T_S1;
        }

        /* S3 dma_op: xDMA, down weights */
        if (!hit) {
            if (out->n_dma_ops >= MOE_MAX_DMA_OPS) return MOE_ERR_OVERFLOW;
            moe_dma_op_t *op = &out->dma_ops[out->n_dma_ops++];
            op->task_idx  = out->n_tasks;
            op->cluster   = cl;
            op->expert_id = eid;
            op->kind      = MOE_DMA_OP_S3;
            op->weight    = MOE_WEIGHT_DOWN;
            op->shape     = MOE_SHAPE_A;
            op->alloc_bw  = 32u;
            op->dma       = MOE_DMA_XDMA;
            op->start_cc  = t_start + GREEDY_T_S1;
            op->end_cc    = t_start + GREEDY_T_S1 + GREEDY_T_S3;
        }

        free_at[ci] = t_end;
        out->n_tasks++;
    }

    out->est_makespan_cc = (free_at[0] > free_at[1]) ? free_at[0] : free_at[1];
    return MOE_OK;
}
