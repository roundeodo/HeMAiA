/* moe_scheduler.c
 * --------------------------------------------------------------------------
 * C port of lite_scheduler.py (lite_schedule()).
 * Reference: Idea_Model/lite_scheduler.py + Idea_Model/four_stage_scheduler.py
 *
 * Algorithm:
 *   Greedy two-cluster four-stage scheduler with lookahead.
 *   Main loop: while experts remain, pick the best candidate from:
 *     n=1 : Method-A (solo) + SPLIT + Method-B (early-start).
 *     both_idle: PAIR(top0,topK K=1..3) + PAIR(topK,topJ) + SPLIT(top0).
 *     not_both_idle: assign top0 to idle cluster at multiple start times.
 *   Candidate cost via continuation_cost():
 *     n=0: max(c2.task_end, c3.task_end)
 *     n=1: sim1() – solo/SPLIT search
 *     n=2 && total_tokens<=4: closed-form min
 *     else: greedy_heuristic()
 *
 * No FP, no malloc, no OS deps.  Pure C99.
 * --------------------------------------------------------------------------
 */
#include "moe_scheduler.h"
#include <stdint.h>
#include <stddef.h>

/* =========================================================================
 * Shape constants (indexed 0=A, 1=B, 2=C)
 * =========================================================================
 * WEIGHT_BYTES_S1 = 2*2048*1408/2 = 2,883,584  (gate+up)
 * WEIGHT_BYTES_S3 = 1*2048*1408/2 = 1,441,792  (down)
 * Shape A: M_dim=8,  bw_req=32, alloc=64
 * Shape B: M_dim=4,  bw_req=64, alloc=64
 * Shape C: M_dim=2,  bw_req=128,alloc=128
 */
static const uint32_t kTs1[3]   = {90112u, 45056u, 22528u};
static const uint32_t kTs3[3]   = {45056u, 22528u, 11264u};
static const uint32_t kTd1[3]   = {45056u, 45056u, 22528u};
static const uint32_t kTd3[3]   = {22528u, 22528u, 11264u};
static const uint32_t kMdim[3]  = {8u, 4u, 2u};
static const uint32_t kAlloc[3] = {64u, 64u, 128u};

#define MAX_BW_CC       128u
#define INF_CC          0xFFFFFFFFu
#define T_DMA3_C        11264u          /* ShapeC t_dma_s3; S3-overlap threshold */
#define EXACT_TAIL_MAX  4u              /* continuation_cost n=2 exact threshold  */
/* Sentinel stored in snap_t.pf_eid to mean "S4 window is available for prefetch
 * of whoever gets assigned next".  Any valid expert_id is >= 0. */
#define PF_EID_GHOST    ((int16_t)(-2))

/* =========================================================================
 * Timing primitives
 * ========================================================================= */
static uint32_t best_s2(uint32_t r){ return ((r+1u)/2u)*22528u; }
static uint32_t best_s4(uint32_t r){ return ((r+1u)/2u)*11264u; }
static uint32_t best_task(uint32_t n){ return ((n+1u)/2u)*33792u; }
static uint32_t best_conc(uint32_t n){ return ((n+3u)/4u)*67584u; }

/* =========================================================================
 * snap_t – cluster state snapshot (mirrors FourStageSnap)
 * ========================================================================= */
typedef struct {
    uint32_t task_start, task_end;
    uint32_t dma1_end, s1_end, s2_end, dma3_end, s3_end, s4_start;
    uint32_t bw_s1, bw_s3;
    int16_t  cur_eid, pf_eid;
    int32_t  pf_start, pf_end;
    uint32_t pf_bw;
    uint8_t  pf_full;
    int32_t  s2pf_start, s2pf_end;
    uint32_t s2pf_bw;
    uint16_t ntok;
} snap_t;

static snap_t snap_idle_at(uint32_t t)
{
    snap_t s;
    s.task_start=t; s.task_end=t; s.dma1_end=t; s.s1_end=t;
    s.s2_end=t; s.dma3_end=t; s.s3_end=t; s.s4_start=t;
    s.bw_s1=0; s.bw_s3=0; s.cur_eid=-1;
    s.pf_eid=-1; s.pf_start=-1; s.pf_end=-1; s.pf_bw=0; s.pf_full=0;
    s.s2pf_start=-1; s.s2pf_end=-1; s.s2pf_bw=0; s.ntok=0;
    return s;
}

/* Initial snap encoding a cached expert (mirrors make_initial_snap). */
static snap_t snap_initial(int16_t ce)
{
    snap_t s = snap_idle_at(0u);
    if (ce >= 0){ s.pf_eid=ce; s.pf_end=0; s.pf_full=1; }
    return s;
}

/* Create snap for one task assignment. */
static snap_t mk_snap(uint32_t start, uint8_t s1, uint8_t s3,
                      uint16_t ntok, int16_t eid, uint8_t s1c, uint8_t s3c)
{
    snap_t r = snap_idle_at(start);
    r.cur_eid=eid; r.ntok=ntok;
    if (s1c){
        r.dma1_end=start; r.s1_end=start; r.bw_s1=0;
        r.s2_end=start+best_s2(ntok);
    } else {
        uint32_t rm=(ntok>kMdim[s1])?(ntok-kMdim[s1]):0u;
        r.dma1_end=start+kTd1[s1]; r.s1_end=start+kTs1[s1];
        r.bw_s1=kAlloc[s1]; r.s2_end=r.s1_end+best_s2(rm);
    }
    if (s3c){
        r.dma3_end=r.s2_end; r.s3_end=r.s2_end; r.s4_start=r.s2_end;
        r.bw_s3=0; r.task_end=r.s2_end+best_s4(ntok);
    } else {
        uint32_t rm=(ntok>kMdim[s3])?(ntok-kMdim[s3]):0u;
        r.dma3_end=r.s2_end+kTd3[s3]; r.s3_end=r.s2_end+kTs3[s3];
        r.s4_start=r.s3_end; r.bw_s3=kAlloc[s3];
        r.task_end=r.s3_end+best_s4(rm);
    }
    return r;
}

/* Apply S2 down-prefetch; returns unchanged if it doesn't fit. */
static snap_t apply_s2pf(snap_t sn, uint8_t s3, uint32_t ps)
{
    uint32_t pe=ps+kTd3[s3];
    if (sn.bw_s3==0u) return sn;
    if (ps<sn.task_start || pe>sn.s2_end) return sn;
    sn.s2pf_start=(int32_t)ps; sn.s2pf_end=(int32_t)pe; sn.s2pf_bw=kAlloc[s3];
    sn.dma3_end=sn.s2_end; sn.s3_end=sn.s2_end; sn.s4_start=sn.s2_end; sn.bw_s3=0;
    sn.task_end=sn.s2_end+best_s4(sn.ntok);
    return sn;
}

/* =========================================================================
 * BW feasibility — merged-segment approach (~7x faster than point-sampling)
 *
 * Within a single snap, only two intervals can overlap in time:
 *   iv1: [task_start, dma1_end)  — S1 DMA   (bw_s1)
 *   iv4: [s2pf_start, s2pf_end) — s2pf DMA  (s2pf_bw)
 * (iv2=[s2_end,dma3_end) and pf_start are always disjoint from iv1/iv4;
 *  after apply_s2pf bw_s3=0 so iv2 is inactive when iv4 is active.)
 *
 * snap_segs() builds the merged piecewise-constant BW profile (≤4 segments).
 * bw_ok() then does a 4×4=16-pair cross-check between two snaps.
 * ========================================================================= */
typedef struct { uint32_t lo, hi, bw; } seg_t;

static int snap_segs(const snap_t *s, seg_t out[4], int *n)
{
    *n = 0;
    /* iv1: S1 DMA */
    int has1 = (s->cur_eid>=0 && s->bw_s1>0u && s->dma1_end>s->task_start);
    uint32_t s1lo=s->task_start, s1hi=s->dma1_end, s1bw=s->bw_s1;
    /* iv4: s2pf DMA (only present after apply_s2pf; bw_s3=0 in that case) */
    int has4 = (s->s2pf_start>=0 && s->s2pf_bw>0u &&
                (uint32_t)s->s2pf_end>(uint32_t)s->s2pf_start);
    uint32_t p4lo=(uint32_t)s->s2pf_start, p4hi=(uint32_t)s->s2pf_end, p4bw=s->s2pf_bw;

    if (has1 && has4 && s1lo<p4hi && p4lo<s1hi) {
        /* iv1 and iv4 overlap: build merged 3-segment profile */
        uint32_t ovl_lo=(s1lo>p4lo)?s1lo:p4lo;
        uint32_t ovl_hi=(s1hi<p4hi)?s1hi:p4hi;
        uint32_t merged=s1bw+p4bw;
        if (merged>MAX_BW_CC) return 0;               /* single-snap violation */
        if (s1lo<p4lo)      out[(*n)++]=(seg_t){s1lo, p4lo, s1bw};
        else if (p4lo<s1lo) out[(*n)++]=(seg_t){p4lo, s1lo, p4bw};
        if (ovl_hi>ovl_lo)  out[(*n)++]=(seg_t){ovl_lo, ovl_hi, merged};
        if (s1hi>p4hi)      out[(*n)++]=(seg_t){p4hi, s1hi, s1bw};
        else if (p4hi>s1hi) out[(*n)++]=(seg_t){s1hi, p4hi, p4bw};
    } else {
        if (has1) out[(*n)++]=(seg_t){s1lo, s1hi, s1bw};
        if (has4) out[(*n)++]=(seg_t){p4lo, p4hi, p4bw};
    }
    /* iv2: S3 DMA — always after iv1/iv4; active only when bw_s3>0 (no s2pf) */
    if (s->cur_eid>=0 && s->bw_s3>0u && s->dma3_end>s->s2_end)
        out[(*n)++]=(seg_t){s->s2_end, s->dma3_end, s->bw_s3};
    return 1;
}

static int bw_ok(const snap_t *a, const snap_t *b)
{
    seg_t sa[4], sb[4]; int na=0, nb=0;
    if (!snap_segs(a,sa,&na)) return 0;
    if (!snap_segs(b,sb,&nb)) return 0;
    for (int i=0;i<na;i++) for (int j=0;j<nb;j++) {
        uint32_t lo=(sa[i].lo>sb[j].lo)?sa[i].lo:sb[j].lo;
        uint32_t hi=(sa[i].hi<sb[j].hi)?sa[i].hi:sb[j].hi;
        if (lo<hi && sa[i].bw+sb[j].bw>MAX_BW_CC) return 0;
    }
    return 1;
}

/* =========================================================================
 * S2 down-prefetch pair optimisation
 * ========================================================================= */
static void try_s2pf_pair(snap_t *sa, uint8_t s3a, snap_t *sb, uint8_t s3b)
{
    uint32_t ca[5]; int na=0;
    if (sa->bw_s3>0u && kTd3[s3a]<=sa->s2_end-sa->task_start){
        uint32_t lo=sa->task_start, hi=sa->s2_end-kTd3[s3a];
        if (hi>=lo){
            ca[na++]=lo;
            if (sa->dma1_end>=lo&&sa->dma1_end<=hi&&sa->dma1_end!=lo) ca[na++]=sa->dma1_end;
            if (sa->s1_end>=lo&&sa->s1_end<=hi&&sa->s1_end!=sa->dma1_end) ca[na++]=sa->s1_end;
            if (hi!=lo) ca[na++]=hi;
        }
    }
    uint32_t cb[5]; int nb=0;
    if (sb->bw_s3>0u && kTd3[s3b]<=sb->s2_end-sb->task_start){
        uint32_t lo=sb->task_start, hi=sb->s2_end-kTd3[s3b];
        if (hi>=lo){
            cb[nb++]=lo;
            if (sb->dma1_end>=lo&&sb->dma1_end<=hi&&sb->dma1_end!=lo) cb[nb++]=sb->dma1_end;
            if (sb->s1_end>=lo&&sb->s1_end<=hi&&sb->s1_end!=sb->dma1_end) cb[nb++]=sb->s1_end;
            if (hi!=lo) cb[nb++]=hi;
        }
    }

    int best_sc=-1; uint64_t best_ss=0xFFFFFFFFFFFFFFFFull;
    snap_t best_a=*sa, best_b=*sb;

    /* No pf */
    if (bw_ok(sa,sb)){ best_sc=0; best_ss=0; best_a=*sa; best_b=*sb; }

    /* Pf on A only */
    for (int ia=0;ia<na;ia++){
        snap_t ta=apply_s2pf(*sa,s3a,ca[ia]);
        if (ta.s2pf_start<0) continue;
        if (bw_ok(&ta,sb)){
            uint64_t ss=(uint64_t)ca[ia];
            if (1>best_sc||(1==best_sc&&ss<best_ss)){best_sc=1;best_ss=ss;best_a=ta;best_b=*sb;}
        }
    }
    /* Pf on B only */
    for (int ib=0;ib<nb;ib++){
        snap_t tb=apply_s2pf(*sb,s3b,cb[ib]);
        if (tb.s2pf_start<0) continue;
        if (bw_ok(sa,&tb)){
            uint64_t ss=(uint64_t)cb[ib];
            if (1>best_sc||(1==best_sc&&ss<best_ss)){best_sc=1;best_ss=ss;best_a=*sa;best_b=tb;}
        }
    }
    /* Pf on both */
    for (int ia=0;ia<na;ia++){
        snap_t ta=apply_s2pf(*sa,s3a,ca[ia]);
        if (ta.s2pf_start<0) continue;
        for (int ib=0;ib<nb;ib++){
            snap_t tb=apply_s2pf(*sb,s3b,cb[ib]);
            if (tb.s2pf_start<0) continue;
            if (!bw_ok(&ta,&tb)) continue;
            uint64_t ss=(uint64_t)ca[ia]+(uint64_t)cb[ib];
            if (2>best_sc||(2==best_sc&&ss<best_ss)){best_sc=2;best_ss=ss;best_a=ta;best_b=tb;}
        }
    }
    *sa=best_a; *sb=best_b;
}

/* =========================================================================
 * Cache hit queries
 * ========================================================================= */
static int swiglu_hit(int16_t eid, const snap_t *s, uint32_t t)
{
    if (s->pf_end < 0 || (uint32_t)s->pf_end > t) return 0;
    /* PF_EID_GHOST means the S4 window can prefetch whoever arrives next:
     * any eid benefits from skip_s1. */
    return s->pf_eid == PF_EID_GHOST || s->pf_eid == eid;
}

static int down_hit(int16_t eid, const snap_t *s, uint32_t t)
{ return swiglu_hit(eid,s,t) && s->pf_full; }

/* =========================================================================
 * Analytical shape selection  (_pick_pair_shapes from lite_scheduler.py)
 * ========================================================================= */
static void pick_shapes(uint16_t na, uint16_t nb,
                        uint8_t sw_a, uint8_t dn_a, uint8_t sw_b, uint8_t dn_b,
                        uint32_t t0,
                        uint8_t *s1a, uint8_t *s3a, uint8_t *s1b, uint8_t *s3b)
{
    if (sw_a||sw_b){ *s1a=2; *s1b=2; } else { *s1a=1; *s1b=1; }

    uint32_t s2a, s2b;
    if (sw_a) s2a=t0+best_s2(na);
    else { uint32_t r=(na>kMdim[*s1a])?(na-kMdim[*s1a]):0u; s2a=t0+kTs1[*s1a]+best_s2(r); }
    if (sw_b) s2b=t0+best_s2(nb);
    else { uint32_t r=(nb>kMdim[*s1b])?(nb-kMdim[*s1b]):0u; s2b=t0+kTs1[*s1b]+best_s2(r); }

    /* hit侧DMA=0 → S4执行全部ntok: ShapeC(M_dim=2,11264cc/tile)匹配best_s4 */
    if (dn_a||dn_b){ *s3a=2; *s3b=2; }
    else {
        uint32_t d=(s2a>=s2b)?(s2a-s2b):(s2b-s2a);
        if (d>=T_DMA3_C){ *s3a=2; *s3b=2; } else { *s3a=1; *s3b=1; }
    }
}

/* =========================================================================
 * Remaining expert entry type (needed by greedy_h / continuation_cost)
 * ========================================================================= */
typedef struct { int16_t eid; uint16_t ntok; } rem_t;

/* =========================================================================
 * Greedy heuristic + continuation_cost + sim1
 * ========================================================================= */
static uint32_t sim1(const snap_t *c2, const snap_t *c3,
                     int16_t eid, uint16_t ntok);

static uint32_t greedy_h(uint32_t c2e, uint32_t c3e,
                          const rem_t *rem, uint8_t nr)
{
    uint32_t max_e=(c2e>c3e)?c2e:c3e;
    if (nr==0u) return max_e;
    if (nr==1u){
        uint32_t nt=rem[0].ntok;
        uint32_t te=(c2e<c3e)?c2e:c3e, tl=max_e;
        uint32_t sc=(tl>(te+best_task(nt)))?tl:(te+best_task(nt));
        uint32_t sp=tl+best_conc((nt+1u)/2u);
        return (sc<sp)?sc:sp;
    }
    if (nr==2u){
        uint32_t te=(c2e<c3e)?c2e:c3e, tl=max_e;
        uint32_t bc0=best_conc(rem[0].ntok), bc1=best_conc(rem[1].ntok);
        uint32_t pc=tl+((bc0>bc1)?bc0:bc1);
        uint32_t ser=te+best_task(rem[0].ntok)+best_task(rem[1].ntok);
        uint32_t serc=(ser>tl)?ser:tl;
        return (pc<serc)?pc:serc;
    }
    uint32_t sum=0u, mx=0u;
    for (uint8_t i=0u;i<nr;i++){
        uint32_t bc=best_conc(rem[i].ntok); sum+=bc; if(bc>mx)mx=bc;
    }
    uint32_t extra=(mx>sum/2u)?mx:sum/2u;
    return max_e+extra;
}

static uint32_t continuation_cost(const snap_t *c2, const snap_t *c3,
                                   const rem_t *rem, uint8_t nr)
{
    if (nr==0u) return (c2->task_end>c3->task_end)?c2->task_end:c3->task_end;
    if (nr==1u) return sim1(c2,c3,rem[0].eid,rem[0].ntok);
    if (nr==2u && rem[0].ntok+rem[1].ntok<=EXACT_TAIL_MAX){
        uint32_t te=(c2->task_end<c3->task_end)?c2->task_end:c3->task_end;
        uint32_t tl=(c2->task_end>c3->task_end)?c2->task_end:c3->task_end;
        uint32_t ss=te+best_task(rem[0].ntok)+best_task(rem[1].ntok);
        uint32_t bc0=best_conc(rem[0].ntok), bc1=best_conc(rem[1].ntok);
        uint32_t pa=tl+((bc0>bc1)?bc0:bc1);
        uint32_t v1=(tl>ss)?tl:ss;
        return (v1<pa)?v1:pa;
    }
    return greedy_h(c2->task_end,c3->task_end,rem,nr);
}

static uint32_t sim1(const snap_t *c2, const snap_t *c3,
                     int16_t eid, uint16_t ntok)
{
    uint32_t t=(c2->task_end>c3->task_end)?c2->task_end:c3->task_end;
    uint32_t best=INF_CC;
    const snap_t *sns[2]={c2,c3};
    for (int ci=0;ci<2;ci++){
        uint8_t cc=(uint8_t)swiglu_hit(eid,sns[ci],t);
        uint8_t cf=(uint8_t)down_hit(eid,sns[ci],t);
        snap_t sn=mk_snap(t,2u,2u,ntok,eid,cc,cf);
        if (sn.task_end<best) best=sn.task_end;
        if (!cc){ snap_t sn2=mk_snap(t,1u,1u,ntok,eid,0u,0u); if(sn2.task_end<best)best=sn2.task_end; }
    }
    if (ntok>=2u){
        uint16_t ca=(uint16_t)((ntok+1u)/2u), cb=ntok-ca;
        uint8_t sw_a=(uint8_t)swiglu_hit(eid,c2,t),dn_a=(uint8_t)down_hit(eid,c2,t);
        uint8_t sw_b=(uint8_t)swiglu_hit(eid,c3,t),dn_b=(uint8_t)down_hit(eid,c3,t);
        uint8_t s1a,s3a,s1b,s3b;
        pick_shapes(ca,cb,sw_a,dn_a,sw_b,dn_b,t,&s1a,&s3a,&s1b,&s3b);
        snap_t sna=mk_snap(t,s1a,s3a,ca,eid,sw_a,dn_a);
        snap_t snb=mk_snap(t,s1b,s3b,cb,eid,sw_b,dn_b);
        try_s2pf_pair(&sna,s3a,&snb,s3b);
        if (bw_ok(&sna,&snb)){
            uint32_t e=(sna.task_end>snb.task_end)?sna.task_end:snb.task_end;
            if (e<best) best=e;
        }
    }
    return (best==INF_CC)?(t+best_task(ntok)):best;
}

/* =========================================================================
 * Plan entry (one scheduled task decision)
 * ========================================================================= */
typedef struct {
    int16_t  eid;
    uint16_t ntok, tok_start;
    uint8_t  cluster;          /* 0=C2, 1=C3 */
    uint8_t  shape_s1, shape_s3;
    uint8_t  skip_s1, skip_s3, has_s2pf;
    uint32_t est_start, est_end, est_s2_end;
    uint32_t est_s2pf_start, est_dma1_end, est_s4_start;
} plan_t;

static plan_t plan_from_snap(const snap_t *sn, uint8_t cl,
                              uint16_t tok_start, uint8_t s1, uint8_t s3)
{
    plan_t p;
    p.eid=sn->cur_eid; p.ntok=sn->ntok; p.tok_start=tok_start;
    p.cluster=cl; p.shape_s1=s1; p.shape_s3=s3;
    p.skip_s1=(sn->bw_s1==0u)?1u:0u;
    p.skip_s3=(sn->bw_s3==0u)?1u:0u;
    p.has_s2pf=(sn->s2pf_start>=0)?1u:0u;
    p.est_start=sn->task_start; p.est_end=sn->task_end;
    p.est_s2_end=sn->s2_end;
    p.est_s2pf_start=(sn->s2pf_start>=0)?(uint32_t)sn->s2pf_start:0u;
    p.est_dma1_end=sn->dma1_end; p.est_s4_start=sn->s4_start;
    return p;
}

/* =========================================================================
 * Remaining expert list helpers
 * ========================================================================= */
static void rem_remove(rem_t *rem, uint8_t *nr, uint8_t idx)
{ for (uint8_t i=idx;i+1u<*nr;i++) rem[i]=rem[i+1u]; (*nr)--; }

/* =========================================================================
 * Candidate comparison
 * ========================================================================= */
typedef struct { uint32_t cost,snap_max,snap_min; uint8_t rem_len; int valid; } ckey_t;

static int cand_better(const ckey_t *b, uint32_t cost, uint32_t smx, uint32_t smn, uint8_t rl)
{
    if (!b->valid) return 1;
    if (cost<b->cost) return 1; if (cost>b->cost) return 0;
    if (rl<b->rem_len) return 1; if (rl>b->rem_len) return 0;
    if (smx<b->snap_max) return 1; if (smx>b->snap_max) return 0;
    if (smn>b->snap_min) return 1;
    return 0;
}

/* =========================================================================
 * Main scheduling loop
 * ========================================================================= */
static uint32_t moe_plan(const moe_request_t *req, plan_t *plan, uint8_t *n_plan)
{
    rem_t rem[MOE_MAX_EXPERTS];
    uint8_t nr=(uint8_t)req->n_experts;
    uint8_t i;
    for (i=0u;i<nr;i++){ rem[i].eid=(int16_t)req->experts[i].expert_id; rem[i].ntok=req->experts[i].ntokens; }
    /* Insertion sort descending */
    for (i=1u;i<nr;i++){
        rem_t key=rem[i]; int j=(int)i-1;
        while (j>=0&&rem[j].ntok<key.ntok){rem[j+1]=rem[j];j--;} rem[j+1]=key;
    }

    snap_t c2=snap_initial(req->cache_eid_c2);
    snap_t c3=snap_initial(req->cache_eid_c3);
    *n_plan=0u;

    while (nr>0u){
        int16_t  t0eid =rem[0].eid;
        uint16_t t0ntok=rem[0].ntok;
        uint32_t t2=c2.task_end, t3=c3.task_end;
        uint32_t tnow=(t2>t3)?t2:t3;
        int both_idle=(t2==t3);

        /* Inject PF_EID_GHOST into each cluster whose S4 window can fit a
         * ShapeA S1 DMA (conservative lower-bound, no BW check here).
         * PF_EID_GHOST means "whoever gets assigned to this cluster next will
         * benefit from skip_s1"; lower_plan fills in the actual expert_id when
         * emitting the S4_PREFETCH DMA op.
         * Condition: pf_eid == -1 (no ghost and no initial cache yet). */
        if (c2.cur_eid>=0 && c2.pf_eid==(int16_t)(-1) &&
                c2.s4_start+kTd1[0]<=c2.task_end){
            c2.pf_eid=PF_EID_GHOST; c2.pf_end=(int32_t)c2.task_end; c2.pf_full=0;
        }
        if (c3.cur_eid>=0 && c3.pf_eid==(int16_t)(-1) &&
                c3.s4_start+kTd1[0]<=c3.task_end){
            c3.pf_eid=PF_EID_GHOST; c3.pf_end=(int32_t)c3.task_end; c3.pf_full=0;
        }

        uint8_t c2c0=(uint8_t)swiglu_hit(t0eid,&c2,tnow), c2f0=(uint8_t)down_hit(t0eid,&c2,tnow);
        uint8_t c3c0=(uint8_t)swiglu_hit(t0eid,&c3,tnow), c3f0=(uint8_t)down_hit(t0eid,&c3,tnow);

        /* ── n=1 ─────────────────────────────────────────────────────────── */
        if (nr==1u){
            uint32_t best_cost=INF_CC;
            snap_t best_sn; uint8_t best_cl=0,best_s1=2,best_s3=2;
            uint8_t is_split=0; uint16_t split_cut=0;
            snap_t split_snb;

            /* Method A: solo on each cluster, all 9 (s1,s3) combos */
            for (int ci=0;ci<2;ci++){
                snap_t *snap_ci=(ci==0)?&c2:&c3;
                snap_t *peer   =(ci==0)?&c3:&c2;
                uint32_t tst=snap_ci->task_end;
                uint8_t cc=(uint8_t)swiglu_hit(t0eid,snap_ci,tst);
                uint8_t cf=(uint8_t)down_hit(t0eid,snap_ci,tst);
                for (uint8_t s1=0;s1<3u;s1++) for (uint8_t s3=0;s3<3u;s3++){
                    snap_t sn=mk_snap(tst,s1,s3,t0ntok,t0eid,cc,cf);
                    if (!bw_ok(&sn,peer)) continue;
                    uint32_t ms=(sn.task_end>peer->task_end)?sn.task_end:peer->task_end;
                    if (ms<best_cost){ best_cost=ms; best_sn=sn; best_cl=(uint8_t)ci; best_s1=s1; best_s3=s3; is_split=0; }
                }
            }

            /* SPLIT */
            if (t0ntok>=2u){
                uint16_t cuts[2]; uint8_t nc=0;
                uint16_t h1=(uint16_t)((t0ntok+1u)/2u), h2=(uint16_t)(t0ntok/2u);
                cuts[nc++]=h1; if (h2!=h1&&h2>=1u&&h2<=(uint16_t)(t0ntok-1u)) cuts[nc++]=h2;
                for (uint8_t ci2=0;ci2<nc;ci2++){
                    uint16_t cut_a=cuts[ci2], cut_b=t0ntok-cut_a;
                    uint8_t s1a,s3a,s1b,s3b;
                    pick_shapes(cut_a,cut_b,c2c0,c2f0,c3c0,c3f0,tnow,&s1a,&s3a,&s1b,&s3b);
                    snap_t sna=mk_snap(tnow,s1a,s3a,cut_a,t0eid,c2c0,c2f0);
                    snap_t snb=mk_snap(tnow,s1b,s3b,cut_b,t0eid,c3c0,c3f0);
                    try_s2pf_pair(&sna,s3a,&snb,s3b);
                    if (!bw_ok(&sna,&snb)) continue;
                    uint32_t e=(sna.task_end>snb.task_end)?sna.task_end:snb.task_end;
                    if (e<best_cost){ best_cost=e; is_split=1; split_cut=cut_a; best_sn=sna; split_snb=snb; best_s1=s1a; best_s3=s3a; }
                }
            }

            /* Method B: early-start on idle cluster (not_both_idle case) */
            if (!both_idle){
                int idle_ci=(t2<t3)?0:1;
                snap_t *idle_s=(idle_ci==0)?&c2:&c3;
                snap_t *busy_s=(idle_ci==0)?&c3:&c2;
                uint32_t idle_t=(idle_ci==0)?t2:t3;
                /* Analytical: 3 time pts (idle_t + busy DMA hi-endpoints) x 1 shape */
                uint32_t tpts[3]; int ntp=0;
                tpts[ntp++]=idle_t;
                { seg_t bsegs[4]; int nbsegs=0; snap_segs(busy_s,bsegs,&nbsegs);
                  for (int bi=0;bi<nbsegs&&ntp<3;bi++){
                    uint32_t ep=bsegs[bi].hi;
                    if (ep>idle_t){
                        int dup=0; for(int ti=0;ti<ntp;ti++) if(tpts[ti]==ep){dup=1;break;}
                        if (!dup) tpts[ntp++]=ep;
                    }
                  }
                }
                for (int ti=0;ti<ntp;ti++){
                    uint32_t tst=tpts[ti];
                    uint8_t cc=(uint8_t)swiglu_hit(t0eid,idle_s,tst);
                    uint8_t cf=(uint8_t)down_hit(t0eid,idle_s,tst);
                    uint8_t s1=2u, s3=2u;  /* ShapeC: fastest; BW conflict handled by next tpt */
                    snap_t sn=mk_snap(tst,s1,s3,t0ntok,t0eid,cc,cf);
                    int ok=(idle_ci==0)?bw_ok(&sn,busy_s):bw_ok(busy_s,&sn);
                    if (!ok) continue;
                    uint32_t ms=(sn.task_end>busy_s->task_end)?sn.task_end:busy_s->task_end;
                    if (ms<best_cost){ best_cost=ms; best_sn=sn; best_cl=(uint8_t)idle_ci; best_s1=s1; best_s3=s3; is_split=0; }
                }
            }

            rem_remove(rem,&nr,0u);
            if (!is_split){
                plan_t pe=plan_from_snap(&best_sn,best_cl,0u,best_s1,best_s3);
                if (*n_plan<MOE_MAX_TASKS) plan[(*n_plan)++]=pe;
                if (best_cl==0) c2=best_sn; else c3=best_sn;
            } else {
                /* Re-derive shapes for split commit */
                uint8_t s1a2,s3a2,s1b2,s3b2;
                pick_shapes(split_cut,(uint16_t)(t0ntok-split_cut),
                            c2c0,c2f0,c3c0,c3f0,tnow,&s1a2,&s3a2,&s1b2,&s3b2);
                plan_t pea=plan_from_snap(&best_sn,0u,0u,s1a2,s3a2);
                plan_t peb=plan_from_snap(&split_snb,1u,split_cut,s1b2,s3b2);
                if (*n_plan<MOE_MAX_TASKS) plan[(*n_plan)++]=pea;
                if (*n_plan<MOE_MAX_TASKS) plan[(*n_plan)++]=peb;
                c2=best_sn; c3=split_snb;
            }
            continue;
        }

        /* ── both_idle ───────────────────────────────────────────────────── */
        if (both_idle){
            ckey_t bkey; bkey.valid=0;
            snap_t bsna, bsnb;
            int16_t beid_a=-1, beid_b=-1;
            uint16_t bntok_a=0, bntok_b=0, btok0_a=0, btok0_b=0;
            uint8_t bs1a=2,bs3a=2,bs1b=2,bs3b=2;
            rem_t brem[MOE_MAX_EXPERTS]; uint8_t bnr=0;

/* Macro to evaluate a PAIR candidate and update best. */
#define EVAL_PAIR_BI(sna_, s1a_, s3a_, ea_, na_, tok0a_, \
                     snb_, s1b_, s3b_, eb_, nb_, tok0b_, \
                     rem_, nrr_, is_spl_) \
do { \
    snap_t ta_=(sna_), tb_=(snb_); \
    try_s2pf_pair(&ta_,s3a_,&tb_,s3b_); \
    if (!bw_ok(&ta_,&tb_)) break; \
    uint32_t cost_=continuation_cost(&ta_,&tb_,(rem_),(nrr_)); \
    uint32_t smx_=(ta_.task_end>tb_.task_end)?ta_.task_end:tb_.task_end; \
    uint32_t smn_=(ta_.task_end<tb_.task_end)?ta_.task_end:tb_.task_end; \
    if (cand_better(&bkey,cost_,smx_,smn_,(nrr_))){ \
        bkey.cost=cost_;bkey.snap_max=smx_;bkey.snap_min=smn_; \
        bkey.rem_len=(nrr_);bkey.valid=1; \
        bsna=ta_;bsnb=tb_; \
        beid_a=(ea_);beid_b=(eb_); \
        bntok_a=(na_);bntok_b=(nb_); \
        btok0_a=(tok0a_);btok0_b=(tok0b_); \
        bs1a=(s1a_);bs3a=(s3a_);bs1b=(s1b_);bs3b=(s3b_); \
        (void)(is_spl_); \
        for(uint8_t ri_=0u;ri_<(nrr_);ri_++) brem[ri_]=(rem_)[ri_]; \
        bnr=(nrr_); \
    } \
} while(0)

            /* PAIR(top0, topK) K=1..min(3,nr-1) */
            uint8_t maxK=(nr-1u<3u)?(uint8_t)(nr-1u):3u;
            for (uint8_t K=1u;K<=maxK;K++){
                int16_t Keid=rem[K].eid; uint16_t Kntok=rem[K].ntok;
                rem_t ra[MOE_MAX_EXPERTS]; uint8_t nra=0;
                for (uint8_t ri=0;ri<nr;ri++) if(rem[ri].eid!=t0eid&&rem[ri].eid!=Keid) ra[nra++]=rem[ri];

                /* Dir 1: top0→C2, topK→C3 */
                { uint8_t sw_a=c2c0,dn_a=c2f0;
                  uint8_t sw_b=(uint8_t)swiglu_hit(Keid,&c3,tnow),dn_b=(uint8_t)down_hit(Keid,&c3,tnow);
                  uint8_t s1a,s3a,s1b,s3b;
                  pick_shapes(t0ntok,Kntok,sw_a,dn_a,sw_b,dn_b,tnow,&s1a,&s3a,&s1b,&s3b);
                  snap_t sa=mk_snap(tnow,s1a,s3a,t0ntok,t0eid,sw_a,dn_a);
                  snap_t sb=mk_snap(tnow,s1b,s3b,Kntok,Keid,sw_b,dn_b);
                  EVAL_PAIR_BI(sa,s1a,s3a,t0eid,t0ntok,0u,sb,s1b,s3b,Keid,Kntok,0u,ra,nra,0u); }
                /* Dir 2: topK→C2, top0→C3 */
                { uint8_t sw_a=(uint8_t)swiglu_hit(Keid,&c2,tnow),dn_a=(uint8_t)down_hit(Keid,&c2,tnow);
                  uint8_t sw_b=c3c0,dn_b=c3f0;
                  uint8_t s1a,s3a,s1b,s3b;
                  pick_shapes(Kntok,t0ntok,sw_a,dn_a,sw_b,dn_b,tnow,&s1a,&s3a,&s1b,&s3b);
                  snap_t sa=mk_snap(tnow,s1a,s3a,Kntok,Keid,sw_a,dn_a);
                  snap_t sb=mk_snap(tnow,s1b,s3b,t0ntok,t0eid,sw_b,dn_b);
                  EVAL_PAIR_BI(sa,s1a,s3a,Keid,Kntok,0u,sb,s1b,s3b,t0eid,t0ntok,0u,ra,nra,0u); }
            }

            /* PAIR(topK, topJ) K>=1, J>K */
            if (nr>=3u){
                uint8_t mKJ=(nr-1u<3u)?(uint8_t)(nr-1u):3u;
                for (uint8_t K=1u;K<mKJ;K++) for (uint8_t J=K+1u;J<=mKJ&&J<nr;J++){
                    int16_t eidK=rem[K].eid; uint16_t ntK=rem[K].ntok;
                    int16_t eidJ=rem[J].eid; uint16_t ntJ=rem[J].ntok;
                    rem_t ra[MOE_MAX_EXPERTS]; uint8_t nra=0;
                    for (uint8_t ri=0;ri<nr;ri++) if(rem[ri].eid!=eidK&&rem[ri].eid!=eidJ) ra[nra++]=rem[ri];
                    if (nra==0u) continue;
                    /* Dir1 */
                    { uint8_t sw_a=(uint8_t)swiglu_hit(eidK,&c2,tnow),dn_a=(uint8_t)down_hit(eidK,&c2,tnow);
                      uint8_t sw_b=(uint8_t)swiglu_hit(eidJ,&c3,tnow),dn_b=(uint8_t)down_hit(eidJ,&c3,tnow);
                      uint8_t s1a,s3a,s1b,s3b;
                      pick_shapes(ntK,ntJ,sw_a,dn_a,sw_b,dn_b,tnow,&s1a,&s3a,&s1b,&s3b);
                      snap_t sa=mk_snap(tnow,s1a,s3a,ntK,eidK,sw_a,dn_a);
                      snap_t sb=mk_snap(tnow,s1b,s3b,ntJ,eidJ,sw_b,dn_b);
                      EVAL_PAIR_BI(sa,s1a,s3a,eidK,ntK,0u,sb,s1b,s3b,eidJ,ntJ,0u,ra,nra,0u); }
                    /* Dir2 */
                    { uint8_t sw_a=(uint8_t)swiglu_hit(eidJ,&c2,tnow),dn_a=(uint8_t)down_hit(eidJ,&c2,tnow);
                      uint8_t sw_b=(uint8_t)swiglu_hit(eidK,&c3,tnow),dn_b=(uint8_t)down_hit(eidK,&c3,tnow);
                      uint8_t s1a,s3a,s1b,s3b;
                      pick_shapes(ntJ,ntK,sw_a,dn_a,sw_b,dn_b,tnow,&s1a,&s3a,&s1b,&s3b);
                      snap_t sa=mk_snap(tnow,s1a,s3a,ntJ,eidJ,sw_a,dn_a);
                      snap_t sb=mk_snap(tnow,s1b,s3b,ntK,eidK,sw_b,dn_b);
                      EVAL_PAIR_BI(sa,s1a,s3a,eidJ,ntJ,0u,sb,s1b,s3b,eidK,ntK,0u,ra,nra,0u); }
                }
            }

            /* SPLIT(top0) */
            if (t0ntok>=2u){
                uint16_t cuts[8]; uint8_t nc=0;
                { uint16_t h1=(uint16_t)((t0ntok+1u)/2u), h2=(uint16_t)(t0ntok/2u);
                  cuts[nc++]=h1;
                  if (h2!=h1&&h2>=1u) cuts[nc++]=h2; }
                uint32_t mds[3]={8u,4u,2u};
                for (int mi=0;mi<3;mi++){
                    if (mds[mi]<t0ntok){ uint16_t k=(uint16_t)mds[mi];
                        int dup=0; for(uint8_t ci=0;ci<nc;ci++) if(cuts[ci]==k){dup=1;break;}
                        if (!dup&&nc<8u) cuts[nc++]=k; }
                    if (t0ntok>mds[mi]){ uint16_t k2=(uint16_t)(t0ntok-mds[mi]);
                        if (k2>=1u){ int dup=0; for(uint8_t ci=0;ci<nc;ci++) if(cuts[ci]==k2){dup=1;break;}
                            if(!dup&&nc<8u) cuts[nc++]=k2; } }
                }
                rem_t ra[MOE_MAX_EXPERTS]; uint8_t nra=0;
                for (uint8_t ri=0;ri<nr;ri++) if(rem[ri].eid!=t0eid) ra[nra++]=rem[ri];
                for (uint8_t ci=0;ci<nc;ci++){
                    uint16_t cut_a=cuts[ci], cut_b=t0ntok-cut_a;
                    if (cut_a==0||cut_b==0) continue;
                    uint8_t s1a,s3a,s1b,s3b;
                    pick_shapes(cut_a,cut_b,c2c0,c2f0,c3c0,c3f0,tnow,&s1a,&s3a,&s1b,&s3b);
                    snap_t sa=mk_snap(tnow,s1a,s3a,cut_a,t0eid,c2c0,c2f0);
                    snap_t sb=mk_snap(tnow,s1b,s3b,cut_b,t0eid,c3c0,c3f0);
                    EVAL_PAIR_BI(sa,s1a,s3a,t0eid,cut_a,0u,sb,s1b,s3b,t0eid,cut_b,cut_a,ra,nra,1u);
                }
            }

            /* Commit */
            if (bkey.valid){
                plan_t pea=plan_from_snap(&bsna,0u,btok0_a,bs1a,bs3a);
                pea.eid=beid_a; pea.ntok=bntok_a;
                plan_t peb=plan_from_snap(&bsnb,1u,btok0_b,bs1b,bs3b);
                peb.eid=beid_b; peb.ntok=bntok_b;
                if (*n_plan<MOE_MAX_TASKS) plan[(*n_plan)++]=pea;
                if (*n_plan<MOE_MAX_TASKS) plan[(*n_plan)++]=peb;
                c2=bsna; c3=bsnb;
                nr=0; for(uint8_t ri=0;ri<bnr;ri++) rem[nr++]=brem[ri];
            } else {
                /* Fallback: assign top0 solo to C2 */
                snap_t sf=mk_snap(tnow,2u,2u,t0ntok,t0eid,c2c0,c2f0);
                plan_t pf=plan_from_snap(&sf,0u,0u,2u,2u);
                if (*n_plan<MOE_MAX_TASKS) plan[(*n_plan)++]=pf;
                c2=sf; rem_remove(rem,&nr,0u);
                /* No ghost rollback needed: PF_EID_GHOST in C3 stays valid;
                 * the next expert assigned to C3 will benefit from skip_s1. */
            }
            continue;
        }

        /* ── not_both_idle ───────────────────────────────────────────────── */
        {
            int idle_ci=(t2<t3)?0:1;
            snap_t *idle_sn=(idle_ci==0)?&c2:&c3;
            snap_t *busy_sn=(idle_ci==0)?&c3:&c2;
            uint32_t idle_t=(idle_ci==0)?t2:t3;

            /* Analytical: 3 time pts (idle_t + busy DMA hi-endpoints) x 1 shape */
            uint32_t tpts[3]; int ntp=0;
            tpts[ntp++]=idle_t;
            { seg_t bsegs[4]; int nbsegs=0; snap_segs(busy_sn,bsegs,&nbsegs);
              for (int bi=0;bi<nbsegs&&ntp<3;bi++){
                uint32_t ep=bsegs[bi].hi;
                if (ep>idle_t){
                    int dup=0; for(int ti=0;ti<ntp;ti++) if(tpts[ti]==ep){dup=1;break;}
                    if (!dup) tpts[ntp++]=ep;
                }
              }
            }

            uint32_t best_ms=INF_CC;
            snap_t best_nb; uint8_t best_ns1=2,best_ns3=2; int best_nbv=0;

            for (int ti=0;ti<ntp;ti++){
                uint32_t tst=tpts[ti];
                uint8_t cc=(uint8_t)swiglu_hit(t0eid,idle_sn,tst);
                uint8_t cf=(uint8_t)down_hit(t0eid,idle_sn,tst);
                uint8_t s1=2u, s3=2u;  /* ShapeC: fastest; BW conflict handled by next tpt */
                snap_t sn=mk_snap(tst,s1,s3,t0ntok,t0eid,cc,cf);
                /* S2 pf attempt (latest valid position) */
                if (sn.bw_s3>0u && kTd3[s3]<=sn.s2_end-sn.task_start){
                    uint32_t hi=sn.s2_end-kTd3[s3];
                    snap_t cand=apply_s2pf(sn,s3,hi);
                    if (cand.s2pf_start>=0){
                        int ok2=(idle_ci==0)?bw_ok(&cand,busy_sn):bw_ok(busy_sn,&cand);
                        if (ok2) sn=cand;
                    }
                }
                int ok=(idle_ci==0)?bw_ok(&sn,busy_sn):bw_ok(busy_sn,&sn);
                if (!ok) continue;
                uint32_t ms=(sn.task_end>busy_sn->task_end)?sn.task_end:busy_sn->task_end;
                if (ms<best_ms){ best_ms=ms; best_nb=sn; best_ns1=s1; best_ns3=s3; best_nbv=1; }
            }

            rem_remove(rem,&nr,0u);
            if (best_nbv){
                plan_t pe=plan_from_snap(&best_nb,(uint8_t)idle_ci,0u,best_ns1,best_ns3);
                if (*n_plan<MOE_MAX_TASKS) plan[(*n_plan)++]=pe;
                if (idle_ci==0) c2=best_nb; else c3=best_nb;
            } else {
                uint8_t cch=(idle_ci==0)?c2c0:c3c0, cfh=(idle_ci==0)?c2f0:c3f0;
                snap_t sf=mk_snap(idle_t,2u,2u,t0ntok,t0eid,cch,cfh);
                plan_t pf=plan_from_snap(&sf,(uint8_t)idle_ci,0u,2u,2u);
                if (*n_plan<MOE_MAX_TASKS) plan[(*n_plan)++]=pf;
                if (idle_ci==0) c2=sf; else c3=sf;
            }
            /* No ghost rollback needed: PF_EID_GHOST in the busy cluster stays
             * valid; the next expert assigned to that cluster will benefit from
             * skip_s1 (lower_plan will fill in the actual expert_id for S4_PREFETCH). */
            continue;
        }
    } /* while (nr > 0) */

    return (c2.task_end>c3.task_end)?c2.task_end:c3.task_end;
}

/* =========================================================================
 * Lowering: plan[] → moe_schedule_t
 * ========================================================================= */
static moe_status_t lower_plan(const plan_t *plan, uint8_t n_plan,
                                const moe_request_t *req, moe_schedule_t *out)
{
    out->n_tasks=0; out->n_dma_ops=0;

    for (uint8_t pi=0;pi<n_plan;pi++){
        const plan_t *p=&plan[pi];
        if (out->n_tasks>=MOE_MAX_TASKS) return MOE_ERR_OVERFLOW;

        int ci=(int)p->cluster;
        moe_cluster_t cl=(ci==0)?MOE_CLUSTER_C2:MOE_CLUSTER_C3;
        moe_shape_t sh1=(moe_shape_t)p->shape_s1;
        moe_shape_t sh3=(moe_shape_t)p->shape_s3;
        uint32_t ntok_u=p->ntok;
        uint8_t  skip_s1=p->skip_s1, skip_s3=p->skip_s3, has_s2pf=p->has_s2pf;
        uint32_t dma1_end=skip_s1?p->est_start:p->est_dma1_end;

        moe_task_t *tk=&out->tasks[out->n_tasks];
        /* (dma_slots[] removed from compact moe_task_t) */

        tk->cluster         =cl;
        tk->expert_id       =(uint16_t)p->eid;
        tk->token_start_rank=p->tok_start;
        tk->ntokens         =(uint16_t)ntok_u;
        tk->shape_s1        =sh1;
        tk->shape_s3        =sh3;
        /* bw_s1/bw_s3 removed: derivable from shape+skip if needed */
        tk->dma_s1          =skip_s1?MOE_DMA_NONE:(kAlloc[p->shape_s1]>=128u?MOE_DMA_BOTH:MOE_DMA_IDMA);
        tk->dma_s3          =skip_s3?MOE_DMA_NONE:(kAlloc[p->shape_s3]>=128u?MOE_DMA_BOTH:MOE_DMA_XDMA);
        tk->skip_s1         =skip_s1;
        tk->skip_s3         =skip_s3;

        if (skip_s1){
            /* S2 handles all tokens: batch count = ceil(ntok / meshRow) */
            uint32_t b2=(ntok_u+kMdim[p->shape_s1]-1u)/kMdim[p->shape_s1];
            tk->m_s2_exec=b2; tk->skip_s2=0u;
        } else {
            /* S2 handles tail tokens after S1 block (kMdim = one S1 batch) */
            uint32_t tail=(ntok_u>kMdim[p->shape_s1])?(ntok_u-kMdim[p->shape_s1]):0u;
            uint32_t b2=(tail+kMdim[p->shape_s1]-1u)/kMdim[p->shape_s1];
            tk->m_s2_exec=b2; tk->skip_s2=(b2==0u)?1u:0u;
        }
        if (skip_s3){
            /* S4 handles all tokens: batch count = ceil(ntok / meshRow) */
            uint32_t b4=(ntok_u+kMdim[p->shape_s3]-1u)/kMdim[p->shape_s3];
            tk->m_s4_exec=b4; tk->skip_s4=0u;
        } else {
            /* S4 handles tail tokens after S3 block */
            uint32_t tail4=(ntok_u>kMdim[p->shape_s3])?(ntok_u-kMdim[p->shape_s3]):0u;
            uint32_t b4=(tail4+kMdim[p->shape_s3]-1u)/kMdim[p->shape_s3];
            tk->m_s4_exec=b4; tk->skip_s4=(b4==0u)?1u:0u;
        }
        /* prefetch_eid/dma_slots/est_start_cc/est_end_cc removed from compact moe_task_t */

        /* S1 DMA: compact op (task_idx, expert_id, kind, dma only) */
        if (!skip_s1){
            if (out->n_dma_ops>=MOE_MAX_DMA_OPS) return MOE_ERR_OVERFLOW;
            moe_dma_op_t *op=&out->dma_ops[out->n_dma_ops++];
            op->task_idx=(uint16_t)out->n_tasks;
            op->expert_id=p->eid;
            op->kind=MOE_DMA_OP_S1;
            op->dma=(kAlloc[p->shape_s1]>=128u)?MOE_DMA_BOTH:MOE_DMA_IDMA;
            /* cluster/weight/shape/alloc_bw/start_cc/end_cc removed */
        }

        /* S3 / S2-prefetch DMA: compact op */
        if (!skip_s3 || has_s2pf){
            if (out->n_dma_ops>=MOE_MAX_DMA_OPS) return MOE_ERR_OVERFLOW;
            moe_dma_op_t *op=&out->dma_ops[out->n_dma_ops++];
            op->task_idx=(uint16_t)out->n_tasks;
            op->expert_id=p->eid;
            op->dma=(kAlloc[p->shape_s3]>=128u)?MOE_DMA_BOTH:MOE_DMA_XDMA;
            op->kind=has_s2pf?MOE_DMA_OP_S2_PREFETCH:MOE_DMA_OP_S3;
            /* cluster/weight/shape/alloc_bw/start_cc/end_cc removed */
        }

        /* S4 prefetch: next expert's gate/up via iDMA during S4 window.
         * Applies regardless of skip_s1: when skip_s1=1, iDMA is idle the
         * entire task, and dma1_end==est_start which is always <= s4_start. */
        {
            int16_t next_eid=-1;
            for (uint8_t pj=pi+1u;pj<n_plan;pj++){
                if (plan[pj].cluster==p->cluster){ next_eid=plan[pj].eid; break; }
            }
            if (next_eid>=0){
                uint8_t nc=0;
                if (ci==0 && req->cache_eid_c2==next_eid) nc=1;
                if (ci==1 && req->cache_eid_c3==next_eid) nc=1;
                if (!nc){
                    uint32_t s4s=p->est_s4_start;
                    uint32_t pfd=kTd1[0]; /* ShapeA gate+up DMA = 45056 cc */
                    if (dma1_end<=s4s && (s4s+pfd)<=p->est_end
                            && out->n_dma_ops<MOE_MAX_DMA_OPS){
                        moe_dma_op_t *op=&out->dma_ops[out->n_dma_ops++];
                        op->task_idx=(uint16_t)out->n_tasks;
                        op->expert_id=next_eid;
                        op->kind=MOE_DMA_OP_S4_PREFETCH;
                        op->dma=MOE_DMA_IDMA;
                    }
                }
            }
        }
        out->n_tasks++;
    }
    /* est_makespan_cc removed from compact moe_schedule_t */
    return MOE_OK;
}

/* =========================================================================
 * Entry point
 * ========================================================================= */
moe_status_t moe_schedule(const moe_request_t *req, moe_schedule_t *out)
{
    if (!req || !out) return MOE_ERR_BAD_INPUT;
    uint16_t ne=req->n_experts;
    if (ne==0u||ne>MOE_MAX_EXPERTS) return MOE_ERR_BAD_INPUT;
    for (uint16_t i=0u;i<ne;i++) if (req->experts[i].ntokens==0u) return MOE_ERR_BAD_INPUT;

    /* Static plan buffer avoids large stack allocation (128*36=4.6KB at 64 experts).
     * Safe for single-threaded CVA6 host execution. */
    static plan_t plan[MOE_MAX_TASKS];
    uint8_t n_plan=0;
    (void)moe_plan(req,plan,&n_plan); /* makespan returned but not stored in compact output */
    return lower_plan(plan,n_plan,req,out);
}
