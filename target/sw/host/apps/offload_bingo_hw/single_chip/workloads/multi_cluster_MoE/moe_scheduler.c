/* Host mirror of Idea_Model/scheduler_rtl_distilled_policy.py.
 *
 * Normative direction: Python -> C.  This file does not derive policy
 * behavior from the RTL.  All internal time values are exact 11,264-cycle
 * lattice ticks; the public task ABI remains cycle-free.
 *
 * Fixed policy:
 *   Top5 + Bottom1 observation window;
 *   28 hard-wired physical profiles;
 *   per-logical-action physical local reduction;
 *   one bounded continuation comparator;
 *   concrete, target-aware S4 prefetch local reduction.
 *
 * Pure C99, no floating point, allocation, recursion, or OS dependency.
 */
#include "moe_scheduler.h"

#if !defined(MOE_ENABLE_HW_SCHEDULER)

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ARRAY_LEN(a) ((uint32_t)(sizeof(a) / sizeof((a)[0])))
#define MIN_U(a,b) ((a) < (b) ? (a) : (b))
#define MAX_U(a,b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV2(x) (((x) + 1u) >> 1u)
#define CEIL_DIV4(x) (((x) + 3u) >> 2u)

enum { A=0, B=1, C=2 };
enum { N=MOE_DMA_NONE, I=MOE_DMA_IDMA, X=MOE_DMA_XDMA, D=MOE_DMA_BOTH };
enum { TERM=0, SYNC=1, ONE=2 };
enum { SINGLE=0, PAIR=1, SPLIT=2 };
enum { T0=0, T1=1, T2=2, T3=3, T4=4, B0=5 };

enum {
    PROFILE_COUNT = 28,
    LOGICAL_MAX = 6,
    PHYSICAL_MAX = 18,
    START_MAX = 32,
    S4PF_MIN_REMAINING = 9,
    S4PF_MIN_GAIN_TICKS = 1
};

/* Shape compute duration and tile width in scheduler ticks. */
static const uint8_t kS1Ticks[3] = {8u,4u,2u};
static const uint8_t kS3Ticks[3] = {4u,2u,1u};
static const uint8_t kMdim[3] = {8u,4u,2u};

static uint32_t dma_s1_ticks(uint8_t dma)
{ return dma==N ? 0u : (dma==D ? 2u : 4u); }
static uint32_t dma_s3_ticks(uint8_t dma)
{ return dma==N ? 0u : (dma==D ? 1u : 2u); }
static uint32_t best_s2_ticks(uint32_t ntok){ return CEIL_DIV2(ntok)*2u; }
static uint32_t best_s4_ticks(uint32_t ntok){ return CEIL_DIV2(ntok); }

typedef struct {
    uint32_t task_start, task_end, dma1_end, s2_end, dma3_end, compute_end;
    int32_t s2pf_end;
    int32_t cache_end;
    int16_t cur_eid, cache_eid;
    uint8_t dma_s1, dma_s3, s2pf_dma, pf_dma;
    uint8_t cache_valid, cache_full;
} snap_t;

typedef struct { int16_t eid; uint16_t ntok; } rem_t;

typedef struct {
    uint8_t slot, mode, family, sel_a, sel_b, logical_id;
    uint8_t split_balanced, active2, active3;
    uint8_t s1[2], s3[2], dma1[2], dma3[2], s2pf[2];
    uint8_t expect_s1_hit[2], expect_s3_ready[2];
} profile_t;

#define P(slot,mode,family,sa,sb,lid,bal,a2,a3,s12,s32,s13,s33,d12,d32,p2,d13,d33,p3,h12,h32,h13,h33) \
 {slot,mode,family,sa,sb,lid,bal,a2,a3,{s12,s13},{s32,s33},{d12,d13},{d32,d33},{p2,p3},{h12,h13},{h32,h33}}

/* Generated mechanically from scheduler_rtl_distilled_profiles.COMPILED_PROFILES.
 * The entries are hard-wired decode cases, not runtime-programmable storage. */
static const profile_t kProfiles[PROFILE_COUNT] = {
    P(0,ONE,SINGLE,B0,B0,0,0,0,1,C,C,C,C,N,N,N,D,D,N,0,0,0,0),
    P(1,ONE,SINGLE,B0,B0,0,0,1,0,C,C,C,C,D,D,N,N,N,N,0,0,0,0),
    P(2,ONE,SINGLE,T0,T0,1,0,1,0,B,B,C,C,D,N,D,N,N,N,0,1,0,0),
    P(3,ONE,SINGLE,T0,T0,1,0,0,1,C,C,B,B,N,N,N,D,N,D,0,0,0,1),
    P(4,ONE,SINGLE,T3,T3,2,0,1,0,B,B,C,C,D,N,D,N,N,N,0,1,0,0),
    P(5,ONE,SINGLE,T3,T3,2,0,0,1,C,C,B,B,N,N,N,D,N,D,0,0,0,1),
    P(6,SYNC,PAIR,B0,T0,0,0,1,1,A,B,B,B,I,N,I,X,X,N,0,1,0,0),
    P(7,SYNC,PAIR,T0,T1,1,0,1,1,B,B,B,B,I,I,N,X,X,N,0,0,0,0),
    P(8,SYNC,PAIR,T0,T1,1,0,1,1,B,B,B,B,I,N,I,X,X,N,0,1,0,0),
    P(9,SYNC,PAIR,T0,T4,2,0,1,1,A,B,B,B,I,N,I,X,X,N,0,1,0,0),
    P(10,SYNC,PAIR,T1,T2,3,0,1,1,B,B,B,B,I,N,I,X,N,X,0,1,0,1),
    P(11,SYNC,PAIR,T2,T3,4,0,1,1,B,B,B,B,I,N,I,X,N,X,0,1,0,1),
    P(12,TERM,SINGLE,T0,T0,0,0,1,0,C,C,C,C,D,D,N,N,N,N,0,0,0,0),
    P(13,TERM,SINGLE,T0,T0,0,0,0,1,C,C,C,C,N,N,N,D,D,N,0,0,0,0),
    P(14,TERM,SPLIT,T0,T0,1,1,1,1,B,B,B,B,I,I,N,X,X,N,0,0,0,0),
    P(15,ONE,SINGLE,T0,T0,1,0,1,0,A,B,C,C,I,I,N,N,N,N,0,0,0,0),
    P(16,ONE,SINGLE,T0,T0,1,0,1,0,B,B,C,C,I,I,N,N,N,N,0,0,0,0),
    P(17,ONE,SINGLE,T0,T0,1,0,0,1,C,C,B,B,N,N,N,X,X,N,0,0,0,0),
    P(18,ONE,SINGLE,T0,T0,1,0,0,1,C,C,C,C,N,N,N,D,D,N,0,0,0,0),
    P(19,ONE,SINGLE,T0,T0,1,0,1,0,C,C,C,C,D,D,N,N,N,N,0,0,0,0),
    P(20,ONE,SINGLE,T0,T0,1,0,1,0,B,C,C,C,I,D,N,N,N,N,0,0,0,0),
    P(21,ONE,SINGLE,T0,T0,1,0,0,1,C,C,A,B,N,N,N,X,X,N,0,0,0,0),
    P(22,TERM,SINGLE,T0,T0,0,0,0,1,C,C,B,B,N,N,N,X,X,N,0,0,0,0),
    P(23,ONE,SINGLE,T0,T0,1,0,0,1,C,C,B,C,N,N,N,X,D,N,0,0,0,0),
    P(24,ONE,SINGLE,T0,T0,1,0,1,0,C,B,C,C,D,I,N,N,N,N,0,0,0,0),
    P(25,TERM,SINGLE,T0,T0,0,0,1,0,B,C,C,C,I,D,N,N,N,N,0,0,0,0),
    P(26,SYNC,SPLIT,T0,T0,5,0,1,1,A,B,A,B,I,I,N,X,X,N,0,0,0,0),
    P(27,SYNC,PAIR,T0,T1,1,0,1,1,C,C,C,C,N,N,N,D,D,N,1,1,0,0),
};
/* CandidateProfile dataclass ordering used by runtime_profile_bank(). */
static const uint8_t kProfileEvalOrder[PROFILE_COUNT] = {
    1,0,15,2,16,20,24,19,21,3,17,23,18,4,5,
    6,7,8,27,9,10,11,26,
    25,12,22,13,14
};
#undef P

typedef struct {
    uint8_t active[2];
    int16_t eid[2];
    uint16_t ntok[2], tok_start[2];
    uint8_t s1[2], s3[2], dma1[2], dma3[2], s2pf[2];
    uint8_t s1_hit[2], s3_hit[2];
    uint32_t start[2];
    uint8_t mode, family, logical_id, profile_slot;
    uint8_t lsel_a, lsel_b, split_balanced;
    uint8_t s4pf_dma[2];
    snap_t child[2];
    uint32_t child_work;
} cand_t;

typedef struct {
    snap_t c[2];
    rem_t rem[MOE_MAX_EXPERTS];
    uint8_t nr;
    uint32_t token_sum, block_sum;
    uint8_t odd_count;
    uint8_t hist[4];
    uint32_t parent_bound_half;
    uint32_t cluster_work;
} state_t;

typedef struct {
    int16_t eid;
    uint16_t ntok, tok_start;
    uint8_t cluster, shape_s1, shape_s3;
    uint8_t dma_s1, dma_s3, s2pf_dma;
    uint8_t skip_s1, skip_s3;
    uint8_t s4pf_dma;
    int16_t s4pf_eid;
} plan_t;

/* The host invokes one scheduler instance at a time.  Reuse one internal
 * plan buffer instead of placing a large candidate-independent array on the
 * CVA6 stack or keeping a second compact-plan copy in BSS. */
static plan_t g_plan_scratch[MOE_MAX_TASKS];

static snap_t snap_idle(int16_t cache_eid)
{
    snap_t s;
    memset(&s,0,sizeof(s));
    s.cur_eid=-1; s.cache_eid=cache_eid;
    s.s2pf_end=-1; s.cache_end=(cache_eid>=0)?0:-1;
    s.cache_valid=(uint8_t)(cache_eid>=0); s.cache_full=s.cache_valid;
    return s;
}

static int s1_hit(const snap_t *s, int16_t eid, uint32_t start)
{ return s->cache_valid && s->cache_eid==eid && s->cache_end>=0 && (uint32_t)s->cache_end<=start; }
static int s3_hit(const snap_t *s, int16_t eid, uint32_t start)
{ return s1_hit(s,eid,start) && s->cache_full; }
static int reserved_eid(const snap_t *s)
{ return (s->cur_eid>=0 && s->cache_valid && !s->cache_full)?s->cache_eid:-1; }

static snap_t make_task(uint32_t start, uint8_t sh1, uint8_t sh3,
                        uint16_t ntok, int16_t eid, uint8_t hit1, uint8_t hit3,
                        uint8_t dma1, uint8_t dma3, uint8_t s2pf)
{
    snap_t s;
    uint32_t d1=hit1?0u:dma_s1_ticks(dma1);
    uint32_t s1phase=hit1?0u:MAX_U(kS1Ticks[sh1],d1);
    uint32_t tail2=hit1?(uint32_t)ntok:
        ((uint32_t)ntok>kMdim[sh1]?(uint32_t)ntok-kMdim[sh1]:0u);
    uint32_t ready3;
    memset(&s,0,sizeof(s));
    s.cur_eid=-1;s.cache_eid=-1;s.s2pf_end=-1;s.cache_end=-1;
    s.task_start=start; s.cur_eid=eid; s.dma_s1=hit1?N:dma1;
    s.dma1_end=start+d1; s.s2_end=start+s1phase+best_s2_ticks(tail2);
    if (hit3) s2pf=N;
    s.s2pf_dma=s2pf;
    if (s2pf!=N) {
        s.s2pf_end=(int32_t)(s.dma1_end+dma_s3_ticks(s2pf));
        ready3=MAX_U(s.s2_end,(uint32_t)s.s2pf_end);
        s.dma_s3=N; s.dma3_end=ready3;
        s.compute_end=ready3+best_s4_ticks(ntok);
    } else if (hit3) {
        s.s2pf_end=-1; s.dma_s3=N; s.dma3_end=s.s2_end;
        s.compute_end=s.s2_end+best_s4_ticks(ntok);
    } else {
        uint32_t d3=dma_s3_ticks(dma3);
        uint32_t s3phase=MAX_U(kS3Ticks[sh3],d3);
        uint32_t tail4=(uint32_t)ntok>kMdim[sh3]?
            (uint32_t)ntok-kMdim[sh3]:0u;
        s.s2pf_end=-1; s.dma_s3=dma3; s.dma3_end=s.s2_end+d3;
        s.compute_end=s.s2_end+s3phase+best_s4_ticks(tail4);
    }
    s.task_end=s.compute_end;
    return s;
}

typedef struct { uint32_t lo,hi; uint8_t dma; } interval_t;

static uint8_t collect_intervals(const snap_t *s, interval_t out[4])
{
    uint8_t n=0;
    if (s->cur_eid>=0 && s->dma_s1!=N && s->task_start<s->dma1_end)
        out[n++]=(interval_t){s->task_start,s->dma1_end,s->dma_s1};
    if (s->s2pf_dma!=N && s->s2pf_end>=0 && s->dma1_end<(uint32_t)s->s2pf_end)
        out[n++]=(interval_t){s->dma1_end,(uint32_t)s->s2pf_end,s->s2pf_dma};
    if (s->cur_eid>=0 && s->dma_s3!=N && s->s2_end<s->dma3_end)
        out[n++]=(interval_t){s->s2_end,s->dma3_end,s->dma_s3};
    if (s->cache_valid && !s->cache_full && s->cache_end>=0 && s->dma3_end<(uint32_t)s->cache_end)
        out[n++]=(interval_t){s->dma3_end,(uint32_t)s->cache_end,s->pf_dma};
    return n;
}

static int bw_feasible(const snap_t *a, const snap_t *b)
{
    interval_t x[4],y[4]; uint8_t nx=collect_intervals(a,x),ny=collect_intervals(b,y);
    uint8_t i,j;
    for(i=0;i<nx;i++) for(j=0;j<ny;j++)
        if (MAX_U(x[i].lo,y[j].lo)<MIN_U(x[i].hi,y[j].hi) && (x[i].dma&y[j].dma)) return 0;
    for(i=0;i<nx;i++) for(j=(uint8_t)(i+1u);j<nx;j++)
        if (MAX_U(x[i].lo,x[j].lo)<MIN_U(x[i].hi,x[j].hi) && (x[i].dma&x[j].dma)) return 0;
    for(i=0;i<ny;i++) for(j=(uint8_t)(i+1u);j<ny;j++)
        if (MAX_U(y[i].lo,y[j].lo)<MIN_U(y[i].hi,y[j].hi) && (y[i].dma&y[j].dma)) return 0;
    return 1;
}

/* Copy a committed snapshot and add a concrete S1-only cache interval. */
static snap_t with_prefetch(const snap_t *src, int16_t eid, uint8_t dma)
{
    snap_t s=*src;
    uint32_t end=s.dma3_end+dma_s1_ticks(dma);
    s.cache_valid=1u; s.cache_full=0u; s.cache_eid=eid; s.cache_end=(int32_t)end;
    s.pf_dma=dma;
    s.task_end=MAX_U(s.task_end,end);
    return s;
}

/* End of low-level tick-domain primitives. */

static uint8_t state_mode(const state_t *s)
{
    if (s->nr==1u) return TERM;
    return s->c[0].task_end==s->c[1].task_end ? SYNC : ONE;
}

static int rem_index_by_eid(const state_t *s, int16_t eid)
{
    uint8_t i;
    for(i=0;i<s->nr;i++) if(s->rem[i].eid==eid) return (int)i;
    return -1;
}

static int resolve_selector(const state_t *s, uint8_t sel)
{
    if(sel<=T4) return sel<s->nr ? (int)sel : -1;
    if(sel==B0) return s->nr ? (int)s->nr-1 : -1;
    return -1;
}

static void add_unique_u32(uint32_t *v, uint8_t *n, uint32_t x)
{
    uint8_t i;
    for(i=0;i<*n;i++) if(v[i]==x) return;
    if(*n<START_MAX) v[(*n)++]=x;
}

static void sort_u32(uint32_t *v, uint8_t n)
{
    uint8_t i;
    for(i=1;i<n;i++){
        uint32_t key=v[i]; int j=(int)i-1;
        while(j>=0 && v[j]>key){v[j+1]=v[j];j--;}
        v[j+1]=key;
    }
}

/* Exact translation of four_stage_scheduler._start_candidates for the
 * concrete profile.  It enumerates only event-aligned legal start regions. */
static uint8_t start_candidates(const snap_t *own, const snap_t *peer,
                                uint16_t ntok, uint8_t sh1, uint8_t sh3,
                                uint8_t dma1, uint8_t dma3, uint8_t profile_s2pf,
                                uint8_t hit1, uint8_t hit3, uint32_t out[START_MAX])
{
    uint32_t releases[4], offsets[16];
    uint8_t nr=0,no=0,ns=0,i,j;
    uint32_t history_floor=peer->cur_eid>=0?peer->task_start:own->task_end;
    uint32_t floor=MAX_U(own->task_end,history_floor);
    add_unique_u32(out,&ns,floor);
    add_unique_u32(out,&ns,MAX_U(floor,peer->task_end));
    if(peer->cur_eid>=0 && peer->dma_s1!=N) releases[nr++]=peer->dma1_end;
    if(peer->s2pf_dma!=N && peer->s2pf_end>=0) releases[nr++]=(uint32_t)peer->s2pf_end;
    if(peer->cur_eid>=0 && peer->dma_s3!=N) releases[nr++]=peer->dma3_end;
    if(peer->cache_valid && !peer->cache_full && peer->cache_end>=0 && peer->pf_dma!=N)
        releases[nr++]=(uint32_t)peer->cache_end;

    /* raw plus all three discovery S2PF bindings when S3 is not cached */
    for(i=0;i<(uint8_t)(hit3?1u:4u);i++){
        uint8_t trial_pf=(i==0u)?N:(i==1u?I:(i==2u?X:D));
        snap_t local=make_task(0u,sh1,sh3,ntok,0,hit1,hit3,dma1,dma3,trial_pf);
        if(local.dma_s1!=N) add_unique_u32(offsets,&no,local.task_start);
        if(local.s2pf_dma!=N) add_unique_u32(offsets,&no,local.dma1_end);
        if(local.dma_s3!=N) add_unique_u32(offsets,&no,local.s2_end);
        add_unique_u32(offsets,&no,local.dma3_end);
    }
    (void)profile_s2pf;
    for(i=0;i<nr;i++) for(j=0;j<no;j++) if(releases[i]>=offsets[j]){
        uint32_t st=releases[i]-offsets[j];
        if(st>=floor && st<=peer->task_end) add_unique_u32(out,&ns,st);
    }
    sort_u32(out,ns);
    return ns;
}

static int single_cluster_legal(const state_t *s, uint8_t cl)
{
    uint32_t t2=s->c[0].task_end,t3=s->c[1].task_end;
    if(t2<t3) return cl==0u || (cl==1u && reserved_eid(&s->c[1])>=0);
    if(t3<t2) return cl==1u || (cl==0u && reserved_eid(&s->c[0])>=0);
    if(cl==0u) return 1;
    return memcmp(&s->c[0],&s->c[1],sizeof(snap_t))!=0;
}

static int build_candidate(const state_t *s, const profile_t *p,
                           int ia, int ib, uint8_t swap, uint32_t start,
                           uint8_t force_hit2, uint8_t force_hit3, cand_t *out)
{
    uint8_t cl;
    int indices[2]={-1,-1};
    cand_t c;
    memset(&c,0,sizeof(c));
    c.mode=p->mode; c.family=p->family; c.logical_id=p->logical_id;
    c.profile_slot=p->slot; c.eid[0]=c.eid[1]=-1;
    c.split_balanced=p->split_balanced;
    c.child[0]=s->c[0]; c.child[1]=s->c[1];
    c.child_work=s->cluster_work;

    if(p->family==SINGLE){
        cl=p->active3?1u:0u; indices[cl]=ia;
    } else if(p->family==PAIR){
        indices[0]=swap?ib:ia; indices[1]=swap?ia:ib;
    } else {
        indices[0]=ia; indices[1]=ia;
    }
    /* Canonical logical labels: Top aliases win when B0 overlaps Top5. */
    c.lsel_a=(uint8_t)((uint32_t)ia<MIN_U(5u,s->nr)?(1+ia):0);
    c.lsel_b=(uint8_t)((uint32_t)ib<MIN_U(5u,s->nr)?(1+ib):0);
    if(c.lsel_b<c.lsel_a){ uint8_t z=c.lsel_a; c.lsel_a=c.lsel_b; c.lsel_b=z; }

    for(cl=0;cl<2u;cl++) if(indices[cl]>=0){
        rem_t r=s->rem[indices[cl]];
        uint16_t ntok=r.ntok,tok_start=0u;
        uint8_t hit1,hit3,dma1,dma3,pf;
        if(p->family==SPLIT){
            uint16_t left=(uint16_t)(r.ntok/2u);
            ntok=cl==0u?left:(uint16_t)(r.ntok-left);
            tok_start=cl==0u?0u:left;
        }
        hit1=(uint8_t)s1_hit(&s->c[cl],r.eid,start);
        hit3=(uint8_t)s3_hit(&s->c[cl],r.eid,start);
        if((cl==0u&&force_hit2)||(cl==1u&&force_hit3)) hit1=1u;
        if(p->expect_s1_hit[cl] && !hit1) return 0;
        if(p->expect_s3_ready[cl] && p->s2pf[cl]==N && !hit3) return 0;
        dma1=hit1?N:p->dma1[cl];
        pf=hit3?N:p->s2pf[cl];
        dma3=(hit3||pf!=N)?N:p->dma3[cl];
        if(!hit1 && dma1==N) return 0;
        if(!hit3 && pf==N && dma3==N) return 0;
        c.active[cl]=1u; c.eid[cl]=r.eid; c.ntok[cl]=ntok;
        c.tok_start[cl]=tok_start; c.s1[cl]=p->s1[cl]; c.s3[cl]=p->s3[cl];
        c.dma1[cl]=dma1; c.dma3[cl]=dma3; c.s2pf[cl]=pf;
        c.s1_hit[cl]=hit1; c.s3_hit[cl]=hit3; c.start[cl]=start;
        c.child[cl]=make_task(start,p->s1[cl],p->s3[cl],ntok,r.eid,
                              hit1,hit3,dma1,dma3,pf);
        c.child_work += c.child[cl].task_end-c.child[cl].task_start;
    }
    if(!bw_feasible(&c.child[0],&c.child[1])) return 0;
    *out=c;
    return 1;
}

typedef void (*candidate_sink_t)(const state_t *, const profile_t *, const cand_t *, void *);

static void materialize_profile(const state_t *s, const profile_t *p,
                                uint8_t force_hit2, uint8_t force_hit3,
                                candidate_sink_t sink, void *opaque)
{
    int ia,ib;
    uint32_t now=MAX_U(s->c[0].task_end,s->c[1].task_end);
    if(p->mode!=state_mode(s)) return;
    ia=resolve_selector(s,p->sel_a); ib=resolve_selector(s,p->sel_b);
    if(ia<0 || ib<0) return;

    if(p->family==PAIR){
        uint8_t sw;
        if(ia==ib) return;
        for(sw=0;sw<2u;sw++){
            int e2=sw?ib:ia,e3=sw?ia:ib;
            int r2=reserved_eid(&s->c[0]),r3=reserved_eid(&s->c[1]);
            cand_t c;
            if((r2>=0&&r2!=s->rem[e2].eid)||(r3>=0&&r3!=s->rem[e3].eid)) continue;
            if(build_candidate(s,p,ia,ib,sw,now,force_hit2,force_hit3,&c)) sink(s,p,&c,opaque);
        }
        return;
    }
    if(p->family==SPLIT){
        cand_t c;
        int r2=reserved_eid(&s->c[0]),r3=reserved_eid(&s->c[1]);
        if(s->rem[ia].ntok<2u) return;
        if(!p->split_balanced && (s->rem[ia].ntok&1u)) return;
        if((r2>=0&&r2!=s->rem[ia].eid)||(r3>=0&&r3!=s->rem[ia].eid)) return;
        if(build_candidate(s,p,ia,ia,0u,now,force_hit2,force_hit3,&c)) sink(s,p,&c,opaque);
        return;
    }
    {
        uint8_t cl=p->active3?1u:0u;
        const snap_t *own=&s->c[cl],*peer=&s->c[1u-cl];
        int own_r=reserved_eid(own),peer_r=reserved_eid(peer);
        uint8_t hit1=(uint8_t)s1_hit(own,s->rem[ia].eid,own->task_end);
        uint8_t hit3=(uint8_t)s3_hit(own,s->rem[ia].eid,own->task_end);
        uint32_t starts[START_MAX]; uint8_t ns,k;
        if(!single_cluster_legal(s,cl)) return;
        if((own_r>=0&&own_r!=s->rem[ia].eid)||peer_r==s->rem[ia].eid) return;
        if((cl==0u&&force_hit2)||(cl==1u&&force_hit3)) hit1=1u;
        ns=start_candidates(own,peer,s->rem[ia].ntok,p->s1[cl],p->s3[cl],
                            hit1?N:p->dma1[cl],hit3?N:p->dma3[cl],
                            hit3?N:p->s2pf[cl],hit1,hit3,starts);
        for(k=0;k<ns;k++){
            cand_t c;
            /* build_candidate would rediscover a later cache hit.  Python
             * freezes SINGLE residency at own.task_end, so mask it explicitly. */
            state_t frozen=*s;
            if(!hit1 && frozen.c[cl].cache_valid && frozen.c[cl].cache_end>=0 &&
               (uint32_t)frozen.c[cl].cache_end>own->task_end)
                frozen.c[cl].cache_valid=0u;
            if(build_candidate(&frozen,p,ia,ia,0u,starts[k],force_hit2,force_hit3,&c))
                sink(s,p,&c,opaque);
        }
    }
}

static uint8_t cand_s2pf_count(const cand_t *c)
{ return (uint8_t)((c->s2pf[0]!=N)+(c->s2pf[1]!=N)); }
static uint8_t cand_s4pf_count(const cand_t *c)
{ return (uint8_t)((c->s4pf_dma[0]!=N)+(c->s4pf_dma[1]!=N)); }
static uint32_t cand_max_end(const cand_t *c)
{ return MAX_U(c->child[0].task_end,c->child[1].task_end); }
static uint32_t cand_sum_end(const cand_t *c)
{ return c->child[0].task_end+c->child[1].task_end; }
static uint32_t cand_latest_start(const cand_t *c)
{
    uint32_t v=0u;
    if(c->active[0]) v=c->start[0];
    if(c->active[1]) v=MAX_U(v,c->start[1]);
    return v;
}

static uint8_t runtime_dma_order(uint8_t dma)
{
    /* Python PhysicalProfile string ordering: BOTH, IDMA, NONE, XDMA. */
    return dma==D?0u:(dma==I?1u:(dma==N?2u:3u));
}

static int runtime_physical_less(const cand_t *a,const cand_t *b)
{
    uint8_t ka[15],kb[15],cl,i=0u;
    for(cl=0;cl<2u;cl++){
        ka[i]=(uint8_t)(a->active[cl]?a->s1[cl]:3u);
        kb[i++]=(uint8_t)(b->active[cl]?b->s1[cl]:3u);
        ka[i]=(uint8_t)(a->active[cl]?a->s3[cl]:3u);
        kb[i++]=(uint8_t)(b->active[cl]?b->s3[cl]:3u);
    }
    for(cl=0;cl<2u;cl++){
        uint8_t a3=a->s2pf[cl]!=N?I:a->dma3[cl];
        uint8_t b3=b->s2pf[cl]!=N?I:b->dma3[cl];
        ka[i]=runtime_dma_order(a->active[cl]?a->dma1[cl]:N);
        kb[i++]=runtime_dma_order(b->active[cl]?b->dma1[cl]:N);
        ka[i]=runtime_dma_order(a->active[cl]?a3:N);
        kb[i++]=runtime_dma_order(b->active[cl]?b3:N);
        ka[i]=runtime_dma_order(a->active[cl]?a->s2pf[cl]:N);
        kb[i++]=runtime_dma_order(b->active[cl]?b->s2pf[cl]:N);
    }
    ka[i]=runtime_dma_order(N);kb[i++]=runtime_dma_order(N);
    for(cl=0;cl<2u;cl++){
        ka[i]=a->active[cl]?a->s1_hit[cl]:0u;
        kb[i++]=b->active[cl]?b->s1_hit[cl]:0u;
        ka[i]=a->active[cl]?a->s3_hit[cl]:0u;
        kb[i++]=b->active[cl]?b->s3_hit[cl]:0u;
    }
    return memcmp(ka,kb,sizeof(ka))<0;
}

static int local_better(const cand_t *a, const cand_t *b)
{
    uint32_t x,y;
    x=cand_max_end(a); y=cand_max_end(b); if(x!=y) return x<y;
    x=cand_sum_end(a); y=cand_sum_end(b); if(x!=y) return x<y;
    x=cand_latest_start(a); y=cand_latest_start(b); if(x!=y) return x<y;
    x=cand_s2pf_count(a); y=cand_s2pf_count(b); if(x!=y) return x>y;
    x=cand_s4pf_count(a); y=cand_s4pf_count(b); if(x!=y) return x>y;
    if(a->profile_slot!=b->profile_slot)return a->profile_slot<b->profile_slot;
    return runtime_physical_less(a,b);
}

static int transition_better(const cand_t *a,const cand_t *b)
{
    uint32_t x,y;
    x=cand_max_end(a);y=cand_max_end(b);if(x!=y)return x<y;
    x=cand_sum_end(a);y=cand_sum_end(b);if(x!=y)return x<y;
    return cand_latest_start(a)<cand_latest_start(b);
}

static int same_runtime_physical(const cand_t *a,const cand_t *b)
{
    uint8_t cl;
    if(a->family!=b->family||a->lsel_a!=b->lsel_a||a->lsel_b!=b->lsel_b||
       a->split_balanced!=b->split_balanced)return 0;
    for(cl=0;cl<2u;cl++){
        if(a->active[cl]!=b->active[cl])return 0;
        if(!a->active[cl])continue;
        if(a->s1[cl]!=b->s1[cl]||a->s3[cl]!=b->s3[cl]||
           a->dma1[cl]!=b->dma1[cl]||a->dma3[cl]!=b->dma3[cl]||
           a->s2pf[cl]!=b->s2pf[cl]||a->s1_hit[cl]!=b->s1_hit[cl]||
           (a->dma3[cl]==N)!=(b->dma3[cl]==N))return 0;
    }
    return 1;
}

typedef struct { uint8_t valid; cand_t cand; } cand_slot_t;

static void target_select_sink(const state_t *s, const profile_t *p,
                               const cand_t *c, void *opaque)
{
    cand_slot_t *best=(cand_slot_t *)opaque;
    (void)s; (void)p;
    if(!best->valid || local_better(c,&best->cand)){best->valid=1u;best->cand=*c;}
}

/* Python materialize_targeted_s4pf_variant(): C2 then C3; local lane then
 * BOTH; reject uncertifiable retroactive intervals; rematerialize the same
 * logical/physical profile with the concrete cache hit. */
static int targeted_variant(const state_t *s, const profile_t *source,
                            const cand_t *base, cand_t *out)
{
    state_t aug=*s;
    profile_t q=*source;
    uint8_t forced[2]={0u,0u},cl;
    cand_slot_t selected;
    if(s->nr<S4PF_MIN_REMAINING) return 0;
    memset(&selected,0,sizeof(selected));

    for(cl=0;cl<2u;cl++){
        snap_t *own=&aug.c[cl],*peer=&aug.c[1u-cl];
        uint8_t trials[2]={(uint8_t)(cl==0u?I:X),D};
        uint8_t k;
        if(!base->active[cl] || base->s1_hit[cl]) continue;
        if(own->cur_eid<0 || own->cache_valid) continue;
        if(peer->cur_eid>=0 && own->dma3_end<peer->task_start) continue;
        for(k=0;k<2u;k++){
            snap_t trial;
            uint8_t dma=trials[k];
            if(own->dma3_end+dma_s1_ticks(dma)>own->compute_end) continue;
            trial=with_prefetch(own,base->eid[cl],dma);
            if(cl==0u){ if(!bw_feasible(&trial,peer)) continue; }
            else { if(!bw_feasible(peer,&trial)) continue; }
            *own=trial; forced[cl]=dma; break;
        }
    }
    if(forced[0]==N && forced[1]==N) return 0;

    /* physical_profile(base action), followed by _profile_with_residency */
    q.active2=base->active[0]; q.active3=base->active[1];
    for(cl=0;cl<2u;cl++){
        if(!base->active[cl]) continue;
        q.s1[cl]=base->s1[cl]; q.s3[cl]=base->s3[cl];
        q.dma1[cl]=base->dma1[cl]; q.dma3[cl]=base->dma3[cl];
        q.s2pf[cl]=base->s2pf[cl]; q.expect_s1_hit[cl]=base->s1_hit[cl];
        q.expect_s3_ready[cl]=(uint8_t)(base->dma3[cl]==N);
        /* physical_profile(action) reports S3 ready for both a true full hit
         * and an S2PF realization.  _profile_with_residency then canonicalizes
         * that field to a real cached-S3 profile and clears S2PF. */
        if(q.expect_s3_ready[cl])q.s2pf[cl]=N;
        if(forced[cl]!=N){q.dma1[cl]=N;q.expect_s1_hit[cl]=1u;}
    }
    materialize_profile(&aug,&q,forced[0]!=N,forced[1]!=N,
                        target_select_sink,&selected);
    if(!selected.valid) return 0;
    selected.cand.s4pf_dma[0]=forced[0];
    selected.cand.s4pf_dma[1]=forced[1];
    *out=selected.cand;
    return 1;
}

typedef struct {
    uint8_t valid;
    uint8_t family, lsel_a, lsel_b, split_balanced;
    cand_slot_t baseline, targeted;
} logical_group_t;

typedef struct {
    logical_group_t group[LOGICAL_MAX];
    uint8_t n;
} group_bank_t;

static int same_logical(const logical_group_t *g, const cand_t *c)
{
    return g->family==c->family && g->lsel_a==c->lsel_a &&
           g->lsel_b==c->lsel_b && g->split_balanced==c->split_balanced;
}

static logical_group_t *find_group(group_bank_t *b, const cand_t *c)
{
    uint8_t i;
    for(i=0;i<b->n;i++) if(same_logical(&b->group[i],c)) return &b->group[i];
    if(b->n>=LOGICAL_MAX) return NULL;
    b->group[b->n].valid=1u; b->group[b->n].family=c->family;
    b->group[b->n].lsel_a=c->lsel_a; b->group[b->n].lsel_b=c->lsel_b;
    b->group[b->n].split_balanced=c->split_balanced;
    return &b->group[b->n++];
}

static void child_state(const state_t *before,const cand_t *c,state_t *out);

static uint8_t swap_lane(uint8_t d){return d==I?X:(d==X?I:d);}

typedef struct {
    uint32_t task_start,task_end,dma1_end,s2_end,dma3_end;
    int32_t pf_start,pf_end,s2pf_start,s2pf_end;
    int8_t pf_label;
    uint8_t dma_s1,dma_s3,active,pf_dma,pf_full,s2pf_dma;
} future_snap_key_t;

typedef struct {
    future_snap_key_t c[2];
    uint32_t cluster_work;
    uint16_t removed_ntok[2];
    uint16_t named_ntok[2];
    uint8_t removed_count,named_count,comparable;
} future_key_t;

static int remains_after_candidate(const state_t *s,const cand_t *c,
                                   int16_t eid,uint16_t *ntok)
{
    int index=rem_index_by_eid(s,eid);uint8_t cl;
    if(index<0)return 0;
    for(cl=0;cl<2u;cl++)if(c->active[cl]&&c->eid[cl]==eid)return 0;
    if(ntok)*ntok=s->rem[index].ntok;
    return 1;
}

static future_snap_key_t future_snap_key(const snap_t *s,uint8_t lanes,
                                         int8_t pf_label)
{
    future_snap_key_t k;memset(&k,0,sizeof(k));
    k.task_start=s->task_start;k.task_end=s->task_end;k.dma1_end=s->dma1_end;
    k.s2_end=s->s2_end;k.dma3_end=s->dma3_end;k.active=(uint8_t)(s->cur_eid>=0);
    k.dma_s1=lanes?swap_lane(s->dma_s1):s->dma_s1;
    k.dma_s3=lanes?swap_lane(s->dma_s3):s->dma_s3;
    if(s->cache_valid){
        k.pf_start=s->cache_full?-1:(int32_t)s->dma3_end;
        k.pf_end=s->cache_end;k.pf_label=pf_label;k.pf_full=s->cache_full;
        k.pf_dma=lanes?swap_lane(s->pf_dma):s->pf_dma;
    }else{k.pf_start=-1;k.pf_end=-1;k.pf_label=-1;}
    if(s->s2pf_dma!=N){
        k.s2pf_start=(int32_t)s->dma1_end;k.s2pf_end=s->s2pf_end;
        k.s2pf_dma=lanes?swap_lane(s->s2pf_dma):s->s2pf_dma;
    }else{k.s2pf_start=-1;k.s2pf_end=-1;}
    return k;
}

/* Exact compact form of four_stage_scheduler._canonical_state_future_key().
 * Because all candidates share the same parent, the post-action remaining
 * multiset is represented losslessly by the one/two removed token counts.
 * Named residents retain their same/different relation and remaining load,
 * while numeric expert IDs are discarded exactly as in Python. */
static future_key_t candidate_future_key(const state_t *s,const cand_t *c)
{
    future_key_t k,trial;uint8_t cl,swap,lanes,have=0u;
    uint16_t removed[2]={0u,0u};uint8_t removed_count=0u;
    memset(&k,0,sizeof(k));
    for(cl=0;cl<2u;cl++)if(c->active[cl] &&
       (cl==0u || !c->active[0] || c->eid[cl]!=c->eid[0])){
        int index=rem_index_by_eid(s,c->eid[cl]);
        if(index<0 || removed_count>=2u)return k;
        removed[removed_count++]=s->rem[index].ntok;
    }
    if(removed_count==2u&&removed[0]>removed[1]){
        uint16_t tmp=removed[0];removed[0]=removed[1];removed[1]=tmp;
    }
    for(swap=0;swap<2u;swap++){
        uint8_t order[2]={(uint8_t)(swap?1u:0u),(uint8_t)(swap?0u:1u)};
        int16_t named_eid[2]={-1,-1};uint16_t named_ntok[2]={0u,0u};
        int8_t label[2]={-1,-1};uint8_t named_count=0u,pos;
        for(pos=0;pos<2u;pos++){
            const snap_t *snap=&c->child[order[pos]];uint16_t ntok;uint8_t n;
            if(!snap->cache_valid)continue;
            if(!remains_after_candidate(s,c,snap->cache_eid,&ntok)){
                label[pos]=2;continue;
            }
            for(n=0u;n<named_count;n++)if(named_eid[n]==snap->cache_eid)break;
            if(n==named_count){
                if(named_count>=2u)return k;
                named_eid[named_count]=snap->cache_eid;
                named_ntok[named_count]=ntok;named_count++;
            }
            label[pos]=(int8_t)n;
        }
        for(lanes=0;lanes<2u;lanes++){
            memset(&trial,0,sizeof(trial));trial.cluster_work=c->child_work;
            trial.removed_count=removed_count;trial.removed_ntok[0]=removed[0];
            trial.removed_ntok[1]=removed[1];trial.named_count=named_count;
            trial.named_ntok[0]=named_ntok[0];trial.named_ntok[1]=named_ntok[1];
            trial.c[0]=future_snap_key(&c->child[order[0]],lanes,label[0]);
            trial.c[1]=future_snap_key(&c->child[order[1]],lanes,label[1]);
            trial.comparable=1u;
            if(!have||memcmp(&trial,&k,sizeof(k))<0){k=trial;have=1u;}
        }
    }
    return k;
}

static int future_key_equal(const future_key_t *a,const future_key_t *b)
{
    return a->comparable&&b->comparable&&memcmp(a,b,sizeof(*a))==0;
}

typedef struct { cand_slot_t item[8]; uint8_t n, overflow; } profile_collect_t;

typedef struct {
    future_key_t key;
    cand_t cand;
    profile_t source;
} physical_entry_t;

typedef struct {
    physical_entry_t item[PHYSICAL_MAX];
    uint8_t n,overflow;
} physical_bank_t;

static void profile_collect_sink(const state_t *s,const profile_t *p,
                                 const cand_t *c,void *opaque)
{
    profile_collect_t *v=(profile_collect_t *)opaque;uint8_t i;
    (void)s;(void)p;
    for(i=0;i<v->n;i++)if(same_runtime_physical(&v->item[i].cand,c)){
        if(transition_better(c,&v->item[i].cand))v->item[i].cand=*c;
        return;
    }
    if(v->n<ARRAY_LEN(v->item)){v->item[v->n].valid=1u;v->item[v->n].cand=*c;v->n++;}
    else v->overflow=1u;
}

static uint8_t source_family_order(uint8_t family)
{
    return family==PAIR?0u:(family==SINGLE?1u:2u);
}

static uint8_t source_mode_order(uint8_t mode)
{ return mode==ONE?0u:(mode==SYNC?1u:2u); }

static uint8_t source_selector_order(uint8_t selector)
{ return selector==B0?0u:(uint8_t)(selector+1u); }

static uint8_t source_split_order(const profile_t *p)
{
    if(p->family!=SPLIT)return 2u; /* NONE */
    return p->split_balanced?0u:1u; /* BALANCED, HALF */
}

static int runtime_profile_less(const profile_t *pa,const cand_t *a,
                                const profile_t *pb,const cand_t *b)
{
#define CMP_VALUE(x,y) do{if((x)!=(y))return (x)<(y);}while(0)
    CMP_VALUE(source_mode_order(pa->mode),source_mode_order(pb->mode));
    CMP_VALUE(source_family_order(pa->family),source_family_order(pb->family));
    CMP_VALUE(source_selector_order(pa->sel_a),source_selector_order(pb->sel_a));
    if(pa->family==PAIR&&pb->family==PAIR)
        CMP_VALUE(source_selector_order(pa->sel_b),source_selector_order(pb->sel_b));
    CMP_VALUE(source_split_order(pa),source_split_order(pb));
#undef CMP_VALUE
    return runtime_physical_less(a,b);
}

static void physical_bank_add(const state_t *s,const profile_t *p,
                              const cand_t *c,physical_bank_t *bank)
{
    future_key_t key=candidate_future_key(s,c);uint8_t i;
    for(i=0u;i<bank->n;i++)if(future_key_equal(&bank->item[i].key,&key)){
        uint8_t fixed=MIN_U(bank->item[i].cand.profile_slot,c->profile_slot);
        if(runtime_profile_less(p,c,&bank->item[i].source,&bank->item[i].cand)){
            bank->item[i].cand=*c;bank->item[i].source=*p;bank->item[i].key=key;
        }
        bank->item[i].cand.profile_slot=fixed;
        return;
    }
    if(bank->n>=PHYSICAL_MAX){bank->overflow=1u;return;}
    bank->item[bank->n].key=key;bank->item[bank->n].cand=*c;
    bank->item[bank->n].source=*p;bank->n++;
}

static int logical_group_less(const logical_group_t *a,const logical_group_t *b)
{
    uint8_t fa=(a->family==PAIR?0u:(a->family==SPLIT?1u:0u));
    uint8_t fb=(b->family==PAIR?0u:(b->family==SPLIT?1u:0u));
    if(a->family!=b->family) return fa<fb;
    if(a->lsel_a!=b->lsel_a) return a->lsel_a<b->lsel_a;
    if(a->lsel_b!=b->lsel_b) return a->lsel_b<b->lsel_b;
    return a->split_balanced<b->split_balanced;
}

static void sort_groups(group_bank_t *b)
{
    uint8_t i;
    for(i=1;i<b->n;i++){
        logical_group_t key=b->group[i]; int j=(int)i-1;
        while(j>=0 && logical_group_less(&key,&b->group[j])){
            b->group[j+1]=b->group[j];j--;
        }
        b->group[j+1]=key;
    }
}

static uint8_t materialize_groups(const state_t *s, group_bank_t *bank)
{
    uint8_t i;physical_bank_t physical;
    memset(bank,0,sizeof(*bank));memset(&physical,0,sizeof(physical));
    for(i=0;i<PROFILE_COUNT;i++){
        const profile_t *p=&kProfiles[kProfileEvalOrder[i]];
        profile_collect_t collected;uint8_t j;
        memset(&collected,0,sizeof(collected));
        materialize_profile(s,p,0u,0u,profile_collect_sink,&collected);
        if(collected.overflow)return 0u;
        for(j=0;j<collected.n;j++)physical_bank_add(s,p,&collected.item[j].cand,&physical);
        if(physical.overflow)return 0u;
    }
    for(i=0u;i<physical.n;i++){
        physical_entry_t *entry=&physical.item[i];
        logical_group_t *g=find_group(bank,&entry->cand);cand_t target;
        if(!g)return 0u;
        if(!g->baseline.valid||local_better(&entry->cand,&g->baseline.cand)){
            g->baseline.valid=1u;g->baseline.cand=entry->cand;
        }
        if(targeted_variant(s,&entry->source,&entry->cand,&target)&&
           (!g->targeted.valid||local_better(&target,&g->targeted.cand))){
            g->targeted.valid=1u;g->targeted.cand=target;
        }
    }
    for(i=0;i<bank->n;i++) if(bank->group[i].targeted.valid){
        uint32_t base=cand_max_end(&bank->group[i].baseline.cand);
        uint32_t target=cand_max_end(&bank->group[i].targeted.cand);
        if(base>=target+S4PF_MIN_GAIN_TICKS)
            bank->group[i].baseline=bank->group[i].targeted;
    }
    sort_groups(bank);
    return bank->n;
}

static void state_remove_eid(state_t *s, int16_t eid)
{
    int idx=rem_index_by_eid(s,eid); uint16_t ntok; uint16_t blocks; uint8_t i;
    if(idx<0) return;
    ntok=s->rem[idx].ntok; blocks=(uint16_t)CEIL_DIV2(ntok);
    s->token_sum-=ntok;
    s->odd_count=(uint8_t)(s->odd_count-(ntok&1u));
    s->block_sum-=blocks;
    if(blocks>=1u && blocks<=4u) s->hist[blocks-1u]--;
    for(i=(uint8_t)idx;i+1u<s->nr;i++) s->rem[i]=s->rem[i+1u];
    s->nr--;
}

static void child_state(const state_t *before,const cand_t *c,state_t *out)
{
    *out=*before; out->c[0]=c->child[0]; out->c[1]=c->child[1];
    out->cluster_work=c->child_work;
    if(c->active[0]) state_remove_eid(out,c->eid[0]);
    if(c->active[1] && (!c->active[0] || c->eid[1]!=c->eid[0])) state_remove_eid(out,c->eid[1]);
}

typedef struct {
    uint32_t committed,compute,release_chain,critical_chain,dma,combined;
} bound_parts_t; /* all fields are half ticks */

static int64_t floor_div_i64(int64_t a,int64_t b)
{
    int64_t q=a/b,r=a%b;
    if(r!=0 && ((r>0)!=(b>0))) q--;
    return q;
}

static uint8_t active_dma_mask(const snap_t *s,uint32_t t)
{
    interval_t v[4]; uint8_t n=collect_intervals(s,v),i,mask=N;
    for(i=0;i<n;i++) if(v[i].lo<=t && t<v[i].hi) mask=(uint8_t)(mask|v[i].dma);
    return mask;
}

static uint32_t dma_capacity_half(const snap_t *c2,const snap_t *c3,
                                  uint32_t start,uint32_t work_half)
{
    uint32_t points[18],tail; uint8_t np=0,i;
    interval_t iv[4]; uint8_t ni;
    add_unique_u32(points,&np,start*2u);
    ni=collect_intervals(c2,iv);
    for(i=0;i<ni;i++){
        if(iv[i].lo>=start)add_unique_u32(points,&np,iv[i].lo*2u);
        if(iv[i].hi>=start)add_unique_u32(points,&np,iv[i].hi*2u);
    }
    ni=collect_intervals(c3,iv);
    for(i=0;i<ni;i++){
        if(iv[i].lo>=start)add_unique_u32(points,&np,iv[i].lo*2u);
        if(iv[i].hi>=start)add_unique_u32(points,&np,iv[i].hi*2u);
    }
    sort_u32(points,np);
    for(i=0;i+1u<np;i++){
        uint32_t lo=points[i],hi=points[i+1u],cap;
        uint8_t used=(uint8_t)(active_dma_mask(c2,lo/2u)|active_dma_mask(c3,lo/2u));
        uint8_t free=(uint8_t)(2u-((used&I)!=0u)-((used&X)!=0u));
        if(!free || hi<=lo) continue;
        cap=(hi-lo)*free;
        if(work_half<=cap) return lo+(work_half+free-1u)/free;
        work_half-=cap;
    }
    tail=np?points[np-1u]:start*2u;
    tail=tail+(work_half+1u)/2u;
    return tail;
}

static uint8_t cache_slot_count(const state_t *s,uint8_t full)
{
    int16_t named[2]={-1,-1}; uint8_t n=0,cl;
    for(cl=0;cl<2u;cl++) if(s->c[cl].cache_valid && s->c[cl].cache_end>=0 &&
       (!full||s->c[cl].cache_full) && rem_index_by_eid(s,s->c[cl].cache_eid)>=0){
        if(n==0u || named[0]!=s->c[cl].cache_eid) named[n++]=s->c[cl].cache_eid;
    }
    return n>s->nr?s->nr:n;
}

static bound_parts_t lower_bound_parts(const state_t *s)
{
    bound_parts_t p; uint32_t e0=s->c[0].task_end,e1=s->c[1].task_end;
    uint32_t early=MIN_U(e0,e1),late=MAX_U(e0,e1),blocks=s->block_sum;
    uint32_t k[4],best=0xffffffffu,idx;
    int64_t num=(int64_t)e1-(int64_t)e0+(int64_t)blocks*3ll;
    int64_t fl=floor_div_i64(num,6ll),ce=-floor_div_i64(-num,6ll);
    uint32_t hot=s->nr?s->rem[0].ntok:0u;
    uint8_t s1slots=cache_slot_count(s,0u),fullslots=cache_slot_count(s,1u);
    uint32_t dma_work=((uint32_t)s->nr-s1slots)*4u+((uint32_t)s->nr-fullslots)*2u;
    uint32_t release=MIN_U(e0,e1);
    p.committed=late*2u;
    k[0]=0u;k[1]=blocks;k[2]=(uint32_t)(fl<0?0:(fl>(int64_t)blocks?blocks:fl));
    k[3]=(uint32_t)(ce<0?0:(ce>(int64_t)blocks?blocks:ce));
    for(idx=0;idx<4u;idx++){
        uint32_t v=MAX_U(e0+3u*k[idx],e1+3u*(blocks-k[idx]));
        if(v<best)best=v;
    }
    p.compute=best*2u;
    if(!s->nr){p.release_chain=late*2u;p.critical_chain=early*2u;}
    else {
        uint32_t hb=CEIL_DIV2(hot);
        p.release_chain=MIN_U(early+3u*hb,late+3u*CEIL_DIV2(hb))*2u;
        p.critical_chain=(early+3u*CEIL_DIV4(hot))*2u;
    }
    if(s->c[0].cur_eid>=0 && !s->c[0].cache_valid) release=MIN_U(release,s->c[0].dma3_end);
    if(s->c[1].cur_eid>=0 && !s->c[1].cache_valid) release=MIN_U(release,s->c[1].dma3_end);
    p.dma=MAX_U(late*2u,dma_capacity_half(&s->c[0],&s->c[1],release,dma_work*2u));
    p.combined=MAX_U(p.committed,MAX_U(p.compute,MAX_U(p.release_chain,MAX_U(p.critical_chain,p.dma))));
    return p;
}

static uint32_t lpt_half(const state_t *s,uint32_t f_half)
{
    uint8_t hist[4],i,b; uint32_t tail_work=s->block_sum*3u;
    uint32_t load[2]={s->c[0].task_end,s->c[1].task_end};
    memcpy(hist,s->hist,sizeof(hist));
    for(i=0;i<MIN_U(5u,s->nr);i++){
        uint32_t blocks=CEIL_DIV2(s->rem[i].ntok);
        tail_work-=3u*blocks;
        if(blocks<=4u) hist[blocks-1u]--;
        b=(uint8_t)(load[0]<=load[1]?0u:1u);load[b]+=3u*blocks;
    }
    for(b=4u;b>0u;b--) for(i=0;i<hist[b-1u];i++){
        uint8_t cl=(uint8_t)(load[0]<=load[1]?0u:1u);
        load[cl]+=3u*b; tail_work-=3u*b;
    }
    if(tail_work){
        uint8_t lo=(uint8_t)(load[0]<=load[1]?0u:1u),hi=(uint8_t)(1u-lo);
        uint32_t fill=MIN_U(load[hi]-load[lo],tail_work);
        load[lo]+=fill;tail_work-=fill;
        load[lo]+=tail_work/2u;load[hi]+=tail_work-tail_work/2u;
    }
    return MAX_U(f_half,MAX_U(load[0],load[1])*2u);
}

typedef struct {
    cand_t cand; state_t child;
    uint32_t f,h,compute,dma;
    uint32_t early,late,g;
    uint16_t selected_sum,selected_max;
    uint8_t s2pf_count,nrem,min_rank,max_rank,selects_t0;
    uint16_t candidate_index;
} score_t;

static void score_candidate(const state_t *before,const cand_t *c,
                            uint16_t index,score_t *o)
{
    bound_parts_t b; uint8_t cl; int rank; uint8_t minr=0xffu,maxr=0u;
    memset(o,0,sizeof(*o));o->cand=*c;o->candidate_index=index;
    child_state(before,c,&o->child);b=lower_bound_parts(&o->child);
    o->f=MAX_U(before->parent_bound_half,b.combined);o->child.parent_bound_half=o->f;
    o->h=lpt_half(&o->child,o->f);o->compute=b.compute;o->dma=b.dma;
    o->early=MIN_U(c->child[0].task_end,c->child[1].task_end);
    o->late=MAX_U(c->child[0].task_end,c->child[1].task_end);o->g=o->late;
    o->nrem=o->child.nr;o->s2pf_count=cand_s2pf_count(c);
    for(cl=0;cl<2u;cl++) if(c->active[cl]){
        o->selected_sum=(uint16_t)(o->selected_sum+c->ntok[cl]);
        o->selected_max=MAX_U(o->selected_max,c->ntok[cl]);
        rank=rem_index_by_eid(before,c->eid[cl]);
        if(rank>=0){minr=MIN_U(minr,(uint8_t)rank);maxr=MAX_U(maxr,(uint8_t)rank);if(rank==0)o->selects_t0=1u;}
    }
    o->min_rank=minr==0xffu?0u:minr;o->max_rank=maxr;
}

typedef struct {
    uint8_t low_work,sparse_hot,mid_plateau,short_tail,large_slack;
    uint16_t min_load;
} regime_t;

static regime_t classify_regime(const state_t *s)
{
    regime_t r; uint16_t top[5]={0u,0u,0u,0u,0u}; uint8_t i,mode=state_mode(s);
    uint32_t imbalance=s->c[0].task_end>s->c[1].task_end?
        s->c[0].task_end-s->c[1].task_end:s->c[1].task_end-s->c[0].task_end;
    memset(&r,0,sizeof(r));for(i=0;i<MIN_U(5u,s->nr);i++)top[i]=s->rem[i].ntok;
    r.min_load=s->nr?s->rem[s->nr-1u].ntok:0u;
    r.low_work=(uint8_t)(mode==ONE&&s->token_sum<=84u&&s->odd_count<=9u&&top[4]<=4u);
    r.sparse_hot=(uint8_t)(mode==SYNC&&s->nr>=2u&&top[0]>=2u*top[1]&&32u*s->hist[0]>11u*s->nr);
    r.mid_plateau=(uint8_t)(mode==ONE&&s->nr>=8u&&top[1]>=5u&&top[0]<=6u&&imbalance==3u);
    r.short_tail=(uint8_t)(mode==ONE&&s->nr>=2u&&s->nr<=7u&&top[1]>=5u&&top[0]<=6u&&imbalance==6u);
    r.large_slack=(uint8_t)(mode==ONE&&s->nr>=8u&&s->nr<=16u&&top[1]>=8u&&imbalance>=9u);
    return r;
}

static int base_rhs_better(const score_t *l,const score_t *r,uint8_t mode)
{
#define ASC(field) do{if(r->field!=l->field)return r->field<l->field;}while(0)
#define DESC(field) do{if(r->field!=l->field)return r->field>l->field;}while(0)
    ASC(f);ASC(h);ASC(compute);ASC(dma);
    if(mode==SYNC){DESC(selected_max);ASC(selected_sum);ASC(g);DESC(s2pf_count);}
    else {ASC(late);ASC(early);ASC(selected_sum);ASC(g);DESC(s2pf_count);ASC(nrem);}
    return r->candidate_index<l->candidate_index;
#undef ASC
#undef DESC
}

static int progress_rhs_better(const score_t *l,const score_t *r)
{
#define ASC(field) do{if(r->field!=l->field)return r->field<l->field;}while(0)
#define DESC(field) do{if(r->field!=l->field)return r->field>l->field;}while(0)
    ASC(f);ASC(h);DESC(s2pf_count);DESC(selected_sum);ASC(compute);ASC(dma);
    ASC(late);ASC(early);ASC(g);DESC(selected_max);
    return r->candidate_index<l->candidate_index;
#undef ASC
#undef DESC
}

static int hotspot_rhs_better(const score_t *l,const score_t *r)
{
#define ASC(field) do{if(r->field!=l->field)return r->field<l->field;}while(0)
#define DESC(field) do{if(r->field!=l->field)return r->field>l->field;}while(0)
    ASC(f);ASC(h);DESC(selected_max);ASC(selected_sum);ASC(compute);ASC(dma);
    ASC(g);DESC(s2pf_count);
    return r->candidate_index<l->candidate_index;
#undef ASC
#undef DESC
}

static int children_equal_raw(const score_t *a,const score_t *b)
{
    uint8_t i;
    if(a->child.nr!=b->child.nr || a->child.cluster_work!=b->child.cluster_work) return 0;
    if(memcmp(a->child.c,b->child.c,sizeof(a->child.c))!=0) return 0;
    for(i=0;i<a->child.nr;i++)
        if(a->child.rem[i].eid!=b->child.rem[i].eid || a->child.rem[i].ntok!=b->child.rem[i].ntok) return 0;
    return 1;
}

static score_t fold_pair(const state_t *before,const regime_t *reg,
                         const score_t *lhs,const score_t *rhs)
{
    const score_t *base=base_rhs_better(lhs,rhs,state_mode(before))?rhs:lhs;
    const score_t *progress=progress_rhs_better(lhs,rhs)?rhs:lhs;
    const score_t *hot=hotspot_rhs_better(lhs,rhs)?rhs:lhs;
    const score_t *selected=base;
    int differ=!children_equal_raw(lhs,rhs);
    uint8_t use=0u;

    if(reg->low_work && differ && progress!=base && base->s2pf_count==0u &&
       progress->selected_sum>base->selected_sum && progress->early<=base->early+1u){
        selected=progress;use=1u;
    }
    if(reg->mid_plateau && differ && progress!=base && base->s2pf_count==0u &&
       progress->s2pf_count>0u && progress->min_rank<=3u &&
       base->max_rank>=(uint8_t)(before->nr-2u) && progress->early==base->early &&
       progress->late<=base->late+6u){selected=progress;use=1u;}
    if((reg->short_tail||reg->large_slack) && differ && progress!=base){
        int common=progress->s2pf_count>0u && progress->selected_sum>base->selected_sum &&
            progress->early>=base->early && progress->f==base->f && progress->h==base->h &&
            progress->dma==base->dma;
        if(reg->short_tail&&common&&base->s2pf_count>0u&&progress->selects_t0&&
           !base->selects_t0&&progress->late<=base->late+3u){selected=progress;use=1u;}
        if(reg->large_slack&&common&&base->s2pf_count==0u&&progress->min_rank<=1u&&
           base->max_rank>=(uint8_t)(before->nr-2u)&&progress->late==base->late){
            selected=progress;use=1u;
        }
    }
    if(reg->sparse_hot && differ && hot!=base && hot->selects_t0&&!base->selects_t0&&
       hot->selected_max>base->selected_max&&hot->f==base->f&&hot->h==base->h&&
       hot->compute<=base->compute+6u&&hot->dma<=base->dma+12u&&
       (base->dma>230u||(reg->min_load>=2u&&base->dma>204u))){selected=hot;use=1u;}
    (void)use;
    return *selected;
}

static int choose_round(const state_t *s,cand_t *winner,state_t *child)
{
    group_bank_t bank; regime_t regime; score_t best,cur; uint8_t i,have=0u;
    if(!materialize_groups(s,&bank)) return 0;
    regime=classify_regime(s);
    for(i=0;i<bank.n;i++) if(bank.group[i].baseline.valid){
        score_candidate(s,&bank.group[i].baseline.cand,i,&cur);
        if(!have){best=cur;have=1u;} else best=fold_pair(s,&regime,&best,&cur);
    }
    if(!have) return 0;
    *winner=best.cand;*child=best.child;
    return 1;
}

static moe_status_t init_state(const moe_request_t *req,state_t *s)
{
    uint16_t i,j;
    memset(s,0,sizeof(*s));
    if(!req||req->n_experts==0u||req->n_experts>MOE_MAX_EXPERTS) return MOE_ERR_BAD_INPUT;
    if(req->cache_eid_c2<-1||req->cache_eid_c2>=MOE_MAX_EXPERTS||
       req->cache_eid_c3<-1||req->cache_eid_c3>=MOE_MAX_EXPERTS)
        return MOE_ERR_BAD_INPUT;
    s->c[0]=snap_idle(req->cache_eid_c2);s->c[1]=snap_idle(req->cache_eid_c3);
    s->nr=(uint8_t)req->n_experts;
    for(i=0;i<req->n_experts;i++){
        uint16_t blocks;
        if(req->experts[i].ntokens==0u||req->experts[i].expert_id>=MOE_MAX_EXPERTS)
            return MOE_ERR_BAD_INPUT;
        for(j=0;j<i;j++) if(req->experts[j].expert_id==req->experts[i].expert_id)
            return MOE_ERR_BAD_INPUT;
        s->rem[i].eid=(int16_t)req->experts[i].expert_id;
        s->rem[i].ntok=req->experts[i].ntokens;
        blocks=(uint16_t)CEIL_DIV2(req->experts[i].ntokens);
        s->token_sum+=req->experts[i].ntokens;
        s->odd_count=(uint8_t)(s->odd_count+(req->experts[i].ntokens&1u));
        s->block_sum+=blocks;
        if(blocks<=4u)s->hist[blocks-1u]++;
    }
    /* Python normalization order: descending load, ascending expert ID. */
    for(i=1u;i<req->n_experts;i++){
        rem_t key=s->rem[i];int k=(int)i-1;
        while(k>=0&&(s->rem[k].ntok<key.ntok||
              (s->rem[k].ntok==key.ntok&&s->rem[k].eid>key.eid))){
            s->rem[k+1]=s->rem[k];k--;
        }
        s->rem[k+1]=key;
    }
    s->parent_bound_half=lower_bound_parts(s).combined;
    return MOE_OK;
}

static void append_winner_plan(plan_t *plan,uint16_t *n,const cand_t *c,
                               int16_t last_by_cluster[2])
{
    uint8_t cl;
    for(cl=0;cl<2u;cl++) if(c->s4pf_dma[cl]!=N && last_by_cluster[cl]>=0){
        plan[last_by_cluster[cl]].s4pf_dma=c->s4pf_dma[cl];
        plan[last_by_cluster[cl]].s4pf_eid=c->eid[cl];
    }
    for(cl=0;cl<2u;cl++) if(c->active[cl]){
        plan_t *p=&plan[*n];memset(p,0,sizeof(*p));
        p->eid=c->eid[cl];p->ntok=c->ntok[cl];p->tok_start=c->tok_start[cl];
        p->cluster=cl;p->shape_s1=c->s1[cl];p->shape_s3=c->s3[cl];
        p->dma_s1=c->dma1[cl];p->dma_s3=c->dma3[cl];p->s2pf_dma=c->s2pf[cl];
        p->skip_s1=c->s1_hit[cl];p->skip_s3=(uint8_t)(c->s3_hit[cl]||c->s2pf[cl]!=N);
        p->s4pf_dma=N;p->s4pf_eid=-1;last_by_cluster[cl]=(int16_t)(*n);(*n)++;
    }
}

static moe_status_t make_plan(const moe_request_t *req,plan_t *plan,
                              uint16_t *n_plan,uint32_t *makespan_ticks)
{
    state_t state,child; cand_t winner; int16_t last[2]={-1,-1};
    moe_status_t st=init_state(req,&state);
    if(st!=MOE_OK)return st;
    *n_plan=0u;
    while(state.nr){
        if(!choose_round(&state,&winner,&child))return MOE_ERR_INTERNAL;
        if(*n_plan+(uint16_t)winner.active[0]+(uint16_t)winner.active[1]>MOE_MAX_TASKS)
            return MOE_ERR_OVERFLOW;
        append_winner_plan(plan,n_plan,&winner,last);state=child;
    }
    if(makespan_ticks)*makespan_ticks=MAX_U(state.c[0].task_end,state.c[1].task_end);
    return MOE_OK;
}

static moe_hw_plan_entry_t hw_entry(const plan_t *p)
{
    moe_hw_plan_entry_t e;memset(&e,0,sizeof(e));e.valid=1u;
    e.desc.cluster=(moe_cluster_t)p->cluster;e.desc.expert_id=(uint16_t)p->eid;
    e.desc.token_start_rank=p->tok_start;e.desc.ntokens=p->ntok;
    e.desc.shape_s1=(moe_shape_t)p->shape_s1;e.desc.shape_s3=(moe_shape_t)p->shape_s3;
    e.desc.skip_s1=p->skip_s1;e.desc.skip_s3=p->skip_s3;
    e.desc.has_s2pf=(uint8_t)(p->s2pf_dma!=N);
    e.desc.dma_s1=(moe_dma_binding_t)p->dma_s1;
    e.desc.dma_s3=(moe_dma_binding_t)p->dma_s3;
    e.desc.s2pf_dma=(moe_dma_binding_t)p->s2pf_dma;
    e.allow_s4pf=0u;e.s4pf_dma=(moe_dma_binding_t)p->s4pf_dma;
    e.s4pf_expert_id=p->s4pf_eid;return e;
}

/* Direct Python-policy-plan -> public C schedule lowering.  Keeping this
 * production path direct is important: the compact RTL-compatible plan is an
 * optional export format, not an algorithmic input to the C scheduler. */
static moe_status_t lower_internal_plan(const plan_t *plan,uint16_t n_plan,
                                        moe_schedule_t *out)
{
    uint16_t pi;
    if(!plan||!out)return MOE_ERR_BAD_INPUT;
    if(n_plan>MOE_MAX_TASKS)return MOE_ERR_OVERFLOW;
    out->n_tasks=0u;out->n_dma_ops=0u;
    for(pi=0u;pi<n_plan;pi++){
        const plan_t *p=&plan[pi];moe_task_t *t;uint32_t tail;
        if(out->n_tasks>=MOE_MAX_TASKS)return MOE_ERR_OVERFLOW;
        t=&out->tasks[out->n_tasks];memset(t,0,sizeof(*t));
        t->cluster=(moe_cluster_t)p->cluster;t->expert_id=(uint16_t)p->eid;
        t->token_start_rank=p->tok_start;t->ntokens=p->ntok;
        t->shape_s1=(moe_shape_t)p->shape_s1;t->shape_s3=(moe_shape_t)p->shape_s3;
        t->dma_s1=(moe_dma_binding_t)p->dma_s1;t->dma_s3=(moe_dma_binding_t)p->dma_s3;
        t->skip_s1=p->skip_s1;t->skip_s3=p->skip_s3;
        tail=p->skip_s1?(uint32_t)p->ntok:
            ((uint32_t)p->ntok>kMdim[p->shape_s1]?
             (uint32_t)p->ntok-kMdim[p->shape_s1]:0u);
        t->m_s2_exec=CEIL_DIV2(tail);t->skip_s2=(uint8_t)(t->m_s2_exec==0u);
        tail=p->skip_s3?(uint32_t)p->ntok:
            ((uint32_t)p->ntok>kMdim[p->shape_s3]?
             (uint32_t)p->ntok-kMdim[p->shape_s3]:0u);
        t->m_s4_exec=CEIL_DIV2(tail);t->skip_s4=(uint8_t)(t->m_s4_exec==0u);
#define ADD_INTERNAL_OP(kind_,dma_,eid_) do{ \
        moe_dma_op_t *op;if(out->n_dma_ops>=MOE_MAX_DMA_OPS)return MOE_ERR_OVERFLOW; \
        op=&out->dma_ops[out->n_dma_ops++];op->task_idx=out->n_tasks;op->kind=(kind_); \
        op->dma=(moe_dma_binding_t)(dma_);op->expert_id=(eid_); \
    }while(0)
        if(!p->skip_s1)ADD_INTERNAL_OP(MOE_DMA_OP_S1,p->dma_s1,p->eid);
        if(p->s2pf_dma!=N)ADD_INTERNAL_OP(MOE_DMA_OP_S2_PREFETCH,p->s2pf_dma,p->eid);
        else if(!p->skip_s3)ADD_INTERNAL_OP(MOE_DMA_OP_S3,p->dma_s3,p->eid);
        if(p->s4pf_dma!=N)ADD_INTERNAL_OP(MOE_DMA_OP_S4_PREFETCH,p->s4pf_dma,p->s4pf_eid);
#undef ADD_INTERNAL_OP
        out->n_tasks++;
    }
    return MOE_OK;
}

static void remove_dma_op(moe_schedule_t *out,uint16_t idx,
                          uint16_t pending[2],uint8_t valid[2])
{
    uint16_t i;uint8_t cl;
    if(idx>=out->n_dma_ops)return;
    for(i=idx;i+1u<out->n_dma_ops;i++)out->dma_ops[i]=out->dma_ops[i+1u];
    out->n_dma_ops--;
    for(cl=0;cl<2u;cl++)if(valid[cl]&&pending[cl]>idx)pending[cl]--;
}

moe_status_t moe_lower_hw_plan(const moe_request_t *req,
                               const moe_hw_plan_entry_t *plan,uint16_t n_plan,
                               moe_schedule_t *out)
{
    uint16_t pi;uint8_t pending_valid[2]={0u,0u};uint16_t pending[2]={0u,0u};
    if(!req||!plan||!out)return MOE_ERR_BAD_INPUT;
    if(n_plan>MOE_MAX_TASKS)return MOE_ERR_OVERFLOW;
    out->n_tasks=0u;out->n_dma_ops=0u;
    for(pi=0;pi<n_plan;pi++)if(plan[pi].valid){
        const moe_hw_plan_desc_t *p=&plan[pi].desc;uint8_t cl=(uint8_t)p->cluster;
        moe_task_t *t;uint8_t single=(uint8_t)(cl==0u?I:X);
        uint8_t d1=(uint8_t)p->dma_s1,d3=(uint8_t)p->dma_s3,pf=(uint8_t)p->s2pf_dma;
        uint32_t tail;
        if(out->n_tasks>=MOE_MAX_TASKS)return MOE_ERR_OVERFLOW;
        if(pending_valid[cl]){
            out->dma_ops[pending[cl]].expert_id=(int16_t)p->expert_id;
            pending_valid[cl]=0u;
        }
        if(!p->skip_s1&&d1==N)d1=(uint8_t)(p->shape_s1==MOE_SHAPE_C?D:single);
        if(p->has_s2pf&&pf==N)pf=single;
        if(!p->skip_s3&&!p->has_s2pf&&d3==N)d3=(uint8_t)(p->shape_s3==MOE_SHAPE_C?D:single);
        t=&out->tasks[out->n_tasks];memset(t,0,sizeof(*t));
        t->cluster=p->cluster;t->expert_id=p->expert_id;t->token_start_rank=p->token_start_rank;
        t->ntokens=p->ntokens;t->shape_s1=p->shape_s1;t->shape_s3=p->shape_s3;
        t->dma_s1=(moe_dma_binding_t)d1;t->dma_s3=(moe_dma_binding_t)d3;
        t->skip_s1=p->skip_s1;t->skip_s3=p->skip_s3;
        tail=p->skip_s1?(uint32_t)p->ntokens:
            ((uint32_t)p->ntokens>kMdim[p->shape_s1]?
             (uint32_t)p->ntokens-kMdim[p->shape_s1]:0u);
        t->m_s2_exec=CEIL_DIV2(tail);t->skip_s2=(uint8_t)(t->m_s2_exec==0u);
        tail=p->skip_s3?(uint32_t)p->ntokens:
            ((uint32_t)p->ntokens>kMdim[p->shape_s3]?
             (uint32_t)p->ntokens-kMdim[p->shape_s3]:0u);
        t->m_s4_exec=CEIL_DIV2(tail);t->skip_s4=(uint8_t)(t->m_s4_exec==0u);
#define ADD_OP(kind_,dma_,eid_) do{ \
        moe_dma_op_t *op;if(out->n_dma_ops>=MOE_MAX_DMA_OPS)return MOE_ERR_OVERFLOW; \
        op=&out->dma_ops[out->n_dma_ops++];op->task_idx=out->n_tasks;op->kind=(kind_); \
        op->dma=(moe_dma_binding_t)(dma_);op->expert_id=(eid_); \
    }while(0)
        if(!p->skip_s1)ADD_OP(MOE_DMA_OP_S1,d1,(int16_t)p->expert_id);
        if(p->has_s2pf)ADD_OP(MOE_DMA_OP_S2_PREFETCH,pf,(int16_t)p->expert_id);
        else if(!p->skip_s3)ADD_OP(MOE_DMA_OP_S3,d3,(int16_t)p->expert_id);
        if(plan[pi].s4pf_dma!=MOE_DMA_NONE)
            ADD_OP(MOE_DMA_OP_S4_PREFETCH,plan[pi].s4pf_dma,plan[pi].s4pf_expert_id);
        else if(plan[pi].allow_s4pf){
            if(out->n_dma_ops>=MOE_MAX_DMA_OPS)return MOE_ERR_OVERFLOW;
            pending_valid[cl]=1u;pending[cl]=out->n_dma_ops;
            ADD_OP(MOE_DMA_OP_S4_PREFETCH,single,-1);
        }
#undef ADD_OP
        out->n_tasks++;
    }
    for(uint8_t cl=0;cl<2u;cl++)if(pending_valid[cl]){
        remove_dma_op(out,pending[cl],pending,pending_valid);pending_valid[cl]=0u;
    }
    return MOE_OK;
}

moe_status_t moe_make_hw_plan(const moe_request_t *req,moe_hw_plan_entry_t *out,
                              uint16_t *n_plan)
{
    uint16_t n=0u,i;moe_status_t st;
    if(!out||!n_plan)return MOE_ERR_BAD_INPUT;
    st=make_plan(req,g_plan_scratch,&n,NULL);if(st!=MOE_OK)return st;
    for(i=0;i<n;i++)out[i]=hw_entry(&g_plan_scratch[i]);
    *n_plan=n;
    return MOE_OK;
}

static moe_status_t schedule_impl(const moe_request_t *req,moe_schedule_t *out,
                                  uint32_t *makespan_ticks)
{
    uint16_t n=0u;moe_status_t st;
    if(!out)return MOE_ERR_BAD_INPUT;
    st=make_plan(req,g_plan_scratch,&n,makespan_ticks);if(st!=MOE_OK)return st;
    return lower_internal_plan(g_plan_scratch,n,out);
}

moe_status_t moe_schedule(const moe_request_t *req,moe_schedule_t *out)
{ return schedule_impl(req,out,NULL); }

#ifdef MOE_SCHEDULER_TEST_API
moe_status_t moe_schedule_debug(const moe_request_t *req,moe_schedule_t *out,
                                uint32_t *makespan_ticks)
{ return schedule_impl(req,out,makespan_ticks); }
#endif

#endif /* !MOE_ENABLE_HW_SCHEDULER */
