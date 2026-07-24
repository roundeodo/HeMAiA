# multi_cluster_MoE_test

This workload reproduces one complete individual-expert slot from the production
`multi_cluster_MoE` graph. It uses the production ABI and device library. The
benchmark and production workload both select the `optimized` slot API family.
The original per-block `discrete` APIs remain selectable for controlled
comparisons; the redundant intermediate fused API family has been removed.

## Workload

C0 and C1 concurrently process slot0 with seven dense INT16 tokens of width 2048
and independent `2048 -> 1024 -> 2048` experts. Gate, up, and down weights use
eight 128-column blocks and the production 16-bank A/B0/B1/D layout.

```text
gather seven slot0 tokens with production 2D iDMA
S1: load0 || configure0, then compute(i) || load(i+1)
S2: compute the remaining three tokens as two shapeC M tiles
S3/S4: cluster-specific down path
prepare the production xDMA 2D store while compute is still running
store slot0 with xDMA while gathering six slot1 tokens with iDMA
stop; slot1 is not computed
```

C0 uses shapeB and iDMA for S1. Its shapeC S2 boundary prefetches all down
weights with the production `BOTH` binding, using iDMA and xDMA concurrently.
C0 skips S3 and shapeC S4 computes all seven tokens without a prefetch. C1 uses
shapeB and xDMA for S1, shapeC S2 without a prefetch, then shapeB and xDMA for
the active S3 load/compute pipeline. While shapeC S4 computes its remaining
three tokens, C1 uses BOTH for the gate and up weights of the next S1 so the
full prefetch fits the short compute window. The focused test reuses expert 0
as the valid synthetic next-expert
source; the runtime descriptor still carries the scheduler-facing valid,
binding, and expert-ID fields. Both clusters use the same production device
ABI and independently check all 28672 output bytes.

The benchmark protocol binds every 64-byte/cycle C0 transfer to iDMA, every
64-byte/cycle C1 transfer to xDMA, and every 128-byte/cycle transfer to BOTH.
The API still reads shape and binding independently from the runtime call
record: the benchmark choices do not bind shapeB/shapeC to a particular stage
or DMA engine.

Set `SLOT_IMPLEMENTATION` in `main_bingo.py` to select one complete path:

- `discrete`: one DFG node per S1/S3 block and the original S2/S4 APIs.
- `optimized`: separate optimized APIs for gather/handoff, S1, S2, S3, and
  phase-batched S4, including the existing intra-stage and cross-stage preload.

The resulting S4 calls are `M=4` on C0 (all seven tokens) and `M=2` on C1
(tokens 4 through 6). These values are derived from the token count and the
runtime shape row count; they are not hard-coded into the device APIs. The
phase-batched compute API executes two bank phases for every runtime `M`, so
its RUN count is `2*M` rather than `block_count*M`.

The optimized APIs consume the runtime shape and DMA binding from the normal
production descriptor. S1/S2/S3/S4 accept any hardware shape supported by the
call ABI, while each DMA operation independently accepts iDMA, xDMA, or BOTH.
S4-prefetch does not infer its binding from the S4 compute shape. The focused
test supplies BOTH
explicitly because the current hardware-lite S4PF descriptor does not yet
encode this dynamic choice. These paths are implemented directly inside the
optimized API family; they do not dispatch to the discrete kernels.
Runtime token counts and `M` remain dynamic, and odd block counts use different
phase bounds rather than a fallback implementation.

The first load of S1/S3 is concurrent with the full block0 streamer/VersaCore
configuration. Each compute starts before patching the next block's B/D base
CSRs. Weight DMA configures the invariant xDMA 2D shape on block0 and only
patches addresses for later blocks. Load block `n >= 2` waits for compute block
`n-2`, so ping/pong storage cannot be overwritten while it is in use.

## Build

The production-slot mode is enabled by default in the workload Makefile.

```text
make -C target/sw clean
make -C target/sw sw
make -C target/sw apps
make -C target/sw bootrom
```

Run Questa outside the filesystem sandbox after sourcing
`questasim_2022.4.rc`. Generate reports in the container with:

```text
make -C target/sim traces
make -C target/sim bingo-vis-traces
```
