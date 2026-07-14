# multi_cluster_MoE_test

This workload keeps the controlled DMA experiments and reduces the MoE body to
one static 8-token slot on C0. C1 participates only in the cross-cluster DMA
probes; it does not run an expert slot.

The front of the DFG measures one complete gate-weight tensor (`360448 B` with
the default dimensions):

```text
C0 iDMA: L3 -> L1
C0 xDMA: L3 -> L1
C0 xDMA: L1 -> L3
C0 dual DMA: same destination bank phase
C0 dual DMA: iDMA destination shifted by 64 B
C0/C1 concurrent xDMA: L3 -> local L1
C0/C1 concurrent iDMA: L3 -> local L1
```

All probes finish before the C0 slot starts. The slot is:

```text
load 8 padded token rows
S1 block 0/1: one iDMA stream loads gate then up while the previous block computes
S1 compute: shape 1 = (4, 8, 8) per VC, combined (4, 8, 16)
S2 compute: shape 2 tail path
S2 prefetch: iDMA loads the left down half while xDMA loads the right half
S3 compute: one full-width shape-1 down node, no output-block split
S3 prefetch: iDMA+xDMA load the hypothetical next expert gate/up tensors
S4 compute: shape 2 tail path
store 8 padded output rows
```

`array_shape=1` selects the fixed `(4,8,8)` hardware shape per VC. `N` is a
temporal output-column loop bound, not a physical array dimension. S3 uses
`N=64` instead of two `N=32` calls so the same combined `(4,8,16)` array shape
traverses the complete 1024-wide output in one node.

The S2 compute and S2 prefetch share the same predecessor and can overlap. S3
starts only after both finish. The S3 compute and next-expert prefetch also
share a boundary and can overlap; the final store joins both paths so the test
cannot exit while the prefetch is still active.

Datagen emits canonical L15 B order and 2080-byte token rows (`2048-byte INT16
payload + 32 zero bytes`). It emits one complete C0 expert and only gate/up for
the hypothetical next expert, so no unused next-expert down tensor is stored.

## Trace observation

Build with `PERF_TRACING=1` and process the instruction traces with the existing
`util/bingo_trace/bingo_trace.py` flow. The outer intervals are:

```text
BINGO_TRACE_DMA_PROBE_IDMA_BASELINE
BINGO_TRACE_DMA_PROBE_XDMA_LOAD_BASELINE
BINGO_TRACE_DMA_PROBE_XDMA_STORE_BASELINE
BINGO_TRACE_DMA_PROBE_DUAL_SAME_PHASE
BINGO_TRACE_DMA_PROBE_DUAL_SHIFTED_PHASE
BINGO_TRACE_DMA_PROBE_XDMA_CROSS_CLUSTER
BINGO_TRACE_DMA_PROBE_IDMA_CROSS_CLUSTER
BINGO_TRACE_DMA_PREFETCH_S2_DOWN
BINGO_TRACE_DMA_PREFETCH_S3_NEXT_SWIGLU
```

Existing `IDMA_CFG/RUN`, `XDMA_CFG/RUN`, and `DUAL_DMA_CFG` intervals remain
nested inside them. The payload-only ideal duration of a 360448-byte transfer
is `360448 / 64 = 5632 cycles` per DMA engine.

Compare the serial iDMA and xDMA baseline intervals before and after increasing
the cluster wide CDC depth. Use the exact same software and payload size for
both RTL runs so any change comes from the RTL rather than the DFG.

## Debug build

For deadlock bring-up, build the software with:

```text
make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  CFG_OVERRIDE=target/rtl/cfg/hemaia_4cluster_singlechip.hjson \
  DEBUG_LEVEL=1 NODE_TIMING=1 PERF_TRACING=1
```

`NODE_TIMING=1` prints every device task's start/end `mcycle`. A task that
prints Start but never End identifies the blocked kernel; a task that never
prints Start is still waiting on a predecessor or core queue. Node names are
preserved in generated `offload_bingo_hw.h`, so each numeric task ID can be
mapped back to the exact probe or S1/S2/S3/S4 operation. Disable `DEBUG_LEVEL`
and `NODE_TIMING` for performance measurements because UART printing is
intrusive; keep `PERF_TRACING=1` for the normal simulation trace flow.
