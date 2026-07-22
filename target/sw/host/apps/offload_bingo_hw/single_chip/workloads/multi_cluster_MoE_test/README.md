# multi_cluster_MoE_test

This workload reproduces one complete individual-expert slot from the production
`multi_cluster_MoE` graph. It uses only device APIs that are also used by the
large workload.

## Workload

C0 and C1 concurrently process slot0 with six dense INT16 tokens of width 2048
and independent `2048 -> 1024 -> 2048` experts. Gate, up, and down weights use
eight 128-column blocks and the production 16-bank A/B0/B1/D layout.

```text
gather six slot0 tokens with production 2D iDMA
S1: load0 || configure0, then compute(i) || load(i+1)
S2: compute the remaining two tokens
S3/S4: cluster-specific down path
prepare the production xDMA 2D store while compute is still running
store slot0 with xDMA while gathering six slot1 tokens with iDMA
stop; slot1 is not computed
```

C0 uses iDMA for S1. Its S2 boundary prefetches all down weights with the
production `BOTH` binding, using iDMA and xDMA concurrently. C0 skips S3 and S4
computes all six tokens. C1 uses xDMA for S1, performs no S2 prefetch, then uses
xDMA for the active S3 load/compute pipeline. While S4 computes its remaining
two tokens, C1 exercises the production BOTH S4-prefetch path, using iDMA for
gate and xDMA for up weights of the next S1. The focused test reuses expert 0
as the valid synthetic next-expert
source; the runtime descriptor still carries the scheduler-facing valid,
binding, and expert-ID fields. Both clusters use the same production device
APIs and independently check all 24576 output bytes.

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
