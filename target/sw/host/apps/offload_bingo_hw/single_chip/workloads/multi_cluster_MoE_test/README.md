# multi_cluster_MoE_test

This workload reproduces one complete individual-expert slot from the production
`multi_cluster_MoE` graph. It uses the production ABI and device library. The
benchmark and production workload both select the `optimized` slot API family.
The original per-block `discrete` APIs remain selectable for controlled
comparisons; the redundant intermediate fused API family has been removed.

## Workload

C0 and C1 concurrently process slot0 with eleven dense INT16 tokens of width
2048 and independent `2048 -> 1024 -> 2048` experts. Gate and up use eight
128-column blocks; down uses four 256-column blocks per VC half. Every logical
load is 256 KiB in the production 16-bank A/B0/B1/D layout.

```text
gather eleven slot0 tokens with production 2D iDMA
S1: load0 || configure0, then compute(i) || load(i+1)
S2: compute the remaining seven tokens as four shapeC M tiles
S3/S4: cluster-specific down path
prepare the production xDMA 2D store while compute is still running
store slot0 with xDMA while gathering six slot1 tokens with iDMA
C0 slot1: streamed S1, S2 compute || iDMA down prefetch, skip S3, full S4
C1 slot1: skip resident S1, full S2 compute || xDMA down prefetch, skip S3, full S4
store slot1 with xDMA and check both slots
```

C0 uses shapeB and iDMA for S1. Its shapeC S2 boundary prefetches all down
weights with the production single-DMA binding, using C0's assigned iDMA.
C0 skips S3 and shapeC S4 computes all eleven tokens without a prefetch. C1 uses
shapeB and xDMA for S1, shapeC S2 without a prefetch, then shapeB and xDMA for
the active S3 load/compute pipeline. While shapeC S4 computes its remaining
seven tokens, C1 uses its assigned xDMA for the gate and up weights of the next
S1. The focused test reuses expert 0
as the valid synthetic next-expert
source; the runtime descriptor still carries the scheduler-facing valid,
binding, and expert-ID fields. Both clusters use the same production device
ABI and independently check all slot0 and slot1 output bytes.

The protocol binds every S2 prefetch to the cluster's single DMA: iDMA on C0
and xDMA on C1. S4 prefetch carries an explicit `SINGLE` or `BOTH` operation;
`SINGLE` again selects the cluster's assigned DMA. The API reads shape and DMA
operation independently from the runtime call record, so the focused choices
do not bind shapeB/shapeC to a particular DMA engine.

Set `SLOT_IMPLEMENTATION` in `main_bingo.py` to select one complete path:

- `discrete`: one DFG node per S1/S3 block and the original S2/S4 APIs.
- `optimized`: separate optimized APIs for gather/handoff, S1, S2, S3, and
  phase-batched S4, including the existing intra-stage and cross-stage
  preload.

For slot0, the resulting S4 calls are `M=6` on C0 (all eleven tokens) and `M=4`
on C1 (tokens 4 through 10). In slot1 both S4 calls are `M=3`. C0 first computes
four tokens in streamed S1 and two in S2; C1 consumes the resident S1 weights by
skipping S1 and computes all six tokens in S2. These values are derived from the
token count and runtime shape row count; they are not hard-coded into the device
APIs. The S4
workers remain one Bingo node per core and exchange bank ownership only at the
two phase boundaries. One compute RUN traverses every same-parity down block,
so the optimized S4 executes `2*M` RUNs when both phases are populated. The
prefetch worker transfers every next-S1 block in that phase using the runtime
`SINGLE` or `BOTH` binding before the workers exchange phases.

The optimized APIs consume the runtime shape and DMA binding from the normal
production descriptor. S1/S2/S3/S4 accept any hardware shape supported by the
call ABI. S2 prefetch always uses the cluster's assigned single DMA. S4
prefetch does not infer its operation from the S4 compute shape: its descriptor
selects `SINGLE` or `BOTH`, and the device API dispatches to the corresponding
phase schedule. The focused test exercises S2 single-DMA prefetch on iDMA and
xDMA, plus the S4 single-xDMA prefetch that makes C1 slot1 resident. The `BOTH`
S4 path remains available to production descriptors.
These paths are implemented directly inside the
optimized API family; they do not dispatch to the discrete kernels. Runtime
token counts, `M`, stage block counts, shapes, and DMA bindings remain dynamic;
unequal or odd S1/S3 block counts use independent runtime phase bounds instead
of falling back to the discrete implementation.

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

## FPGA Timing

The test reuses the production workload's low-intrusion runtime timing path.
With `MOE_RUNTIME_TIMING=1`, every active individual-expert kernel records its
timing fields in the existing 64-byte Bingo task scratchpad. Device kernels do
not print while the DFG is running. After the scheduler returns, CVA6 scans the
records and prints the fixed-schema lines between `MOE_TIMING_BEGIN` and
`MOE_TIMING_END`.

Build the FPGA test image with runtime timing enabled and simulation-only trace
markers disabled:

```text
make -C target/sw clean
make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  CFG_OVERRIDE=target/rtl/cfg/hemaia_4cluster_singlechip.hjson \
  MOE_RUNTIME_TIMING=1 DEBUG_LEVEL=0 NODE_TIMING=0 PERF_TRACING=0
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_RUNTIME_TIMING=1 DEBUG_LEVEL=0 NODE_TIMING=0 PERF_TRACING=0
make bootrom
```

Analyze the captured FPGA UART log with the production parser and this test's
model dimensions:

```text
python3 target/sw/host/apps/offload_bingo_hw/single_chip/workloads/\
multi_cluster_MoE/analyze_moe_runtime_timing.py /path/to/uart.log \
  --params target/sw/host/apps/offload_bingo_hw/single_chip/workloads/\
multi_cluster_MoE_test/params.hjson --details
```
