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

The `s2pf_both` profile reuses the default two-slot production handoff and its
byte-exact checks. It changes only C0 slot0 S2PF from IDMA to `BOTH`; C1 slot0
still has no S2PF, and both slot1 bindings remain unchanged. It retains the
compact baseline input, two expert weight sets, and 68 KiB output per cluster.
Every checked output byte is written by the two slots, so setup does not clear
the output buffers before the device path starts.

The fixed-order scheduler experiments are separate workload profiles. The new
case-0 policy-0 workload is selected with `MOE_TEST_SCHEDULE=static_desc`.
Case-0 policy-1 uses `MOE_TEST_SCHEDULE=dynamic_desc`.
Case-0 policy-2 uses `MOE_TEST_SCHEDULE=dynamic_two_ended`.
Case-0 policy-3 uses `MOE_TEST_SCHEDULE=full_scheduler`.
Case-1 policy-0 uses `MOE_TEST_SCHEDULE=m70_three_hot_static_desc`.
Case-1 policy-1 uses `MOE_TEST_SCHEDULE=m70_three_hot_dynamic_desc`.
Case-1 policy-2 uses `MOE_TEST_SCHEDULE=m70_three_hot_dynamic_two_ended`.
Case-1 policy-3 uses `MOE_TEST_SCHEDULE=m70_three_hot_full_scheduler`.
Case-2 policy-0 uses `MOE_TEST_SCHEDULE=m92_parameter_order_static_desc`.
Case-2 policy-1 uses `MOE_TEST_SCHEDULE=m92_parameter_order_dynamic_desc`.
Case-2 policy-2 uses
`MOE_TEST_SCHEDULE=m92_parameter_order_dynamic_two_ended`.
Case-2 policy-3 uses
`MOE_TEST_SCHEDULE=m92_parameter_order_full_scheduler`.
Case-3 policy-0 uses `MOE_TEST_SCHEDULE=m60_high_skew_static_desc`.
Case-3 policy-1 uses `MOE_TEST_SCHEDULE=m60_high_skew_dynamic_desc`.
Case-3 policy-2 uses
`MOE_TEST_SCHEDULE=m60_high_skew_dynamic_two_ended`.
Case-3 policy-3 uses
`MOE_TEST_SCHEDULE=m60_high_skew_full_scheduler`.
The Bingo-rejection experiment uses
`MOE_TEST_SCHEDULE=m70_three_hot_dynamic_desc_skip_elided` with the exact same
policy-1 schedule.
The earlier experiments remain available with `MOE_TEST_SCHEDULE=high_to_low` or
`MOE_TEST_SCHEDULE=low_to_high` or `MOE_TEST_SCHEDULE=ends_inward`; the default
remains `baseline`:

```text
make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=s2pf_both MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=s2pf_both MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=static_desc MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=static_desc MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=dynamic_desc MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=dynamic_desc MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=dynamic_two_ended MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=dynamic_two_ended MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=full_scheduler MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=full_scheduler MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_static_desc MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_static_desc MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_dynamic_desc MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_dynamic_desc MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_dynamic_two_ended MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_dynamic_two_ended MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_full_scheduler MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_full_scheduler MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m92_parameter_order_static_desc MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m92_parameter_order_static_desc MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m92_parameter_order_dynamic_desc MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m92_parameter_order_dynamic_desc MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m92_parameter_order_dynamic_two_ended \
  MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m92_parameter_order_dynamic_two_ended \
  MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m92_parameter_order_full_scheduler \
  MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m92_parameter_order_full_scheduler \
  MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m60_high_skew_static_desc MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m60_high_skew_static_desc MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m60_high_skew_dynamic_desc MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m60_high_skew_dynamic_desc MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m60_high_skew_dynamic_two_ended \
  MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m60_high_skew_dynamic_two_ended \
  MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m60_high_skew_full_scheduler \
  MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m60_high_skew_full_scheduler \
  MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_dynamic_desc_skip_elided \
  MOE_RUNTIME_TIMING=1
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=m70_three_hot_dynamic_desc_skip_elided \
  MOE_RUNTIME_TIMING=1

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=high_to_low
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=high_to_low

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=low_to_high
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=low_to_high

make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=ends_inward
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=ends_inward
```

The `static_desc` profile is the first distribution's `STATIC_DESC` policy from
`scheduler_showcase_fpga_workloads.json`. It uses the exported 70-token routing
exactly. The global queue is high-to-low, but the generated runtime records have
no physical-policy choices: every task uses S1=B and S3=B, C2 uses IDMA for both
weight loads, C3 uses XDMA, and S2PF/S4PF are disabled. C2 receives
E0,E3,E4,E5,E7,...,E41 and ends at tick 159; C3 receives
E1,E2,E6,E8,...,E42 and ends at tick 162.

Applying the established 3/4-tick DFG/API allowance to each cluster-local slot
before taking the maximum gives:

```text
C2: 159 + 22 * 3/4 = 175.50 ticks
C3: 162 + 21 * 3/4 = 177.75 ticks
global lower bound = max(C2, C3) = 177.75 ticks
```

At 8192 cycles per focused-workload tick this is 1,456,128 cycles. The pure
four-stage policy result is 162 ticks; 177.75 ticks is the FPGA DFG/API-adjusted
structural reference. FPGA correctness and measured runtime still require the
output checks and timing records from a completed run.

The `dynamic_desc` profile uses the same exported distribution, token routing,
descending issue order, and cluster assignment as `static_desc`, but preserves
the policy's dynamic physical decisions. E0 and E1 use A/B with cluster-local
S2PF. E1 preloads E2 S1 through XDMA, and E3 preloads E4 S1 through BOTH; E2
and E4 therefore skip S1. E5 through E21 use B/B with BOTH S1 and BOTH S2PF.
E22 uses C/C with BOTH S1 and S3. The remaining tail uses B/B with each
cluster's local DMA lane. The 21 cross-cluster stage-level DMA release edges
preserve the exported IDMA/XDMA operation order without adding a slot barrier.

Both model streams finish at tick 159. Applying the same per-cluster DFG/API
allowance gives:

```text
C2: 159 + 22 * 3/4 = 175.50 ticks
C3: 159 + 21 * 3/4 = 174.75 ticks
global lower bound = max(C2, C3) = 175.50 ticks
```

At 8192 cycles per focused-workload tick this is 1,437,696 cycles. The pure
four-stage policy result is 159 ticks; 175.50 ticks is the FPGA DFG/API-adjusted
structural reference.

The `dynamic_two_ended` profile replays case-0 policy-2. C2 repeatedly takes the
hottest remaining expert and executes E0 through E11. C3 independently takes
the coldest remaining expert and executes E42 through E12. There is no global
slot barrier. The profile preserves all exported physical choices, including
19 early S2 prefetches, E21's BOTH S4 prefetch for E20, E20's skipped S1, and
the 26 cross-cluster DMA release edges.

The pure four-stage streams end at tick 137 on C2 and tick 134 on C3. Applying
the established per-slot DFG/API allowance independently to each cluster gives:

```text
C2: 137 + 12 * 3/4 = 146.00 ticks
C3: 134 + 31 * 3/4 = 157.25 ticks
global lower bound = max(C2, C3) = 157.25 ticks
```

At 8192 cycles per focused-workload tick this is 1,288,192 cycles. The policy's
ideal model makespan is 137 ticks; 157.25 ticks is the FPGA DFG/API-adjusted
structural reference.

The `full_scheduler` profile replays case-0 policy-3. The certified scheduler
jointly chooses issue order, cluster assignment, shape, DMA binding, and S2PF.
C2 executes 11 slots and C3 executes 32 slots. E0, E1, and E2 use IDMA S2PF;
E0 and E1 use the early S1-overlap event, while E2 starts S2PF at S1 end. The
remaining cold tail uses C/C with BOTH on C3, and the replay contains six
cross-cluster DMA release edges.

Both pure four-stage streams end at tick 129. Applying the same per-slot
DFG/API allowance independently to each cluster gives:

```text
C2: 129 + 11 * 3/4 = 137.25 ticks
C3: 129 + 32 * 3/4 = 153.00 ticks
global lower bound = max(C2, C3) = 153.00 ticks
```

At 8192 cycles per focused-workload tick this is 1,253,376 cycles. The
certified scheduler-model lower bound is 129 ticks; 153 ticks is the current
FPGA DFG/API-adjusted structural reference.

The `m70_three_hot_static_desc` profile is the second distribution's policy-0
replay. Its exact distribution is `[28x3, 6x4, 2x16, 0x41]`, with 70 source
tokens and 140 Top-2 routes. Every task uses B/B, C2 uses IDMA, C3 uses XDMA,
and S2PF/S4PF are disabled. C2 executes E0,E2,E8,E10,...,E22 in 10 local slots
and ends at tick 132. C3 executes E1,E3,E4,E5,E6,E7,E9,...,E21 in 13 local
slots and ends at tick 126.

The model makespan is therefore 132 ticks. Applying the established 3/4-tick
DFG/API allowance separately to the two cluster-local streams gives:

```text
C2: 132 + 10 * 3/4 = 139.50 ticks
C3: 126 + 13 * 3/4 = 135.75 ticks
global lower bound = max(C2, C3) = 139.50 ticks
```

At 8192 cycles per focused-workload tick, the model makespan is 1,081,344
cycles and the production structural lower bound is 1,142,784 cycles.

The `m70_three_hot_dynamic_desc` profile replays case-1 policy-1 with the same
distribution, exact routing, descending issue order, and dynamic physical
selection. E0 and E1 use early S2PF and preload E2 and E3 respectively. E3
through E6 continue the S4PF chain through E7, so E2 through E7 skip S1. E8
uses C/C with BOTH, and the remaining two-token tail returns to B/B with the
cluster-local DMA lane. Four cross-cluster DMA release edges preserve the
exported IDMA/XDMA operation order without adding a global slot barrier.

Both model streams end at tick 126. C2 has 9 local slots and C3 has 14. Applying
the established per-slot DFG/API allowance independently to each cluster gives:

```text
C2: 126 + 9 * 3/4 = 132.75 ticks
C3: 126 + 14 * 3/4 = 136.50 ticks
global lower bound = max(C2, C3) = 136.50 ticks
```

At 8192 cycles per focused-workload tick, the model makespan is 1,032,192
cycles and the production structural reference is 1,118,208 cycles.

The `m70_three_hot_dynamic_desc_skip_elided` experiment keeps those queues,
DMA bindings, prefetches, and release edges unchanged. It models Bingo rejecting
the three statically empty S1 stage tasks before dispatch for each `skip_s1`
slot: 3 stage nodes disappear on C2 and 15 on the critical C3 stream. All S2,
S3, S4, prefetch/prepare, and store nodes remain unchanged so this run isolates
only the cost of issuing the six cache-hit S1 skips.

The `m60_high_skew_static_desc` profile replays case-3 policy-0. Its exact
64-expert distribution is `[36,22,13,6,2x17,1x9,0x34]`, with 60 input tokens,
120 Top-2 routes, and 30 active experts. Every task uses B/B without prefetch;
C2 has 14 IDMA slots and ends at tick 135, while C3 has 16 XDMA slots and ends
at tick 138. The dedicated lanes require no cross-cluster DMA release edges.

```text
C2: 135 + 14 * 3/4 = 145.50 ticks
C3: 138 + 16 * 3/4 = 150.00 ticks
global lower bound = max(C2, C3) = 150.00 ticks
```

At 8192 cycles per focused-workload tick, the model makespan is 1,130,496
cycles and the production structural lower bound is 1,228,800 cycles.

The `m60_high_skew_dynamic_desc` profile replays case-3 policy-1 with the same
distribution and exact Top-2 routing. E0/E1 use early S2PF; their S4PF events
preload E3/E2, and E2 preloads E4. Therefore E2, E3, and E4 skip S1. The
remaining tail uses C/C with BOTH and follows the exported shared-DMA order.
Its 26 cross-cluster DMA release edges are lowered directly into the DFG.

```text
C2: 133 + 14 * 3/4 = 143.50 ticks
C3: 130 + 16 * 3/4 = 142.00 ticks
global lower bound = max(C2, C3) = 143.50 ticks
```

At 8192 cycles per focused-workload tick, the model makespan is 1,089,536
cycles and the production structural lower bound is 1,175,552 cycles.

The `m60_high_skew_dynamic_two_ended` profile replays case-3 policy-2. C2
executes the hot-end stream E0,E1,E2, while C3 executes the cold-end stream
E29,E28,...,E3. E1 and E2 use early S2PF; E3 uses S2PF after its B-shape S1
DMA completes. The profile has no S4PF or cache-hit task, and its six exact
cross-cluster DMA release edges are lowered into the DFG.

The pure four-stage streams end at tick 111 on C2 and tick 94 on C3. Applying
the established per-slot DFG/API allowance independently to each cluster gives:

```text
C2: 111 + 3 * 3/4 = 113.25 ticks
C3: 94 + 27 * 3/4 = 114.25 ticks
global lower bound = max(C2, C3) = 114.25 ticks
```

At 8192 cycles per focused-workload tick, the model makespan is 909,312 cycles
and the production structural lower bound is 935,936 cycles.

The `m60_high_skew_full_scheduler` profile replays case-3 policy-3. C2 executes
E0,E13,E12,...,E6,E2, while C3 executes E29,E28,...,E14,E1,E5,E4,E3. E0 uses
IDMA S2PF; E1 and E2 use BOTH S2PF. The profile has no S4PF or cache-hit task,
and its three exact cross-cluster DMA release edges are lowered into the DFG.

Both pure four-stage streams end at tick 99. Applying the established per-slot
DFG/API allowance independently to each cluster gives:

```text
C2: 99 + 10 * 3/4 = 106.50 ticks
C3: 99 + 20 * 3/4 = 114.00 ticks
global lower bound = max(C2, C3) = 114.00 ticks
```

At 8192 cycles per focused-workload tick, the model makespan is 811,008 cycles
and the production structural lower bound is 933,888 cycles.

The high-to-low profile uses the exact 64-expert distribution
`[22, 18, 14, 3x19, 2x8, 1x13, 0x21]`. The left-to-right issue order is split
by replaying `fixed_orders.descending` from
`scheduler_rtl_distilled_showcase.json`; it is not an even/odd static split.
The first concurrent pair is E0(22) on C0 and E1(18) on C1. C1 then runs
E2(14) at tick 27 while C0 runs E3(3) at tick 33. E0/E1/E2 use A/B with
S2PF; E3 uses A/B with an ordinary S3-B path. E4 through E21 use B/B, and
the special E22(2) also uses B/B. E23 waits
for E22's S3 DMA release. E23 through E42 use C/C and form the serialized
BOTH-DMA tail. This reference history does not
use S4PF. Logical expert IDs remain
distinct, but all 64 IDs map to the single physical synthetic weight backing.

Both slot0 gathers wait for the same one-time setup release. After that point,
each cluster advances from its own previous store. The E22-to-E42 tail adds
only cross-cluster DMA-release edges from one S3 load to the next S1 load; it
does not add a whole-slot barrier or delay the next task's compute-side CSR
preconfiguration. The schedule manifest checks C0=163 and C1=160, hence a
left-to-right makespan of 163 abstract scheduler ticks. With the focused
2048x1024 dimensions, one abstract tick is 8192 cycles and the model makespan
is 1,335,296 cycles.

The low-to-high profile replays `ascending` from the FPGA showcase handoff for
the same distribution and input data. C0 executes
E41,E39,...,E3,E1 and the first E0 slice; C1 executes
E42,E40,...,E4,E2 and the second E0 slice. Each cluster has 22 local slots.
The final SPLIT gives C0 E0 token ranks 0 through 10 and C1 ranks 11 through
21. The generated golden checks use those same disjoint ranges. E1 uses BOTH
for S2 prefetch, E2 retains its explicit C/C xDMA-S1 and iDMA-S3 bindings, and
the two E0 slices use their cluster-local single-DMA S2 prefetches. E0 starts
only after both E1 and E2 have stored their outputs; there is no per-round
global barrier elsewhere.

The exported four-stage replay ends at C0=165 and C1=165, so its fixed-order
model makespan is 165 ticks. For FPGA comparison, use the same structural/API
allowance already used for high-to-low: 3/4 tick for every slot on the critical
cluster-local stream. Therefore the theoretical reachable lower bound is

```text
165 + 22 * 3/4 = 181.5 ticks
```

This makes low-to-high 2.0 ticks, or 1.11%, above the corresponding 179.5-tick
high-to-low bound. Under this focused FPGA workload's existing 8192-cycle tick,
181.5 ticks is 1,486,848 cycles. This conversion deliberately does not use the
handoff model's original 11,264-cycle tick. The values 165 and 181.5 are
model-derived references; only an FPGA run with output checks and timing
records can establish measured correctness and runtime.

The ends-inward profile replays the handoff's `ends_inward` method. C0 executes
E0,E41,E2 and then E4 through E21. C1 executes E42,E1,E40,E3 and then E39
through E22. It contains 43 tasks with no SPLIT: C0 has 21 local slots and C1
has 22. The early tasks preserve the exported nonuniform profiles: E41 is C/B
with BOTH S1 and BOTH S2 prefetch; E42 and E40 use A/B with xDMA S1 and BOTH
S3; E3 uses A/C with BOTH S1 and iDMA S3. Seven stage-level cross-cluster DMA
release edges preserve the exported IDMA/XDMA order. They do not introduce a
global slot barrier.

The fixed four-stage replay reaches tick 58 after the early special tasks. C0
then executes E4 through E21 as 18 six-tick tasks, while C1 executes E39 through
E22 as 18 six-tick tasks. Both cluster streams therefore end at
`58 + 18*6 = 166` ticks. Applying the same per-slot structural/API allowance
separately gives:

```text
C0: 166 + 21 * 3/4 = 181.75 ticks
C1: 166 + 22 * 3/4 = 182.50 ticks
global lower bound = max(C0, C1) = 182.5 ticks
```

This is 1.0 tick above low-to-high's 181.5-tick bound and 3.0 ticks above
high-to-low's 179.5-tick bound. At 8192 cycles per focused-workload tick, the
ends-inward lower bound is 1,495,040 cycles. These remain model-derived
references until an FPGA run passes both output checks and timing validation.

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

The test reuses the production workload's runtime timing path.
`MOE_RUNTIME_TIMING=1` records only the two cluster-local begin/end pairs for a
low-overhead global time. `MOE_RUNTIME_TIMING=2` records every static DFG task,
including skipped tasks, for slot/stage/task diagnosis. Device kernels do not
print while the DFG is running. After the scheduler returns, CVA6 prints the
schema-v3 records between `MOE_TIMING_BEGIN` and `MOE_TIMING_END`.

Build the FPGA test image with low-overhead timing and simulation-only trace
markers disabled. Substitute `MOE_RUNTIME_TIMING=2` in both commands when a
full diagnostic capture is required:

```text
make -C target/sw clean
make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  CFG_OVERRIDE=target/rtl/cfg/hemaia_4cluster_singlechip.hjson \
  MOE_TEST_SCHEDULE=ends_inward \
  MOE_RUNTIME_TIMING=1 DEBUG_LEVEL=0 NODE_TIMING=0 PERF_TRACING=0
make apps HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE_test DEV_APP=snax-bingo-offload \
  MOE_TEST_SCHEDULE=ends_inward \
  MOE_RUNTIME_TIMING=1 DEBUG_LEVEL=0 NODE_TIMING=0 PERF_TRACING=0
make bootrom
```

Analyze the captured FPGA UART log with the production parser and this test's
model dimensions:

```text
python3 target/sw/host/apps/offload_bingo_hw/single_chip/workloads/\
multi_cluster_MoE/analyze_moe_runtime_timing.py /path/to/uart.log --details
```

The analyzer computes one wrap-safe local span per cluster and reports their
maximum as global time. It never subtracts timestamps from different clusters.
It prints active-slot S1/S2/S3/S4/store wall times and, with `--details`, every
DFG task. It does not infer gaps or idle time from uncovered intervals.
