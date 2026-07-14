# multi_cluster_MoE workload

## Source layout

- `params.hjson`: the only model/workload parameter input.
- `moe_l15_layout.py`: derives every tensor dimension, byte size, stride and
  L15 offset. Both generators import this module.
- `multi_cluster_MoE_datagen.py`: emits runtime input, packed weights and the
  named 91-word fused-L15 streamer configuration.
- `main_bingo.py`: emits the static Bingo DFG and all generated kernel args.
- `multi_cluster_MoE_config.h`: generated compile-time HW-path dimensions;
  included before `host.h` so the host kernels compile directly against the
  selected workload shape.
- `moe_router_host.h`: software-router implementation; excluded from a HW build.
- `moe_scheduler.c/.h`: software scheduler; emits no scheduler functions in a
  HW build.
- `moe_scheduler_hw_mmio.h`: HW scheduler MMIO register/protocol definitions.
- `host_moe_hw_path.h`: HW FIFO drain and direct compact-plan lowering.
- `host_moe_sw_path.h`: software schedule lowering only.

`multi_cluster_MoE_data.h`, `multi_cluster_MoE_config.h` and
`offload_bingo_hw.h` are generated files. Do not edit them directly. They are
ignored by Git and removed by the workload clean target.

## Parameter flow

Edit only `params.hjson`. A rebuild passes it, together with the selected SNAX
hardware configuration, to `derive_workload_params()`. This updates datagen,
L3/L1 allocations, DMA sizes, GEMM dimensions, streamer configuration and DFG
args in one step.

The current physical token row is:

```text
hidden_size INT16 payload + 32-byte zero padding
1024 elements * 2 bytes + 32 bytes = 2080 bytes
```

VersaCore reads/writes the payload and skips the 32-byte tail. Four static init
nodes initialize the output tails once before shared/individual compute. The
runtime compute path does not clear output regions. Individual gather clears
only hardware-minimum-M rows that do not correspond to a routed token.

Changing dimensions keeps the functional ABI synchronized. The RTL scheduler
cost constants are calibrated separately; recalibrate them when a new model
dimension must also produce cycle-optimal scheduling.

## Runtime argument flow

The pure-HW host path drains each compact RTL plan word directly into its final
L3 dynamic-slot record. `MoEExecute` writes the two runtime headers and copies
the used C2/C3 slot ranges to L1; it does not rebuild another task or schedule
representation.

On the device, each static Bingo node carries only a dynamic-slot address, a
shared static-record address and, for block nodes, the block index. Weight
nodes compute the final source/destination offsets and call the DMA dispatcher
directly. Compute nodes consume the dynamic call fields plus the shared static
record and configure SwiGLU/down directly. There is no intermediate pre-args
structure or one-use forwarding helper between the slot record and the actual
DMA/GEMM configuration.

The final shared static record is generated once per cluster during host
initialization and copied to L1 separately from the dynamic slot records. This
avoids duplicating invariant addresses and dimensions in every slot. The named
91-word fused-L15 structure is also intentional: it is the typed configuration
record copied to TCDM and consumed directly, not a positional conversion layer.

## Compute API paths

- Router: `__snax_bingo_kernel_moe_router_gemm_s0` consumes one typed router
  arg record and directly invokes the inlined dual-VC Mode-1 implementation.
- Shared experts: `__snax_bingo_kernel_dual_vc_l15_moe_full` consumes the
  named L15 config in L1, completes Mode-0 SwiGLU, then configures and runs
  Mode-1 down projection. Its output uses padded token rows.
- Individual experts: static Bingo nodes point at one final dynamic slot plus
  one cluster-shared static context. DMA nodes derive weight/token addresses
  from those two records; compute nodes directly program the relevant S1/S2/
  S3/S4 call. No task/request ABI is reconstructed on the device.
- Ordinary dual-VC GEMM/SwiGLU: the generic streamer helper retains the typed
  parameter API, but its fixed CSR loops are expanded and the operation helper
  is forced inline. It therefore supports non-L15 callers without adding a
  runtime forwarding call.

## Software builds

Run in the `hemaia_main_dev` environment from the HeMAiA repository root.

Pure HW scheduler path:

```sh
make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE DEV_APP=snax-bingo-offload \
  CFG_OVERRIDE=target/rtl/cfg/hemaia_4cluster_singlechip.hjson \
  MOE_HW_SCHEDULER=1
```

Pure software comparison path: use the same command with
`MOE_HW_SCHEDULER=0`.

`DEBUG_LEVEL` defaults to `0`; use `DEBUG_LEVEL=1` only for value dumps.
`PERF_TRACING` defaults to `1`; use `PERF_TRACING=0` to remove NOP trace
markers. `MOE_MCYCLE_DETAIL=1` enables detailed end-of-run counters. These
instrumentation choices are not required for functional execution and can
perturb timing. There is no HW/SW check, legacy or runtime fallback build mode.

## End-of-run runtime timing

Add `MOE_RUNTIME_TIMING=1` to collect trace-like timing without printing from a
running device kernel. Every active individual stage writes a compact record
to its existing 64-byte scratchpad. After the complete DFG exits, CVA6 only
prints the raw fixed-schema records needed by the offline analyzer.

```sh
make single-sw HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=single_chip \
  WORKLOAD=multi_cluster_MoE DEV_APP=snax-bingo-offload \
  CFG_OVERRIDE=target/rtl/cfg/hemaia_4cluster_singlechip.hjson \
  MOE_HW_SCHEDULER=1 MOE_RUNTIME_TIMING=1 \
  DEBUG_LEVEL=0 NODE_TIMING=0 PERF_TRACING=0
```

Analyze an FPGA UART log with:

```sh
python3 analyze_moe_runtime_timing.py /path/to/uart.log
```

The default report contains host phase timing, slot windows, cluster windows,
and C2/C3 iDMA, xDMA, and VersaCore utilization. Every resource utilization uses
the full cluster window, from the first node of the first active slot to the last
node of the final assigned slot. Add `--details` to print each slot's node and
resource start/end timestamps. A `dma_both` interval contributes to both DMA
resource timelines.

With the flag absent, all device recording macros, the generated address table,
the final scan, and all profile strings are removed by preprocessing.
