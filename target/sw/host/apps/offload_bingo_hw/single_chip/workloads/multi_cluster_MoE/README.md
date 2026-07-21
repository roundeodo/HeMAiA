# multi_cluster_MoE workload

## Source layout

- `params.hjson`: the only model/workload parameter input.
- `moe_layout.py`: separates common dimensions, the legacy L1.5 comparison
  layout, and the current bank-partition addresses and strides.
- `multi_cluster_MoE_datagen.py`: emits dense token input and canonical packed
  weight panels. L1 bank placement is performed by 2D DMA at runtime.
- `main_bingo.py`: emits the static Bingo DFG and all generated kernel args.
- `multi_cluster_MoE_config.h`: generated compile-time HW-path dimensions;
  included before `host.h` so the host kernels compile directly against the
  selected workload shape.
- `moe_router_host.h`: software-router implementation; excluded from a HW build.
- `moe_scheduler.c/.h`: software scheduler; emits no scheduler functions in a
  HW build.
- `moe_scheduler_hw_mmio.h`: HW scheduler MMIO register/protocol definitions.
- `host_moe_hw_path.h`: HW FIFO drain plus CVA6 lowering from each RTL task
  word into the complete device call records.
- `host_moe_sw_path.h`: pure-SW schedule traversal plus the same CVA6-side
  complete call-record generation.

`multi_cluster_MoE_data.h`, `multi_cluster_MoE_config.h` and
`offload_bingo_hw.h` are generated files. Do not edit them directly. They are
ignored by Git and removed by the workload clean target.

## Parameter flow

Edit only `params.hjson`. A rebuild passes it, together with the selected SNAX
hardware configuration, to `derive_bank_workload_params()`. This updates
datagen, L3/L1 allocations, DMA sizes, GEMM dimensions, streamer configuration
and DFG args in one step.

With the checked-in parameters the model is `A=[8,2048]`,
`W/V=[2048,1024]` and logical `W2=[1024,2048]`. `total_tokens` is
parameterized and need not be a multiple of eight. L3 token rows are dense, so
row stride equals `hidden_size*sizeof(INT16)` and contains no 32-byte padding.

Token staging issues one 2D descriptor per valid token with `size=16`,
`src_stride=16`, `dst_stride=512`, and `repeat=hidden_size/8`. For local token
`t`, `lane=t%8` selects banks `2*lane..2*lane+1`, while `page=t/8` selects a
non-overlapping row-depth region. The address is
`arena + page*(token_bytes/16)*512 + lane*16`. Thus token 8 returns to banks
0..1 at the next page instead of overwriting token 0.

An individual scheduler task gathers its complete `ntokens` slice once. S1/S2
and S3/S4 may use several VersaCore launches to cover that slice, but these are
internal compute tiles of the same task, not separate token-transfer waves.
Call records carry task-local logical token starts; the device derives physical
page addresses from a fixed L1 arena base, so there is no mutable global current
address.

Weights use 64-byte 2D rows with `src_stride=64` and `dst_stride=512`.
Mode-0 W/V use banks 16..47 in the first 4 MiB; Mode-1 W2L/W2R use the same
bank groups in the next 2 MiB. Mode-0 output uses banks 48..63 and Mode-1
output uses banks 0..15. Every arena base is explicitly aligned to 512 bytes.

`weight_chunk_cols` is the only block-granularity input. The generator derives
`s1_block_count=intermediate_size/weight_chunk_cols` and
`s3_block_count=(hidden_size/2)/weight_chunk_cols`. The dynamic ABI reserves
eight S1 and eight S3 call entries, so supported configurations change only
generated loop bounds and DFG node count, not the C ABI layout.

S1 and S3 keep separate generated counts because they partition different
matrix axes: S1 covers the full intermediate dimension, while each S3 VC
covers one half of the hidden output dimension. They are both eight for the
checked-in `2048 -> 1024 -> 2048` model, but that equality is not an ABI
requirement. The fused shared kernel reads both precomputed counts directly;
it performs no block-count division at run time. Within one stage it programs
the complete streamer and VersaCore state only for block 0. Later S1 blocks
patch three base CSRs and later S3 blocks patch four base CSRs before launch.

The production 2D weight-pair staging API likewise programs the invariant DMA
shape once per node. It submits all block descriptors before one final wait;
the iDMA and xDMA halves of a mixed binding remain concurrent. These fast paths
are used by the full workload and the API test workload alike.

Changing dimensions keeps the functional ABI synchronized. The RTL scheduler
cost constants are calibrated separately; recalibrate them when a new model
dimension must also produce cycle-optimal scheduling.

## Runtime argument flow

The pure-HW host path decodes each compact RTL task word directly into a
344-byte CVA6-lowered task record stored in a 384-byte L3/L1 slot. This record
is not the raw RTL descriptor: CVA6 resolves skip/shape/DMA/S4PF semantics and
materializes the final S1/S2/S3/S4 call records, including addresses, M/N and
array shape.
`MoEExecute` writes the two runtime headers and copies the used C2/C3 argument
ranges to L1; it does not rebuild another task or schedule representation. The
per-cluster static context uses an independent 192-byte L1 ABI slot, so changing
the dynamic argument record cannot truncate it. This context is initialized
once and is not part of each batch's dynamic flush.

On the device, each static Bingo node carries only a dynamic-slot address, a
shared static-record address and, for block nodes, the block index. Weight DMA
nodes derive only the selected expert/block weight address because the selected
expert and prefetch target remain dynamic. Compute nodes do not derive shape,
dimensions or input/output addresses: they consume the complete CVA6-generated
call record and configure SwiGLU/down directly. There is no intermediate
pre-args structure or one-use forwarding helper between the slot record and the
actual DMA/GEMM call.

The final shared static record is generated once per cluster during host
initialization and copied to L1 separately from the dynamic slot records. This
avoids duplicating invariant addresses and dimensions in every slot. Streamer
CSRs are generated directly from the parameterized bank layout; no L15
configuration record is copied to TCDM.

## Post-routing metadata status

The production HW path emits token-major TopK scores and an expert-major packed
token-reference table. Each 16-bit reference stores `token_id[14:0]` and the
TopK rank in bit 15. Individual gather masks off the rank; future final weighted
accumulation can use it to select `probability[token_id][rank]`. A separate
runtime `kpos` table and a second scatter-metadata pass are therefore not
required. Token-major expert IDs remain debug/SW-path data only.

Delayed softmax and final weighted accumulation nodes are not yet inserted into
this workload DFG. Their required metadata is now produced on the production HW
path without changing the RTL scheduler descriptor, `MoEPrepare`, or
`MoEExecute`; the only added RouterSched store is two int32 scores per token.

## Compute API paths

- Router: `__snax_bingo_kernel_moe_router_gemm_s0` consumes one typed router
  arg record and directly invokes the inlined dual-VC Mode-1 implementation.
  It temporarily uses a private `router_input_A` compatibility copy because
  this legacy kernel still encodes the historical 32-byte row gap; expert
  paths use dense `input_A` exclusively.
- Shared experts: `__snax_bingo_kernel_dual_vc_bank_moe_full` consumes the
  512-byte-aligned bank arena, completes Mode-0 SwiGLU, then configures and runs
  Mode-1 down projection. `__snax_bingo_kernel_moe_store_tokens_2d` gathers the
  two output halves into one dense L3 token-major matrix using local xDMA.
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
the individual-expert compute efficiency, and C2/C3 iDMA, xDMA, and VersaCore
resource utilization. Compute efficiency is useful ideal compute cycles divided
by the timespan from the first active slot start to the last active slot end. Its
numerator is `sum(ntokens) * 3 * hidden_size * intermediate_size` MAC divided by
the two-cluster theoretical peak; measured `vc_api`/`vc_hw` busy cycles are only
resource diagnostics. Add `--details` to print each slot's node and resource
start/end timestamps. A `dma_both` interval contributes to both DMA resource
timelines.

With the flag absent, all device recording macros, the generated address table,
the final scan, and all profile strings are removed by preprocessing.
