# MoE Hardware-Friendly Schedule Descriptor Design

## 1. 目标

当前 workload 的 MoE 调度器负责把 router 之后的 per-expert token count 转成硬件可执行的调度描述。硬件控制逻辑相对固化，因此最终下发格式必须满足：

- **硬件不做全局搜索**，只执行软件生成的描述符。
- **硬件不 join 两张动态表**，尽量避免按 `task_idx` 查找 `dma_ops[]`。
- 每个 task 的控制结构固定，硬件只看固定字段和 `valid/skip` bit。
- 后台 prefetch 必须显式化，不能只依赖 `prefetch_eid` hint。

因此调度器内部可以保留 `tasks[] + dma_ops[]` 作为软件 IR 和调试视图，但真正面向硬件的主接口是 **每个 task 内嵌固定 DMA slot**。

## 2. 当前接口位置

相关文件：

- `moe_scheduler.h`
- `moe_scheduler.c`

当前 `moe_schedule_t` 包含：

```c
moe_task_t tasks[MOE_MAX_TASKS];
moe_dma_op_t dma_ops[MOE_MAX_DMA_OPS];
uint16_t n_tasks;
uint16_t n_dma_ops;
uint32_t est_makespan_cc;
```

其中：

- `tasks[]` 是硬件主消费表。
- `dma_ops[]` 是软件 IR / 调试表，按绝对时间记录所有 DMA 操作。
- 每个 `moe_task_t` 内部新增 `dma_slots[4]`，这是建议硬件真正消费的固定 DMA slot。

## 3. 每个 task 的固定执行语义

调度器输出的每个 `moe_task_t` 表示一次 **scheduled expert run**：在一个 cluster 上执行某个 expert 的一段 routed-token 区间。通常一个 expert 对应一条 task；当调度器选择 `SPLIT` 时，同一个 expert 会按 token 维度拆成两条 task，例如 C2 负责前一段 token，C3 负责后一段 token。这里的 split 不是把 S1/S2/S3/S4 拆成四条 task，也不是按 N 方向权重 block 拆 task。

因此应当区分四层对象：

| 层次 | 对象 | 数量由什么决定 | 是否直接发给 device |
| --- | --- | --- | --- |
| 调度记录 | `tasks[]` 中的一条 `moe_task_t` | scheduler 对 expert/token 的全局调度结果 | 是，host lowering 后写入 cluster-local dynamic arg |
| 分析 stage | S1/S2/S3/S4 | 分析模型的 gate/up、down 时间窗口 | 否，只作为 shape/时间/依赖语义来源 |
| DMA metadata | `dma_slots[0..3]` | 每条 task 固定 4 个 DMA/prefetch 角色 | 是，写入 dynamic arg 的 fixed fields |
| Static DFG node | `load_gate_up_block[n]` / `compute_gate_up_block[n]` / `load_down_block[n]` / `compute_down_block[n]` | 静态 `indiv_N2` / `indiv_down_N2` | 是，Bingo 调度这些 kernel node |

换言之，一条 task 内部包含完整的 gate/up/SwiGLU 和 down projection 执行流程；S1/S2/S3/S4 是这条 task 的分析时序分段，不是四条 task。

调度器沿用分析模型中的 S1/S2/S3/S4 命名，但这里的 stage 不是 N 方向的 weight tile，也不是最终 Bingo DFG 里的 node 数量。它们描述的是 **按 token 数切分的计算时间窗口**：

```text
S1: gate/up 的初始 token window，窗口大小由 shape_s1 的 M_dim 决定
S2: gate/up 的剩余 token window，调度模型用 Shape C 分块估计
S3: down projection 的初始 token window，窗口大小由 shape_s3 的 M_dim 决定
S4: down projection 的剩余 token window，调度模型用 Shape C 分块估计
```

例如 Shape A/B/C 的初始窗口分别覆盖 8/4/2 个 token。因此 S1/S3 不是 “first tile”，而是 “first token window”。Phase4 进一步把 gate/up 和 down 权重按 `N2`/`N2_down` 拆成 block-level DFG，这是另一层 lowering。

硬件不需要重新推导 S1/S2/S3/S4 的 shape 选择。硬件只需要按 host 已经展开好的 task arg 和 fixed DMA slot 执行；Phase4 的实际 Bingo DFG 是 resident block layout。S1/S2 在 device 侧表现为 “gate/up 权重 block load + gate/up/SwiGLU block compute”，S3/S4 表现为 “down 权重 block load + down projection block compute”：

```text
gather current task tokens
for n in gate/up N-blocks:
    if dma_slots[S1].valid: load_gate_up_block[n] moves gate/up weight block n into L1_B_gate/up[n]
    compute_gate_up_block[n] runs gate/up/SwiGLU over this task's token slice
if dma_slots[S2_PREFETCH].valid: prefetch down weight
for n in down N-blocks:
    if dma_slots[S3].valid: load_down_block[n] moves down weight block n into L1_B_down[n]
    compute_down_block[n] runs down projection over this task's token slice
if dma_slots[S4_PREFETCH].valid: prefetch next task gate/up weight
store current task output and mark local slot complete
```

这里的 `load_*` kernel 只做 DMA，这是有意设计：DM core 和 GEMM/VersaCore core 是不同执行资源，把 load 与 compute 拆成不同 node 后，Bingo 才能表达 `load block n+1` 与 `compute block n` 的重叠。`skip_dma_s1` 和 `skip_dma_s3` 只表示是否跳过 foreground DMA，不表示跳过 compute。

## 4. 固定 DMA slot 定义

每个 `moe_task_t` 固定包含 4 个 DMA slot。这里的 slot 是 **DMA/prefetch metadata slot**，不是 compute stage，也不是 Bingo node。slot 编号只是为了让 host/device 用固定字段传递 DMA 角色：

```c
#define MOE_TASK_DMA_SLOT_S1          0u
#define MOE_TASK_DMA_SLOT_S3          1u
#define MOE_TASK_DMA_SLOT_S2_PREFETCH 2u
#define MOE_TASK_DMA_SLOT_S4_PREFETCH 3u
```

每个 slot 的结构为：

```c
typedef struct {
    uint8_t valid;
    moe_dma_op_kind_t kind;
    moe_weight_kind_t weight;
    int16_t expert_id;
    moe_shape_t shape;
    uint16_t alloc_bw;
    moe_dma_binding_t dma;
    uint32_t start_cc;
    uint32_t end_cc;
} moe_task_dma_slot_t;
```

### Slot 0: S1 foreground DMA

- `kind = MOE_DMA_OP_S1`
- `weight = MOE_WEIGHT_GATE_UP`
- 搬当前 task 的 gate/up 权重。若 Phase4 使用 block-level DFG，它会被拆成 `load_gate_up_block[n]`，每个 block 写入 resident L1_B_gate/up 的第 n 块。
- 若 `valid=0`，说明 S1 权重已经 resident 或被前序 prefetch 准备好。

### Slot 1: S3 foreground DMA

- `kind = MOE_DMA_OP_S3`
- `weight = MOE_WEIGHT_DOWN`
- 搬当前 task 的 down 权重，Phase4 中会拆成 `load_down_block[n]` 写入 resident L1_B_down layout。
- 若 `valid=0`，说明 down 权重已经 resident，或者当前 task 已经通过 Slot 2 的 S2 down-prefetch 准备好。

### Slot 2: S2 down-prefetch

- `kind = MOE_DMA_OP_S2_PREFETCH`
- `weight = MOE_WEIGHT_DOWN`
- 在当前 task 的 gate/up 计算窗口内提前搬 **当前 expert** 的 down 权重。它是 DMA prefetch 角色，不代表有一个额外的 S2 compute DMA 节点。
- 若该 slot 有效，通常当前 task 的 `skip_dma_s3=1`。
- 该 slot 是为了把 Slot 1 的 foreground S3 DMA 转成后台 prefetch，降低 down projection 启动前的等待。

### Slot 3: S4 next-S1 prefetch

- `kind = MOE_DMA_OP_S4_PREFETCH`
- `weight = MOE_WEIGHT_GATE_UP`
- 在当前 task 的 down projection 计算窗口内提前搬后续 expert 的 gate/up 权重。
- `expert_id` 是被预取的下一个 expert，不一定等于当前 task 的 `expert_id`。
- 这是 **S1-only prefetch**，不能被解释成 full cache；后续 task 只能跳过 S1 DMA，不能因此跳过 S3 DMA。

## 5. 当前 workload 落地方式

本轮实现已经把 Phase4 从“全局 task index 直接对应 slot”改成 **cluster-local slot lowering**。

相关文件：

- `main_bingo.py`
- `host_kernel_lib.h`
- `snax_kernel_lib.h`
- `bingo_kernel_args.py`
- `device_kernel_args.h`
- `host_kernel_args.h`

### 5.1 Host lowering

`__host_bingo_kernel_moe_execute()` 现在执行以下操作：

1. 如 `schedule->n_tasks == 0`，调用 `moe_schedule(request, schedule)`。
2. 清零 L3 runtime state：

```c
runtime_state[0] = C2 completed task count
runtime_state[1] = C3 completed task count
runtime_state[2] = global iDMA sequence
runtime_state[3] = global xDMA sequence
```

3. 遍历 `schedule->tasks[]`，按 `task.cluster` 分别写入 C2/C3 的本地 slot：

```text
C2 slot = 第几个 C2 task
C3 slot = 第几个 C3 task
```

这避免了旧实现中 `tasks[i] -> slot i` 导致 split/pair 被串行化的问题。

4. 对每个 task 计算跨 cluster 前置关系：

```text
wait_for_peer_slots = count(peer tasks whose est_end_cc <= this_task.est_start_cc)
```

device 端 slot 的 S1 kernel 会先等待 peer completed counter 达到该值。这样 early-start task 不会被无谓 round barrier 阻塞，而需要等待对端完成的 task 也不会提前启动。

5. 遍历已经按 `start_cc/end_cc` 排序并完成 lane binding 的 `schedule->dma_ops[]`，给每个 task 的 4 个 fixed DMA slot 写入：

```text
valid/kind/expert_id/shape/dma/start_cc/end_cc/idma_seq/xdma_seq
```

其中 `idma_seq` 和 `xdma_seq` 是每条物理 DMA lane 的严格发射序号。device 端每个 DMA/prefetch kernel 在发起 DMA 前等待对应 lane counter，完成后递增 counter。

### 5.2 Static Bingo DFG

每个 cluster 的 Phase4 static slot 链已经改成一套 block-level 静态图。外层仍保留调度器的 S1/S2/S3/S4 时间窗口语义，但 DFG 节点数由 workload 的 `indiv_N2` 和 `indiv_down_N2` 决定，而不是由 runtime shape 动态决定。

当前配置中 `indiv_N2=2`、`indiv_down_N2=2`，所以每个 dynamic slot 生成：

- 2 个 `load_gate_up_block[n]`，对应 gate/up 权重的 2 个 N-block DMA。
- 2 个 `compute_gate_up_block[n]`，对应 gate/up/SwiGLU 的 2 个 N-block compute。
- 2 个 `load_down_block[n]`，对应 down 权重的 2 个 N-block DMA。
- 2 个 `compute_down_block[n]`，对应 down projection 的 2 个 N-block compute。

如果配置改成更多 N-block，重新生成 DFG 后这些 block node 的数量会随 `indiv_N2` / `indiv_down_N2` 增加。`shape_s1` / `shape_s3` 不改变静态 node 数量；它们影响 runtime 使用的 token 维度 `M`、分析 timing 和 DMA lane sequence，而不是 N 方向 block 数。

`compute_gate_up_block[n]` 不是 “gate 一个节点、up 一个节点”，也不是只执行分析模型里的 S2 remainder。它是一次调用 `dual_vc_swiglu_full`，同时消费 gate/up 两组权重，覆盖当前 task 的 token slice，并产生第 n 个 SwiGLU 输出 block。`compute_down_block[n]` 则是第 n 个 down projection 输出 block。

因此，分析模型里的 S1/S2 与物理 DFG 不是一一对应关系：S1 是 gate/up 的初始 token-window timing，S2 是剩余 token-window timing；物理实现为了利用 DM/GEMM 并行度，按 N-block 生成 gate/up load/compute 链。S3/S4 与 down projection 同理。

静态图结构如下：

```text
gather_s1(DM)
    -> load_gate_up_block[0](DM) -> compute_gate_up_block[0](GEMM)
    -> load_gate_up_block[1](DM) -> compute_gate_up_block[1](GEMM)
    -> ...
    -> prefetch_s2_down(DM)
    -> load_down_block[0](DM) -> compute_down_block[0](GEMM)
    -> load_down_block[1](DM) -> compute_down_block[1](GEMM)
    -> ...
    -> prefetch_s4_next_s1(DM)
    -> store(DM)
```

实际 edge 比上面的线性写法更细：`load[n+1]` 只等 `load[n]`，不等 `compute[n]`；`compute[n]` 等 `load[n]` 和 `compute[n-1]`。因此在不同 core 上，DMA 可以把后续 block 提前搬入递增的 resident L1 地址，而 VersaCore 按 block 顺序消费已经就绪的权重。

```text
DMA may load block n+1 while VersaCore computes block n
```

这里没有使用 double buffering。L1_B_gate、L1_B_up 和 L1_B_down 的空间按完整 resident weight layout 分配；`load_gate_up_block[n]` 写入 `L1_B_gate/up + n * tile_bytes`，`load_down_block[n]` 写入 `L1_B_down + n * tile_bytes` 和右半区 `L1_B_down + (N2_down + n) * tile_bytes`。因此不需要跨两个 block 的防覆盖依赖。

当 `skip_dma_s1=1` 或 `skip_dma_s3=1` 时，相关 `load_*_block[n]` 节点立即成功返回；compute 节点仍按同一个 resident layout 读取第 n 个权重块。同一套 DFG 退化成纯计算路径。Phase4 不需要 runtime 选择另一张 DFG，也不需要 runtime 增删 edge。

C2 和 C3 不再有旧的 round barrier。每个 cluster 只依赖自己的前一个 store；跨 cluster 顺序由 runtime counter 表达。

### 5.3 Device execution

当前 block-level device kernels：

- `__snax_bingo_kernel_moe_dynamic_expert_gather_s1`
- `__snax_bingo_kernel_moe_dynamic_expert_load_gate_up_block`
- `__snax_bingo_kernel_moe_dynamic_expert_compute_gate_up_block`
- `__snax_bingo_kernel_moe_dynamic_expert_load_down_block`
- `__snax_bingo_kernel_moe_dynamic_expert_compute_down_block`
- `__snax_bingo_kernel_moe_dynamic_expert_prefetch_s2_down`
- `__snax_bingo_kernel_moe_dynamic_expert_prefetch_s4_next_s1`
- `__snax_bingo_kernel_moe_dynamic_expert_store`

device DFG 中实际是一对 load/compute 链。S1/S2/S3/S4 只保留在 scheduler timing 和 DMA slot role 中，不再作为 block compute kernel 的名字：

| 分析/slot 语义 | DMA slot | Static DFG node | 作用 |
| --- | --- | --- | --- |
| S1 foreground gate/up DMA | Slot 0 | `load_gate_up_block[n]` | 只做第 n 个 gate/up weight block 搬运 |
| S1/S2 gate/up compute timing | 无 compute slot | `compute_gate_up_block[n]` | 对第 n 个 N-block 执行 gate/up/SwiGLU compute |
| S2 down prefetch | Slot 2 | `prefetch_s2_down` | 在 gate/up compute 之后、down compute 前预取当前 expert 的 down weight |
| S3 foreground down DMA | Slot 1 | `load_down_block[n]` | 只做第 n 个 down weight block 搬运 |
| S3/S4 down compute timing | 无 compute slot | `compute_down_block[n]` | 对第 n 个 N-block 执行 down projection compute |
| S4 next-S1 prefetch | Slot 3 | `prefetch_s4_next_s1` | 预取后续 expert 的 gate/up weight |

因此不能问“`load_s1` 里面为什么没有 compute”：现在的准确表达是 Slot 0 只描述 gate/up foreground DMA；gate/up 的 VersaCore 计算由 `compute_gate_up_block[n]` 执行。换句话说，S1 分析窗口有 compute timing，但物理 DFG 没有一个叫“S1 compute”的独立 kernel。

如果问“S1 到底是不是边算边搬”，准确答案是：在 **分析 stage** 层面，S1 是 gate/up 初始 token-window 的 timing；在 **物理 block DFG** 层面，gate/up 确实支持边搬边算，即 DM 可以搬 `gate_up block n+1`，同时 VersaCore 计算 `gate_up block n`。这不是单个 S1 kernel 同时 DMA 和 compute，而是两个资源上的两个静态 node 通过依赖边实现 overlap。

这些 foreground block kernels 消费 DMA slot metadata：

- `gather_s1` 在启动前等待 peer task counter，然后串行 gather 当前 task 的 token 到 L1_A。
- `load_gate_up_block[0]` 等待 S1 DMA lane sequence；每个 block 写入递增的 resident L1_B_gate/up 地址，最后一个 gate/up block 完成后递增 lane counter。
- `compute_gate_up_block[n]` 读取 resident L1_B_gate/up layout 中的第 n 块，并用 `dual_vc_swiglu_full` 同时完成 gate/up/SwiGLU 计算。
- `prefetch_s2_down` 消费 slot 2，把当前 expert 的 down weight 提前搬入 `l1_b_down`。
- `load_down_block[0]` 等待 S3 DMA lane sequence；每个 block 写入递增的 resident L1_B_down 地址，最后一个 down block 完成后递增 lane counter。
- `compute_down_block[n]` 读取 resident L1_B_down layout 中的第 n 块和右半区第 n 块，完成 down projection 计算。
- `prefetch_s4_next_s1` 消费 slot 3，把下一 expert 的 gate/up weight 搬入 `l1_b_gate/l1_b_up`。
- `store` 成功写回后更新本 cluster completed counter。

## 6. 硬件推荐消费方式

为了避免硬件动态 join `tasks[]` 和 `dma_ops[]`，建议硬件只读取 `tasks[]` 中的固定字段：

```c
task = tasks[i]
slot_s1  = task.dma_slots[MOE_TASK_DMA_SLOT_S1]
slot_s3  = task.dma_slots[MOE_TASK_DMA_SLOT_S3]
slot_s2p = task.dma_slots[MOE_TASK_DMA_SLOT_S2_PREFETCH]
slot_s4p = task.dma_slots[MOE_TASK_DMA_SLOT_S4_PREFETCH]
```

硬件执行策略：

1. 按 `tasks[]` 顺序接收 task descriptor。
2. 根据 `task.cluster` 将 task 下发到 C2 或 C3 的执行队列。
3. 对每个 task，固定检查 4 个 slot 的 `valid` bit。
4. `valid=0` 的 slot 直接忽略。
5. `valid=1` 的 slot 使用已经给定的 `dma` lane、`alloc_bw`、`expert_id`、`weight`、`shape` 发起 DMA。
6. 硬件不重新计算 cache 命中、不重新选择 shape、不重新选择 DMA lane。

## 7. 时间字段的使用建议

`start_cc` 和 `end_cc` 是分析调度器给出的绝对周期估计。对硬件有两种使用方式：

### 方式 A：仅作为 debug/profiling 字段

硬件按 task 内固定 stage 顺序执行，不严格等待 `start_cc`。这是最简单的硬件实现。

适用条件：

- 软件已经按照 task 顺序和 cluster 分配保证依赖关系。
- DMA/compute manager 能自然按队列顺序 backpressure。

### 方式 B：作为 earliest issue hint

硬件在 `current_cycle >= start_cc` 后才允许发起对应 DMA slot。这能更接近分析模型，但硬件需要 cycle compare。

当前 workload 采用介于 A/B 之间的实现：不做 cycle compare，但利用 `start_cc/end_cc` 推导跨 cluster 前置计数，并利用 `dma_ops[]` 的排序结果生成 iDMA/xDMA lane sequence。因此硬件不需要周期比较器，却能保留 Phase3 给出的**部分序**和 DMA lane 顺序。

## 8. `dma_ops[]` 的定位

`dma_ops[]` 仍然保留，但不建议硬件直接消费。它的用途是：

- 软件侧验证所有 DMA 操作是否被显式生成。
- 调试 prefetch 是否真实落表。
- 生成 trace 或可视化。
- 将来如果要做 command stream lowering，可以从 `dma_ops[]` 生成。

硬件主接口仍然是 `tasks[].dma_slots[]`。当前实现中，软件内部仍保留 `tasks[] + dma_ops[]` 两张表：`tasks[]` 表示 scheduled expert run 的全局顺序和 cluster 分配，`dma_ops[]` 表示调度器搜索出来的 DMA 时间窗口。它们在调度器内部可以分开，因为这样便于分析、lane binding 和 trace；但它们只在 host lowering 阶段被合并。Bingo/device 端看到的是每个 dynamic slot arg 中已经展开好的 fixed DMA slot 字段，而不是运行时再 join 两张表。

## 9. Cache 和 Prefetch 语义

必须区分两种 cache hit：

1. **初始 full cache**
   - 来自上一轮计算流遗留。
   - S1 gate/up 和 S3 down 都 ready。
   - 可以同时跳过 `skip_dma_s1=1` 和 `skip_dma_s3=1`。

2. **运行时 S4 next-S1 prefetch**
   - 只搬 gate/up。
   - 后续 task 只能跳过 S1 foreground DMA。
   - 不能跳过 S3 foreground DMA，除非另有 down-prefetch 或 full cache。

S2 down-prefetch 则是给当前 expert 的 down 权重提前搬运。如果成功，当前 task 的 S3 foreground DMA 可以被跳过。

当前实现对 `cam_state` 的更新规则也按这个语义处理：如果某 cluster 的最后一个 task 执行了 S4 next-S1 prefetch，则该 cluster 退出时不是 full cache，`cam_state` 置为 `-1`；否则记录最后完成 task 的 expert id。

## 10. PPA 与实现权衡

- **Power**：runtime counters 使用 L3 volatile polling。它比 RTL FSM 更耗能，但当前只覆盖 Phase4 的少量 slot，适合先验证调度语义。
- **Performance**：去掉全局 round barrier 后，pair/split 和 early-start 可以按 Phase3 的 cluster-local 顺序执行；S2/S4 prefetch 也有独立 DM-core node，可与 GEMM stage 并行。
- **Area**：没有新增 RTL 面积；代价是 dynamic arg 从 256 B 扩大到 512 B，并新增 16 B L3 runtime state。
- **时序风险**：未改动 RTL critical path。device 端新增 polling loop 和 fence，只影响运行时行为。

当前实现是软件控制的硬件友好版本。若后续需要更低功耗，可以把 runtime counters 迁移到 bingo manager 或 xDMA 附近的小型 RTL sequencer。

## 11. 后续硬件化建议

如果后续要把当前软件 runtime counter 方案迁移到 RTL，建议保持以下边界：

1. `moe_schedule()` 仍作为 Phase3 搜索算法入口，RTL 不实现全局搜索。
2. Host 或硬件 sequencer 只接收已经展开好的 task descriptor 和 fixed DMA slots。
3. 每个 cluster 保留本地 task FIFO，以表达同 cluster 顺序。
4. 增加两个跨 cluster completed counters，以表达 `wait_for_peer_slots`。
5. 增加 iDMA/xDMA 两条 lane sequence counters，以表达 `idma_seq/xdma_seq`。
6. 不让 RTL 动态遍历 `dma_ops[]` 匹配 task；`dma_ops[]` 只用于 host lowering、trace 和验证。
7. 若要减少 polling 功耗，可把当前 L3 counters 替换成 bingo manager 内的寄存器或小 FIFO 状态机。

## 12. 轻量化策略现状

当前已经做了一个无损轻量化：

- `moe_schedule()` 不再固定评估 4 个 initial-cache mask。
- 只有 cache expert 确实出现在当前 request 中时，才评估对应 mask。
- 无有效 cache 时只跑一次搜索。

服务器 x86 `-O2` 上，100 次复杂混合 benchmark 从约 `78.5 ms/call` 降到约 `20.9 ms/call`。

进一步轻量化建议使用可配置 fast mode，而不是直接删除 full mode：

- full mode 用于离线评估和论文结果。
- fast mode 用于 CVA6 在线调度。

可剪枝项：

- 限制 `PAIR(topK, topJ)` 搜索范围。
- 可选关闭 `WAIT-SINGLE-PAIR`。
- 可选关闭 `split_hot_tail_cost`。
- 减少 split cut set。

这些剪枝会带来一定调度质量损失，应单独用 beam/eval 脚本评估。

## 13. 当前验证状态

此前已做过轻量 C 验证：

- workload `moe_scheduler.c` 可用 `gcc -std=c99 -O2 -Wall -Wextra -c` 编译通过。
- 关键分布 makespan 保持不变：
  - `[17,3,1,1,1,1] -> 304128`
  - `[10,1,1]` with C2 cache 0 -> `135168`
  - `[9,2,1]` with C2 cache 0 -> `135168`
  - `[9,1,1,1] -> 202752`

本轮 resident block layout 和 block-kernel 语义重命名后，已重新生成 `offload_bingo_hw.h`、`final_dfg.csv` 和 DFG 图；host workload、`snax-bingo-offload` device app、以及最终嵌入 device bin 的 host finalize build 均已编译通过。尚未跑 VSIM；跑 VSIM 时必须按服务器 license 保护流程在结束后终止仿真并退出 QuestaSim。
