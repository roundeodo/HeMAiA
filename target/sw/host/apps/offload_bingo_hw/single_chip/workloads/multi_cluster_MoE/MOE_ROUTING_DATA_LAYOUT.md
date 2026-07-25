# Multi-cluster MoE routing and data-layout handoff

本文档说明 `multi_cluster_MoE` 从 Router 输出、TopK、硬件调度、token
gather、individual/shared expert 输出，到未来 delayed softmax 和最终加权归约的
数据结构与地址关系。

本文档以当前 checked-in 模型作为例子，但地址公式本身是参数化的：

| 符号 | 当前值 | 含义 |
|---|---:|---|
| `M` | 8 | 输入 token 数，即 `M_total` |
| `E` | 8 | individual expert 数 |
| `K` | 2 | 每个 token 选择的 expert 数 |
| `H` | 2048 | 输入和 down projection 输出宽度 |
| `I` | 1024 | SwiGLU 中间宽度 |
| `MAX_TOK` | 8 | 每个 expert 固定预留的 token 数，当前等于 `M` |
| `TOKEN_BYTES` | 4096 | 一个 INT16 token/output row 的字节数，`H * 2` |

当前 Router output 是 `M * E * sizeof(int16_t) = 128 bytes`。shared legacy
row stride 是 `4096 + 32 = 4128 bytes`，所以每个 shared expert span 是
`8 * 4128 = 33024 bytes`，两个 shared experts 共 66048 bytes。

参数入口是 `params.hjson`，派生逻辑在 `moe_layout.py`。Bingo 的
`BingoMemAlloc` 在生成/运行时确定实际 L3 基地址，因此源码里没有稳定的绝对地址；
本文使用 `*_BASE` 表示对应 allocation 的起始地址。

## 1. 连续性的三个不同含义

阅读下文时必须区分：

1. **分配连续**：整张表位于一段连续地址中。
2. **有效数据紧凑**：有效项之间没有预留空洞。
3. **可一次 DMA**：源和目标都能被一个 1D/2D descriptor 表达。

一张表可以连续分配，但访问仍然离散。例如 `input_A[M][H]` 全局连续，某个
expert 选择的 token ID 却通常不连续，因此 expert gather 仍是离散 gather。

## 2. 端到端数据流

```text
Router tiled INT16 scores
          |
          v
        TopK (token first, then scan all experts)
          |
          +--> TopK_Scores[M][K] ---------> delayed softmax
          |       token-major                   |
          |                                     v
          |                               Probability[M][K]
          |                                     |
          +--> Expert_Counts[E]                 |
          |                                     |
          +--> Expert_Token_Refs[E][MAX_TOK]    |
                    expert-major                |
                    |                           |
                    +--> RTL scheduler          |
                    +--> token gather           |
                              |                 |
                              v                 |
                    individual expert compute   |
                              |                 |
                              v                 |
              Individual_Output[E][MAX_TOK][H] |
                              |                 |
                              +--------+--------+
                                       v
                    weighted scatter-add by token_id/rank
                                       |
                Shared0[M][H] ---------+
                Shared1[M][H] ---------+
                                       v
                              Final_Output[M][H]
```

当前生产 workload 已实现到 individual/shared 输出写回。`Probability`、delayed
softmax、`Final_Output` 和基于新 token-ref ABI 的 weighted accumulation 尚未加入
当前 DFG；第 12 节单独记录这一边界。

## 3. Router 原始输出

### 3.1 逻辑含义

逻辑上 Router 产生：

```text
router_score[token_id][expert_id]
```

每项来自 VersaCore INT16 输出，TopK 时 sign-extend 为 INT32。

### 3.2 物理布局

Router 输出不是普通的 `[M][E]` row-major 矩阵，而是 VersaCore tiled 布局：

```text
[M2 tile][N2 tile][M1][N1][mesh row][mesh col]
```

生产 HW path 通过 `tile/n2/n1/col` 恢复某个 token 的全部 expert score。
相关实现位于 `target/sw/host/runtime/host_kernel_lib.h` 的
`__host_bingo_kernel_moe_router_schedule()`。

### 3.3 TopK 遍历顺序

```text
for each Router M tile
    for each valid token in the tile
        for each N2/N1/mesh-column expert
            update best0 and best1
```

因此外层是 token 顺序，内层扫描所有 expert。score 相同时选择较小的
`expert_id`。

## 4. Token-major TopK 表

### 4.1 `L3_Alloc_TopK_Scores`

类型和维度：

```c
int32_t topk_scores[M][K];
```

地址：

```text
score_addr(t, rank) = TOPK_SCORES_BASE + 4 * (t * K + rank)
```

当前布局和大小：

```text
[T0.R0][T0.R1][T1.R0][T1.R1] ... [T7.R0][T7.R1]

8 * 2 * 4 = 64 bytes
```

属性：

- 连续分配。
- 有效数据完全紧凑。
- 可顺序读取并计算 softmax。
- 这是生产数据，不是 debug-only 数据。

### 4.2 `L3_Alloc_TopK_Indices`

类型和维度：

```c
uint16_t topk_indices[M][K];
```

地址：

```text
index_addr(t, rank) = TOPK_INDICES_BASE + 2 * (t * K + rank)
```

当前大小：

```text
8 * 2 * 2 = 32 bytes
```

重要状态：pure-HW scheduler 路径只有在 `MOE_SCHED_DEBUG_PRINT` 打开时才写这张
表。生产 gather、scheduler 和未来 weighted accumulation 都不应依赖它。

## 5. Expert-major 路由表

### 5.1 `L3_Alloc_Expert_Counts`

类型和维度：

```c
uint32_t expert_token_counts[E];
```

地址：

```text
count_addr(e) = EXPERT_COUNTS_BASE + 4 * e
```

当前大小为 `8 * 4 = 32 bytes`。所有 count 的总和应为：

```text
sum(expert_token_counts) = M * K
```

它由 TopK 生成，由 RTL/SW scheduler 读取，并定义每个 expert token-ref 区域的
有效前缀长度。

### 5.2 `L3_Alloc_Expert_Token_Refs`

类型和逻辑维度：

```c
uint16_t expert_token_refs[E][MAX_TOK];
```

地址：

```text
ref_index(e, j) = e * MAX_TOK + j
ref_addr(e, j)  = EXPERT_REFS_BASE + 2 * ref_index(e, j)
```

当前大小为：

```text
8 experts * 8 reserved entries * 2 bytes = 128 bytes
```

每个 `uint16_t` 的打包方式：

```text
bit 15      = TopK rank (0=rank0, 1=rank1)
bits 14:0   = original token_id
```

定义位于：

```text
target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h
```

宏为 `BINGO_MOE_TOKEN_REF_PACK/TOKEN/RANK`。

每个 expert 只有以下前缀有效：

```text
expert_token_refs[e][0 .. expert_token_counts[e]-1]
```

其余预留位置未定义，不要求清零，也不得读取。

### 5.3 顺序保证

TopK 按 `T0,T1,...,T(M-1)` 遍历，并用 `local_counts[e]++` 向 expert 区域追加。
因此每个 expert 内部，解包后的 `token_id` 严格递增：

```text
E3: [T0/R0][T1/R1][T2/R0][T4/R1][T7/R0]
```

完整 packed `uint16_t` 数值不保证递增，因为 rank1 设置 bit 15。

### 5.4 示例

假设路由结果为：

```text
T0 -> E3/R0, E1/R1
T1 -> E0/R0, E3/R1
T2 -> E3/R0, E5/R1
T3 -> E1/R0, E0/R1
T4 -> E5/R0, E3/R1
T5 -> E0/R0, E1/R1
T6 -> E7/R0, E5/R1
T7 -> E3/R0, E0/R1
```

则：

```text
Counts: E0=4 E1=3 E2=0 E3=5 E4=0 E5=3 E6=0 E7=1

E0: [T1/R0][T3/R1][T5/R0][T7/R1][-][-][-][-]
E1: [T0/R1][T3/R0][T5/R1][-][-][-][-][-]
E2: [-][-][-][-][-][-][-][-]
E3: [T0/R0][T1/R1][T2/R0][T4/R1][T7/R0][-][-][-]
E4: [-][-][-][-][-][-][-][-]
E5: [T2/R1][T4/R0][T6/R1][-][-][-][-][-]
E6: [-][-][-][-][-][-][-][-]
E7: [T6/R0][-][-][-][-][-][-][-]
```

## 6. Scheduler task range and split semantics

RTL task descriptor/task word 至少携带：

```text
expert_id
tok_start       -> CVA6 lowering 后为 token_ref_start
ntokens
cluster
shape_s1/shape_s3
skip_s1/skip_s3
has_s2pf
```

当一个 expert 在 C2/C3 间拆分时，scheduler 生成两个连续且不重叠的范围：

```text
C2: token_ref_start = 0,   ntokens = cut
C3: token_ref_start = cut, ntokens = total - cut
```

实现见 `Scheduler_hw/sched_candidate_evaluator.sv` 中的 split candidate，以及
`Scheduler_hw/sched_task_word_pack.sv` 的 `tok_start` 打包。

因此 output 的位置由 `token_ref_start` 决定，不由任务完成顺序决定。C3 即使先完成，
也只会写自己的 `[cut,total)` 区间。

## 7. Input A 和 token gather

### 7.1 L3 输入布局

Individual 输入是 dense token-major：

```c
int16_t input_A[M][H];
```

地址：

```text
input_addr(t) = INPUT_A_BASE + t * TOKEN_BYTES
```

当前：

```text
INPUT_A_BASE
  + 0 * 4096 -> T0
  + 1 * 4096 -> T1
  ...
  + 7 * 4096 -> T7
```

整张输入连续，但某个 expert 选择的 token ID 通常不连续。例如 E3 需要
`T0,T1,T2,T4,T7`，所以不能普遍合并成一次连续 1D copy。

### 7.2 Gather 访问

设备端 task 使用：

```text
start = expert_id * MAX_TOK + token_ref_start
ref   = expert_token_refs[start + local_t]
t     = TOKEN_REF_TOKEN(ref)
src   = INPUT_A_BASE + t * TOKEN_BYTES
```

rank bit 在 gather 阶段被屏蔽，因为 gather 只需要原始 token 地址。

当前每个 token 发一个规则 2D descriptor，将 dense L3 row 映射到 conflict-free
TCDM bank pair：

```text
size       = 16 bytes
src_stride = 16 bytes
dst_stride = 512 bytes
repeat     = H / 8 = 256
```

因此：

- 多 token 之间是不规则 gather。
- 单个 token 内是规则 2D transfer。
- gather 后 local token 编号为 `0..ntokens-1`，expert 计算不再关心原始 token ID。

## 8. L1 token/weight/output bank layout

每个 individual cluster 有一个 bank-partition arena：

```text
banks  0..15 : input A / Mode1 output
banks 16..47 : gate/up/down weight ping-pong regions
banks 48..63 : Mode0 output
```

八个 local token 使用 banks `2*lane .. 2*lane+1`，其中 `lane=local_t%8`；超过
八个 token 时使用 `page=local_t/8` 进入更深的 TCDM row，不覆盖前一页。

weights 使用 64-byte 2D rows：

```text
size       = 64 bytes
src_stride = 64 bytes
dst_stride = 512 bytes
```

这些是规则 strided layout，可由 2D DMA/streamer 表达，不属于 token-ID
不规则 gather。

## 9. Individual output

### 9.1 L3 布局

逻辑维度：

```c
int16_t individual_output[E][MAX_TOK][H];
```

地址：

```text
OUTPUT_EXPERT_STRIDE = MAX_TOK * TOKEN_BYTES

output_addr(e, j)
    = INDIV_OUTPUT_BASE
    + e * OUTPUT_EXPERT_STRIDE
    + j * TOKEN_BYTES
```

当前：

```text
OUTPUT_EXPERT_STRIDE = 8 * 4096 = 32768 bytes
total allocation     = 8 * 32768 = 262144 bytes = 256 KiB
```

### 9.2 refs 与 output 的位置不变量

```text
expert_token_refs[e][j]  <---->  individual_output[e][j]
```

例如：

```text
refs[E3][2]   = T2/R0
output[E3][2] = E3 对 T2 的 H 元素 down-projection 输出
```

task 写回地址为：

```text
dst = INDIV_OUTPUT_BASE
    + expert_id * OUTPUT_EXPERT_STRIDE
    + token_ref_start * TOKEN_BYTES
```

所以同一 task 的 `[token_ref_start, token_ref_start+ntokens)` 是连续 L3 目标范围。
L1 源是 banked layout，因此实际 store 使用规则 2D xDMA；语义上的 output row
顺序仍与 refs 完全一致。

### 9.3 有效区域

每个 expert 只有：

```text
individual_output[e][0 .. expert_token_counts[e]-1]
```

有效。expert 区域之间因为固定 `MAX_TOK` stride 可能存在未使用洞。

## 10. Shared output

当前固定有两个 shared experts，分别在 C0/C1 执行。输出分别写入：

```text
SHARED_OUTPUT_BASE + 0 * SHARED_EXPERT_SPAN
SHARED_OUTPUT_BASE + 1 * SHARED_EXPERT_SPAN
```

shared 仍使用 legacy L1.5 layout：每个 token row 包含 `H*2` 字节 payload 和
32-byte compatibility gap。当前 host iDMA 会复制整个 padded span。

未来归约不能把 gap 当作有效数据；应按 shared row stride 读取每个 token 的
`H` 个 payload 元素。两个 shared expert 的结果都按原始 token 顺序排列，不需要
token-ref 表。

## 11. Delayed softmax 所需结构

### 11.1 输入

softmax 直接读取已有的 token-major：

```c
int32_t topk_scores[M][K];
```

它不应读取 expert-major refs，也不需要生产路径中的 TopK indices。

### 11.2 输出

建议新增：

```c
uint32_t probability[M][K];
```

地址：

```text
prob_addr(t, rank) = PROBABILITY_BASE + 4 * (t * K + rank)
```

当前需要 `8 * 2 * 4 = 64 bytes`。它必须保持与 `topk_scores` 完全相同的
`[token][rank]` 顺序。

旧实现使用定点概率，并在归约时右移 16 位；若沿用该 ABI，应明确
`probability` 为 Q16，满足：

```text
sum_rank probability[t][rank] ~= 65536
```

softmax 只依赖 TopK，可与 scheduler、DMA 和 expert compute 并行，最终归约再等待
它完成。

## 12. 最终 weighted accumulation

### 12.1 建议输出

最终结果应恢复为紧凑 token-major：

```c
final_output[M][H];
```

地址：

```text
final_addr(t, d) = FINAL_BASE + (t * H + d) * sizeof(final_element)
```

若为 INT32，当前大小为：

```text
8 * 2048 * 4 = 65536 bytes = 64 KiB
```

若归约后立即 requantize 为 INT16，则为 32 KiB。正式实现前必须明确输出类型和
shared/individual 的量化尺度。

### 12.2 建议算法

先初始化 shared contribution，再扫描 expert-major 有效区域：

```c
for (t = 0; t < M; t++) {
    for (d = 0; d < H; d++) {
        final[t][d] = shared0[t][d] + shared1[t][d];
    }
}

for (e = 0; e < E; e++) {
    for (j = 0; j < expert_token_counts[e]; j++) {
        uint16_t ref = expert_token_refs[e * MAX_TOK + j];
        uint32_t t = BINGO_MOE_TOKEN_REF_TOKEN(ref);
        uint32_t rank = BINGO_MOE_TOKEN_REF_RANK(ref);
        uint32_t p = probability[t * K + rank];

        for (d = 0; d < H; d++) {
            final[t][d] +=
                ((int64_t)individual_output[e][j][d] * p) >> 16;
        }
    }
}
```

访问性质：

- `expert_token_refs[e][j]`：每个 expert 内连续读取。
- `individual_output[e][j][:]`：连续读取完整 H 元素 row。
- `probability[t][rank]`：标量查表；token ID 在 expert 内递增，但可跳号。
- `final[t][:]`：每次写一个连续 H 元素 row；不同 j 之间可能跳过 token row。
- 最终 `final_output` 本身是无洞、连续的 token-major 张量。

如果未来让多个 cluster 并行执行归约，两个 expert contribution 可能同时更新同一个
`final[t]`，不能直接无同步 scatter-add。可选方案是单 host core 顺序归约，或先写
rank0/rank1 独立 partial buffers，再做一次连续 final reduction。

### 12.3 为什么不再需要旧 reverse-index 表

旧 single-cluster workload 使用：

```text
reverse_original_token_flat_idx[position] = token_id * K + rank
```

当前 packed token ref 已直接保存同样的信息，并且与 fixed-stride output 的
`[e][j]` 位置对齐，因此无需第二次 scatter-metadata pass，也无需单独 k-position
表。

旧 `experts_result_accumulate()` 假设 individual output 是全局 compact row stream，
不能直接用于当前 `[E][MAX_TOK][H]` fixed-stride layout。正式 workload 需要一个
基于 `counts + refs + output_expert_stride` 的新归约实现。

## 13. 调度与执行控制结构

### 13.1 CAM state

```c
int32_t cam_state[2];
```

```text
cam_state[0] = C2 当前 resident expert
cam_state[1] = C3 当前 resident expert
-1           = 无 resident expert（初始化语义）
```

当前 allocation 大小为 8 bytes。软件维护该状态，scheduler 根据它决定 S1/S3
是否 cache hit、是否需要 load/prefetch。

### 13.2 Runtime state/header

`L3_Alloc_MoE_Runtime_State` 为 64 bytes。C2/C3 stage 和 L1 dynamic block 的前
64 bytes 都是 runtime header，其中包括 C2/C3 active slot count。`MoEExecute`
更新 header 并只复制 active slot 范围。

### 13.3 C2/C3 stage buffers

每个 cluster 的 L3 stage：

```text
[64-byte runtime header]
[slot0: 384 bytes]
[slot1: 384 bytes]
...
[slot(E-1): 384 bytes]
```

每个 dynamic task record 实际结构为 344 bytes，ABI slot 固定为 384 bytes。结构
`__snax_bingo_kernel_moe_dynamic_expert_args_t` 包括：

- `ctrl + expert_id`
- `token_ref_start + ntokens`
- `m_s2_exec + m_s4_exec`
- 四个 DMA slot 的 valid/binding/target expert
- 最多八个 S1 call records
- 一个 S2 call record
- 最多八个 S3 call records
- 一个 S4 call record

定义位于 `target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h`。

### 13.4 Per-cluster static context

C2/C3 各有一个 192-byte static slot，保存所有 task 共享的：

- L3 token/weight/output base
- L1 A/B/D addresses
- token/output strides
- S1/S3 block counts
- model dimensions
- DMA/streamer 派生常数

因此 dynamic slot 只保存 runtime 选择，不重复保存每个 task 都相同的模型地址。

### 13.5 L1 token-ref replicas

`MoEExecute` 将完整 `E * MAX_TOK * 2` token-ref 表分别复制到 C2/C3 L1。设备 DM
core 的 gather 因此只做本地标量 ref 读取，不会在提交每个 DMA descriptor 时访问
L3 metadata。

### 13.6 Pipeline control

每个 dynamic slot 预留 1024 bytes pipeline control，热字段从 slot 内 byte 448
开始，以便映射到指定 TCDM banks。它保存跨 stage 的 preload/ready/pending 状态，
不是模型数据，也不影响 token/output 的逻辑顺序。

## 14. L3 allocation inventory

| Allocation | 类型/逻辑维度 | 当前大小 | 状态 |
|---|---|---:|---|
| `L3_Alloc_Router_Output` | tiled INT16 Router output | 由 Router tile 参数派生 | 已实现 |
| `L3_Alloc_TopK_Indices` | `uint16_t[M][K]` | 32 B | HW path debug-only |
| `L3_Alloc_TopK_Scores` | `int32_t[M][K]` | 64 B | 已实现、生产数据 |
| `L3_Alloc_Expert_Counts` | `uint32_t[E]` | 32 B | 已实现 |
| `L3_Alloc_Expert_Token_Refs` | `uint16_t[E][MAX_TOK]` | 128 B | 已实现 |
| `L3_Alloc_CAM_State` | `int32_t[2]` | 8 B | 已实现 |
| `L3_Alloc_MoE_Runtime_State` | runtime header/state | 64 B | 已实现 |
| `L3_Alloc_C2_Stage` | header + `E` dynamic slots | `64 + E*384` | 已实现 |
| `L3_Alloc_C3_Stage` | header + `E` dynamic slots | `64 + E*384` | 已实现 |
| `L3_Alloc_MoE_Request` | SW scheduler request | 256 B | SW build only |
| `L3_Alloc_MoE_Schedule` | SW scheduler schedule | 32768 B | SW build only |
| `L3_Alloc_Indiv_Down_Output` | `int16_t[E][MAX_TOK][H]` | 256 KiB | 已实现 |
| `L3_Alloc_Shared_Down_Output` | two padded shared spans | 参数派生 | 已实现 |
| proposed `Probability` | `uint32_t[M][K]` | 64 B | 尚未分配/接线 |
| proposed `Final_Output` | `int32_t[M][H]` | 64 KiB | 尚未分配/接线 |

当前 `E=8` 时，每个 C2/C3 stage 为 `64 + 8*384 = 3136 bytes`。C2/C3 L1
各持有一份 128-byte token-ref replica；每个 cluster 另有一个 192-byte static
context 和 `8 * 1024 = 8192 bytes` pipeline-control 区域。

## 15. 连续与离散操作总结

### 完全连续

- `TopK_Scores[M][K]`
- future `Probability[M][K]`
- `Expert_Counts[E]`
- 全局 `Input_A[M][H]` allocation
- future `Final_Output[M][H]`

### 固定分区连续，但存在无效预留洞

- `Expert_Token_Refs[E][MAX_TOK]`
- `Individual_Output[E][MAX_TOK][H]`

每个 expert 的有效前缀连续，expert 之间按照固定最大容量跳转。

### 必须处理不规则 token 映射

- 从全局 Input A 为某个 expert gather token。
- 将 expert-major output 按 token ref 加权并 scatter-add 回 final token。

### 规则 2D，不属于不规则 gather

- 单个 dense token row 到 conflict-free L1 bank pair。
- weight 到 ping/pong bank region。
- banked L1 expert output 写回连续 L3 expert range。

## 16. 当前实现边界和交接事项

当前 `main_bingo.py` 的生产 DFG 已完成：

```text
Router -> TopK/counts/refs -> scheduler/lowering -> gather
       -> individual S1/S2/S3/S4 -> individual store
       -> shared output store
```

当前尚未完成：

1. 分配并生成 `Probability[M][K]`。
2. 在 TopK 后加入 delayed-softmax 节点，并允许它与 expert execution overlap。
3. 明确 shared/individual output 的量化尺度和 final element type。
4. 分配 `Final_Output[M][H]`。
5. 实现基于 `counts + packed refs + fixed expert stride` 的 weighted accumulation。
6. 合并 C0/C1 两个 shared expert 输出。
7. 为 final accumulation 建立正确 DFG 依赖：等待 probability、全部 individual
   stores 和两个 shared stores。

不得直接复用旧 single-cluster `experts_result_accumulate()`：旧函数依赖 compact
individual rows 和 `reverse_original_token_flat_idx`，与当前 fixed-stride expert-major
输出不兼容。

## 17. 关键源码索引

- 参数和维度：`multi_cluster_MoE/params.hjson`
- 地址/stride 派生：`multi_cluster_MoE/moe_layout.py`
- L3/L1 allocations 与 DFG：`multi_cluster_MoE/main_bingo.py`
- TopK/counts/refs 构建：`target/sw/host/runtime/host_kernel_lib.h`
- packed token-ref ABI：`target/sw/host/runtime/libbingo/include/libbingo/device_kernel_args.h`
- HW scheduler task lowering：`target/sw/host/runtime/host_moe_hw_path.h`
- token gather：`offload_hw_kernels/moe_dynamic_stage_s1.h`
- individual output store：`offload_hw_kernels/moe_dynamic_stage_store.h`
- scheduler split：`Scheduler_hw/sched_candidate_evaluator.sv`
- task-word layout：`Scheduler_hw/sched_pkg.sv`、`Scheduler_hw/sched_task_word_pack.sv`
