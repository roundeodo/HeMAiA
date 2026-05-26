// ============================================================================
// multi_cluster_MoE 的公共静态参数
// ============================================================================
// 这个文件主要给两类代码读：
//   1. Python 侧生成器：main_bingo.py / multi_cluster_MoE_datagen.py
//   2. Host 侧一些后处理/工具函数
//
// 它描述的是“这个 workload 想算什么尺寸”，不是“Phase 4 调度算法怎么写”。
// 调度算法接口和执行逻辑是另外一层。
//
// 当前硬件是 snax_dual_versacore_int16x4_rebalanced：
//   Mode 0: gate/up 双路输入做 SwiGLU
//   Mode 1: 普通 GEMM，用在 router 和 down projection
// ============================================================================

// 矩阵维度（中等尺寸测试，array_shape=0 S0: meshRow=8, tileSize=8, meshCol=4）
#define input_dimension          1024  // K_total = K1×tileSize = 128×8 = 1024
#define expert_hidden_dimension  512   // N_hidden = N2×N1×meshCol = 2×64×4 = 512
#define router_output_dimension  8     // N_router = N1_r×meshCol = 2×4 = 8
#define output_dimension         1024  // N_out (down proj) = N2_down×N1_down×down_meshCol = 2×64×8 = 1024

// 专家配置
#define expert_number_each_layer 8     // 总 individual expert 数
#define individual_expert_number_k 2   // 每个 token 选择的 expert 数 (top-k)
#define shared_expert_number_k   2     // shared expert 数（C0=expert0, C1=expert1）

// Token 配置
#define max_input_tokens         32    // 输入 token 数（= M2 × M1 × meshRow = 4 × 1 × 8 ）

// 调度相关常量
#define num_indiv_clusters       2     // 当前只有两个 indiv 执行位：C2 + C3
#define EXPERT_INVALID           0xFF  // 哨兵值：该 slot 无有效 expert

// 量化参数
#define final_shift_step         24
#define swish_glu_scale_in       0.000015258789f
#define swish_glu_scale_out      32.0f
#define softmax_scale            65536.0f
