# 用 CUTLASS 实现 Flash Attention 2: 从原理到代码的完整指南

> 本文以导师的视角，带你从零开始理解并用 CUTLASS 2.x (C++) 框架实现 Flash Attention 2 的 forward pass。
> 目标架构: NVIDIA Ampere (SM80)，数据类型: FP16/BF16。

---

## 目录

- [第零章：前置知识检查清单](#第零章前置知识检查清单)
- [第一章：Flash Attention 2 算法原理](#第一章flash-attention-2-算法原理)
  - [1.1 标准 Attention 的问题](#11-标准-attention-的问题)
  - [1.2 Tiling + Online Softmax = Flash Attention](#12-tiling--online-softmax--flash-attention)
  - [1.3 Flash Attention 2 的改进](#13-flash-attention-2-的改进)
  - [1.4 伪代码](#14-伪代码)
- [第二章：CUTLASS 架构速览](#第二章cutlass-架构速览)
  - [2.1 五层抽象](#21-五层抽象)
  - [2.2 关键模板参数](#22-关键模板参数)
  - [2.3 共享内存管理](#23-共享内存管理)
  - [2.4 Epilogue 系统](#24-epilogue-系统)
- [第三章：整体架构设计](#第三章整体架构设计)
  - [3.1 为什么不能直接用一个 GEMM？](#31-为什么不能直接用一个-gemm)
  - [3.2 双 GEMM 融合架构](#32-双-gemm-融合架构)
  - [3.3 数据流全景图](#33-数据流全景图)
  - [3.4 共享内存布局设计](#34-共享内存布局设计)
- [第四章：实现步骤 — 分阶段动手](#第四章实现步骤--分阶段动手)
  - [Phase 0: 搭建项目骨架](#phase-0-搭建项目骨架)
  - [Phase 1: 实现 GEMM0 (Q @ K^T)](#phase-1-实现-gemm0-q--kt)
  - [Phase 2: 实现 Online Softmax（寄存器级别）](#phase-2-实现-online-softmax寄存器级别)
  - [Phase 3: Accumulator → SMEM 传递](#phase-3-accumulator--smem-传递)
  - [Phase 4: 实现 GEMM1 (Attn @ V) — 从 SMEM 读 A](#phase-4-实现-gemm1-attn--v--从-smem-读-a)
  - [Phase 5: 自定义 Epilogue — Rescale 输出](#phase-5-自定义-epilogue--rescale-输出)
  - [Phase 6: 组装 Kernel 主循环](#phase-6-组装-kernel-主循环)
  - [Phase 7: Causal Masking 支持](#phase-7-causal-masking-支持)
- [第五章：关键代码深度解析](#第五章关键代码深度解析)
  - [5.1 iterative_softmax 逐行解读](#51-iterative_softmax-逐行解读)
  - [5.2 AccumLambdaIterator 的设计哲学](#52-accumlambdaiterator-的设计哲学)
  - [5.3 MmaFromSharedMemory — 背靠背 GEMM 的桥梁](#53-mmafromsharedmemory--背靠背-gemm-的桥梁)
  - [5.4 EpiloguePipelined + MemoryEfficientAttentionNormalize](#54-epiloguepipelined--memoryefficientattentionnormalize)
- [第六章：性能优化清单](#第六章性能优化清单)
- [第七章：调试与验证](#第七章调试与验证)
- [第八章：扩展方向](#第八章扩展方向)
- [附录A：CUTLASS Example 41 文件索引](#附录acutlass-example-41-文件索引)
- [附录B：参考资料](#附录b参考资料)

---

## 第零章：前置知识检查清单

在开始之前，确保你对以下内容有基本了解：

- [x] **CUDA 编程基础**: `__global__`, `__shared__`, `__syncthreads()`, warp/lane 概念
- [x] **Tensor Core 编程**: 理解 `mma.sync` PTX 指令的含义（不需要手写 PTX）
- [x] **CUTLASS GEMM 基础**: 跑通过 `examples/00_basic_gemm` 或类似示例
- [x] **Attention 机制**: 理解 `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`
- [x] **内存层次**: 理解 HBM → SMEM → Register 的带宽和延迟差异

如果某些部分不熟悉，建议先补课。特别是 CUTLASS GEMM 的五层抽象，是本文的核心依赖。

---

## 第一章：Flash Attention 2 算法原理

### 1.1 标准 Attention 的问题

标准的 Self-Attention 计算：

```
S = Q @ K^T          # [N, N] — 注意力分数矩阵
P = softmax(S / sqrt(d))  # [N, N] — 注意力权重
O = P @ V             # [N, d] — 输出
```

**问题**: 中间矩阵 S 和 P 的大小是 O(N^2)。当序列长度 N=4096、head_dim=128 时：
- S 矩阵: 4096 x 4096 x 2B (FP16) = **32MB per head**
- 需要写到 HBM 再读回来，两次完整的 HBM 读写

在 A100 上，HBM 带宽约 2TB/s，但 Tensor Core 算力是 312 TFLOPS (FP16)。标准 Attention 是 **memory-bound** — 瓶颈在内存读写而非计算。

### 1.2 Tiling + Online Softmax = Flash Attention

Flash Attention 的核心思想：

1. **分块 (Tiling)**: 不一次性计算完整的 S 矩阵，而是分成小块逐步计算
2. **Online Softmax**: 边计算 S 的分块，边维护 softmax 所需的统计量（max 和 sum），避免存储完整的 S
3. **融合 (Fusion)**: 将 `Q@K^T -> softmax -> @V` 三步融合在一个 kernel 中

**Online Softmax 的数学基础**:

对于 softmax `p_i = e^{x_i} / sum_j(e^{x_j})`，可以用如下的递推方式计算：

假设已经处理了前 j-1 块，维护:
- `m^{(j-1)}`: 到目前为止的行最大值
- `l^{(j-1)}`: 到目前为止的 `sum(e^{x_i - m^{(j-1)}})`（缩放后的指数和）
- `O^{(j-1)}`: 到目前为止的部分输出

当新的一块 `S^{(j)} = Q_i @ K_j^T` 到来时：

```
m^(j)     = max(m^(j-1), rowmax(S^(j)))
l^(j)     = exp(m^(j-1) - m^(j)) * l^(j-1) + rowsum(exp(S^(j) - m^(j)))
O^(j)     = exp(m^(j-1) - m^(j)) * O^(j-1)  +  exp(S^(j) - m^(j)) @ V_j
```

最终: `O = O^(last) / l^(last)`

关键：**不需要存储完整的 N x N 矩阵**。只需要 O(N) 的额外空间存 m 和 l。

### 1.3 Flash Attention 2 的改进

相比 Flash Attention 1, FA2 做了以下优化:

1. **减少非矩阵乘法运算**: 将 rescale 操作推迟，减少 warp 间同步
2. **更好的并行度**: 外层循环在 Q 的 block 上（而非 K），内层循环在 K 的 block 上
3. **warp 间分工更优化**: 不同 warp 不再各自做 softmax 的部分，而是共享同一个 Q block

在 CUTLASS example 41 的实现中（也是 xFormers 的后端）:
- **外层并行**: grid 在 Q 的 blocks 上分 CTA (一个 block of Q per CTA)
- **内层循环**: CTA 内部顺序遍历所有 K blocks
- 与论文一致：outer loop on Q, inner loop on K

### 1.4 伪代码

```python
# Grid: (ceil(Sq / Br), num_heads, batch_size)
# 每个 CTA 处理一个 Q_block (大小 Br x d)

def flash_attention_forward_kernel(Q, K, V, O):
    # 1. 加载 Q_block 到寄存器/SMEM
    q_block_idx = blockIdx.x
    Q_i = Q[q_block_idx * Br : (q_block_idx+1) * Br, :]  # [Br, d]

    # 2. 初始化
    m_i = [-inf] * Br      # 每行的 running max
    l_i = [0.0] * Br       # 每行的 running sum
    O_i = zeros(Br, d)     # 每行的 running output

    # 3. 内层循环: 遍历所有 K/V blocks
    for j in range(0, Sk, Bc):
        K_j = K[j : j+Bc, :]      # [Bc, d]
        V_j = V[j : j+Bc, :]      # [Bc, d]

        # --- GEMM 0: Q @ K^T ---
        S_ij = Q_i @ K_j^T         # [Br, Bc], 在寄存器中

        # --- (可选) Causal Masking ---
        S_ij = apply_causal_mask(S_ij, q_block_idx, j)

        # --- Online Softmax ---
        m_ij = rowmax(S_ij)         # 当前块的行最大值
        m_new = max(m_i, m_ij)      # 更新全局行最大值

        # rescale factor
        alpha = exp(m_i - m_new)    # 旧值的缩放因子
        P_ij = exp(S_ij - m_new)    # 当前块的 softmax 分子

        l_new = alpha * l_i + rowsum(P_ij)

        # rescale 之前的输出
        O_i = alpha * O_i

        # --- GEMM 1: P @ V ---
        O_i += P_ij @ V_j           # [Br, d], 累加到寄存器

        # 更新统计量
        m_i = m_new
        l_i = l_new

    # 4. 最终归一化
    O_i = O_i / l_i

    # 5. 写回全局内存
    O[q_block_idx * Br : ...] = O_i
```

---

## 第二章：CUTLASS 架构速览

### 2.1 五层抽象

CUTLASS 将 GEMM 分解为 5 层嵌套：

```
+------------------------------------------------------+
| Device Level    (gemm/device/gemm_universal.h)       |
|   - 参数打包、kernel launch、workspace 管理            |
+------------------------------------------------------+
| Kernel Level    (gemm/kernel/gemm_universal.h)       |
|   - __global__ 函数体                                 |
|   - Swizzle 映射 blockIdx -> tile 坐标                |
|   - 实例化 Mma + Epilogue                             |
+------------------------------------------------------+
| Threadblock Level (gemm/threadblock/mma_multistage.h)|
|   - 管理 SMEM 的 double/multi-buffer                  |
|   - cp.async 从 global -> SMEM                       |
|   - 循环 K-tiles, 调用 warp-level MMA                 |
+------------------------------------------------------+
| Warp Level      (gemm/warp/mma_tensor_op.h)          |
|   - LDMATRIX 从 SMEM -> registers                    |
|   - 调用 mma.sync PTX 指令                            |
+------------------------------------------------------+
| Thread Level    (arch/mma_sm80.h)                    |
|   - PTX mma.sync.aligned.m16n8k16 指令封装            |
+------------------------------------------------------+
```

**对于 Flash Attention，我们主要工作在 Kernel 层和 Threadblock 层**。因为我们需要自定义两个 GEMM 之间的数据流，这超出了标准 GEMM 的 Device-level 接口。

### 2.2 关键模板参数

```cpp
// 三层 Tile 形状 — 以 GemmShape<M, N, K> 表示
ThreadblockShape = GemmShape<128, 128, 32>  // 一个 CTA 每次 K 迭代处理的 tile
WarpShape       = GemmShape<64, 64, 32>     // 一个 warp 处理的 tile
InstructionShape= GemmShape<16, 8, 16>      // 一条 mma.sync 指令的 shape

// 派生值（编译期计算）
WarpCount::kM = ThreadblockShape::kM / WarpShape::kM = 128/64 = 2
WarpCount::kN = ThreadblockShape::kN / WarpShape::kN = 128/64 = 2
Total warps   = WarpCount::kM * WarpCount::kN = 4
Total threads = 4 * 32 = 128
```

**对于 Flash Attention 中 GEMM0 (Q@K^T)**:
```
ThreadblockShape = GemmShape<kQueriesPerBlock, kKeysPerBlock, ThreadK>
                 // 例如: GemmShape<64, 64, 32>
WarpShape        = GemmShape<32, 32, WarpK>
```

### 2.3 共享内存管理

CUTLASS 中的 SMEM 是静态类型的 struct，编译期确定大小：

```cpp
// MmaBase::SharedStorage (in gemm/threadblock/mma_base.h)
struct SharedStorage {
    AlignedBuffer<ElementA, ShapeA::kCount> operand_A;  // A tile 的环形缓冲
    AlignedBuffer<ElementB, ShapeB::kCount> operand_B;  // B tile 的环形缓冲
};
```

对于 Ampere 的 `MmaMultistage` (3+ stages)：
- SMEM 中有 `kStages` 个 slot 的环形缓冲区
- `cp.async` 异步填充后续 slot，同时 MMA 消费当前 slot
- `cp_async_wait<kStages - 2>()` 确保不会读到还没写完的数据

**在 Flash Attention 中，SMEM 还有额外用途**:
- 存储 GEMM0 的输出 (attention scores) 供 GEMM1 读取
- 存储 online softmax 的统计量 (`m_prime`, `s_prime`, `mi`, `out_rescale`)

### 2.4 Epilogue 系统

标准 GEMM 的 Epilogue 负责将 warp 寄存器中的累加器结果写回全局内存:

```
AccumulatorFragment (registers)
   -> WarpTileIterator -> SMEM (重排列)
   -> __syncthreads()
   -> SharedLoadIterator -> 以合并访问的顺序读取
   -> OutputOp (如 alpha*acc + beta*C)
   -> OutputTileIterator -> Global Memory
```

Flash Attention 中需要**自定义 Epilogue**:
- GEMM0 不需要标准 epilogue — 结果留在寄存器中做 softmax
- GEMM1 需要特殊的 OutputOp: `O_final = O_accum / s_prime` (除以 softmax 的分母)

---

## 第三章：整体架构设计

### 3.1 为什么不能直接用一个 GEMM？

Flash Attention 涉及两个矩阵乘法，中间穿插一个 softmax。标准 CUTLASS GEMM 只做一次矩阵乘:

```
标准 GEMM:  C = a * A @ B + b * C

Flash Attention:
  Step 1:  S = Q @ K^T              (GEMM)
  Step 2:  P = online_softmax(S)    (非线性，逐元素 + 规约)
  Step 3:  O += P @ V               (GEMM)
```

Step 1 的输出 (S) 是 Step 2 的输入，Step 2 的输出 (P) 是 Step 3 的输入。而且这一切发生在**循环内部**（遍历 K blocks）。因此我们需要**手动编排两个 GEMM 的调度**。

### 3.2 双 GEMM 融合架构

```
+------------------- Kernel 主循环 -------------------+
|                                                      |
|  for each K_block:                                   |
|                                                      |
|    +-----------------------------+                   |
|    | GEMM0: Q @ K_j^T           |                   |
|    |  - Q 从 Global -> SMEM     |                   |
|    |  - K_j 从 Global -> SMEM   |                   |
|    |  - MMA -> accum (寄存器)    |                   |
|    +-------------+---------------+                   |
|                  | accum 仍在寄存器中                  |
|                  v                                    |
|    +-----------------------------+                   |
|    | Online Softmax              |                   |
|    |  - 从 accum 计算 rowmax     |                   |
|    |  - 更新 m_prime, s_prime    |  <-- SMEM 统计量   |
|    |  - accum = exp(accum - mi)  |                   |
|    |  - rescale O_accum          |                   |
|    +-------------+---------------+                   |
|                  | softmax 结果写入 SMEM              |
|                  v                                    |
|    +-----------------------------+                   |
|    | GEMM1: P @ V_j             |                   |
|    |  - P 从 SMEM 读 (operand A)|                   |
|    |  - V_j 从 Global -> SMEM   |                   |
|    |  - MMA -> O_accum (寄存器)  |                   |
|    +-----------------------------+                   |
|                                                      |
|  end for                                             |
|                                                      |
|  Epilogue: O = O_accum / s_prime -> Global           |
+------------------------------------------------------+
```

### 3.3 数据流全景图

```
全局内存 (HBM)                        寄存器             共享内存 (SMEM)
-----------------                   ----------        ------------------
Q [Br x d]   --cp.async---------+-> SMEM_Q --LDMATRIX--> frag_Q
K_j [Bc x d] --cp.async---------+-> SMEM_K --LDMATRIX--> frag_K
                                                |
                                    mma.sync: frag_Q x frag_K
                                                |
                                                v
                                    accum_S [寄存器]  (Q@K^T 的结果)
                                                |
                                    iterative_softmax
                                    |-- atomicMaxFloat --> mi[SMEM]
                                    |-- exp2f(accum - mi)
                                    |-- update s_prime[SMEM]
                                    +-- rescale out_rescale[SMEM]
                                                |
                                    accumToSmem(accum_S --> si[SMEM])
                                                |
                                                v
                                           si [SMEM]  (P 矩阵的 tile)
                                                |
V_j [Bc x d] --cp.async---------+-> SMEM_V     |
                                   --LDMATRIX--> frag_V
                                                |
              WarpIteratorFromSmem <------ si[SMEM]
                                   --> frag_P
                                                |
                                    mma.sync: frag_P x frag_V
                                                |
                                                v
                                    accum_O [寄存器]  (累加的输出)
                                                |
                                    [循环结束后]
                                    EpiloguePipelined
                                    |-- accum_O / s_prime
                                    +-----------------------------> O [HBM]
```

### 3.4 共享内存布局设计

Flash Attention 的 SMEM 需要精心安排，因为在 kernel 的不同阶段需要存储不同的数据：

```cpp
struct SharedStorage {
    // ===== 始终存在 =====
    struct ScalingCoefs {
        Array<float, Br> m_prime;      // 上一轮的 max
        Array<float, Br> s_prime;      // 上一轮的 sum(exp)
        Array<float, Br> mi;           // 当前轮的 max (通过 atomicMax 更新)
        Array<float, Br> out_rescale;  // exp(m_prime - mi) rescale 因子
        Array<float, Br * WarpCount::kN> addition_storage; // 跨 warp 的部分 sum
    };

    // ===== GEMM0 阶段 =====
    typename MM0::Mma::SharedStorage mm0;  // Q 和 K 的 SMEM 缓冲

    // ===== GEMM0 结束 ~ GEMM1 阶段 (复用 mm0 的空间) =====
    union {
        typename MM0::BiasLoader::SmemTile bias;     // attention bias (可选)
        typename MM0::AccumulatorSharedStorage si;   // softmax 结果 P 存在这里
        typename MM1::Mma::SharedStorage mm1;        // V 的 SMEM 缓冲
    };

    // ===== Epilogue 阶段 (可以复用前面的空间) =====
    typename Epilogue::SharedStorage epilogue;
};
```

**关键设计**: 使用 `union` 在不同阶段复用 SMEM 空间。GEMM0 的 Q/K buffer 在 GEMM0 完成后不再需要，可以被 GEMM1 的 V buffer 或 P 矩阵复用。

---

## 第四章：实现步骤 — 分阶段动手

### Phase 0: 搭建项目骨架

**目标**: 创建一个可编译的空 kernel，能正确 launch 和接收参数。

**步骤**:

1. 在 `examples/` 下创建你的目录（如 `my_flash_attention/`）

2. 定义 `Params` 结构体:

```cpp
struct Params {
    // 输入指针
    scalar_t const* query_ptr;      // [B, Sq, num_heads, head_dim]
    scalar_t const* key_ptr;        // [B, Sk, num_heads, head_dim]
    scalar_t const* value_ptr;      // [B, Sk, num_heads, head_dim]
    scalar_t* output_ptr;           // [B, Sq, num_heads, head_dim]
    float* logsumexp_ptr;           // [B, num_heads, Sq] (可选, 供反向传播)

    // Strides (按行的 stride, 即相邻两行的偏移量)
    int32_t q_strideM;     // = num_heads * head_dim
    int32_t k_strideM;
    int32_t v_strideM;
    int32_t o_strideM;

    // 问题尺寸
    int32_t num_queries;   // Sq
    int32_t num_keys;      // Sk
    int32_t head_dim;      // d (Q/K 的 head dim)
    int32_t head_dim_value;// d_v (V 的 head dim, 通常等于 head_dim)
    int32_t num_heads;
    int32_t num_batches;

    // Attention 参数
    float scale;           // 1.0 / sqrt(head_dim)

    // Causal mask
    int32_t custom_mask_type;       // 0=no mask, 1=causal, 2=causal from top-left
    int32_t causal_diagonal_offset; // 通常为 num_keys - num_queries
};
```

3. Grid 和 Block 配置:

```cpp
dim3 grid(
    ceil_div(num_queries, kQueriesPerBlock),  // Q blocks
    num_heads,                                 // heads
    num_batches                                // batch
);
dim3 block(kWarpSize, kNumWarpsPerBlock, 1);   // 例如 (32, 4, 1) = 128 threads
```

4. 每个 CTA 计算自己负责的 Q/K/V 的起始地址:

```cpp
// 在 kernel body 开头
int32_t batch_id = blockIdx.z;
int32_t head_id  = blockIdx.y;
int64_t q_start  = batch_id * (num_queries * q_strideM)
                  + head_id * head_dim;
// 类似地计算 key/value 的 offset
query_ptr += q_start;
key_ptr   += k_start;
value_ptr += v_start;
output_ptr += o_start;
```

**验证点**: kernel 能正确 launch，`printf` 能从 GPU 输出 batch_id/head_id。

---

### Phase 1: 实现 GEMM0 (Q @ K^T)

**目标**: 使用 CUTLASS 的 threadblock-level MMA 计算 Q_block @ K_block^T。

**核心组件**:

1. **选择 MMA 类型**: 用 `FindDefaultMma` 或 `DefaultMma` 根据架构自动选择

```cpp
// 参考 examples/41_fused_multi_head_attention/kernel_forward.h:414
using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
    scalar_t,                        // ElementA (Q)
    cutlass::layout::RowMajor,       // LayoutA — Q 是行主序
    kAlignmentA,                     // 128 / sizeof_bits<scalar_t> = 8 for fp16
    scalar_t,                        // ElementB (K)
    cutlass::layout::ColumnMajor,    // LayoutB — K^T, 所以 K 本身是列主序
    kAlignmentB,
    float,                           // 累加器用 float
    cutlass::layout::RowMajor,       // LayoutC
    cutlass::arch::OpClassTensorOp,  // 使用 Tensor Cores
    cutlass::arch::Sm80,             // Ampere
    ThreadblockShape,                // 如 GemmShape<64, 64, 32>
    WarpShape,                       // 如 GemmShape<32, 32, 32>
    InstructionShape,                // GemmShape<16, 8, 16> for fp16
    4,                               // Stages (Ampere 支持 3+)
    cutlass::arch::OpMultiplyAdd     // 操作
>::DefaultMma;
```

2. **创建 Tile Iterators (Q 和 K 的全局内存 -> SMEM)**:

```cpp
// Q 的 iterator — 读取 [Br, d] 的 tile
typename MM0::IteratorA iterator_A(
    typename MM0::IteratorA::Params(
        typename MM0::MmaCore::LayoutA(q_strideM)),  // stride
    query_ptr,                                        // 起始指针
    {problem_size_0_m, head_dim},                     // extent
    thread_id(),                                      // 线程ID
    {0, 0}                                            // 偏移
);

// K 的 iterator — 读取 [d, Bc] (列主序 = K^T)
typename MM0::IteratorB iterator_B(
    typename MM0::IteratorB::Params(
        typename MM0::MmaCore::LayoutB(k_strideM)),
    key_ptr + iter_key_start * k_strideM,             // 每轮 K block 的偏移
    {head_dim, problem_size_0_n},
    thread_id(),
    {0, 0}
);
```

3. **执行 MMA**:

```cpp
// 创建 MMA 对象
typename MM0::Mma mma(shared_storage.mm0, thread_id(), warp_id, lane_id);

// 清空累加器
typename MM0::Mma::FragmentC accum;
accum.clear();

// K 维度的迭代次数
auto gemm_k_iterations = ceil_div(head_dim, MM0::Mma::Shape::kK);

// 执行 threadblock-level GEMM
mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
// 此时 accum 持有 Q_block @ K_block^T 的结果，在寄存器中
```

**验证点**: 将 accum 写到全局内存，用 CPU 代码验证 Q@K^T 结果正确。

**关键理解**: `accum` 是一个 `Fragment` 类型（本质上是一个固定大小的 `Array<float, N>`），分布在 warp 内的各个线程的寄存器中。每个线程持有一部分元素，这些元素的 (row, col) 映射由 Tensor Core 的 MMA 布局决定。

---

### Phase 2: 实现 Online Softmax（寄存器级别）

**目标**: 在寄存器中的 `accum` 上原地执行 online softmax。

这是 Flash Attention 最核心也最精巧的部分。

**挑战**: `accum` 的元素分布在不同线程中，但 softmax 需要对**整行**求 max 和 sum。需要 warp 内归约 + SMEM 通信。

**实现思路** (参考 `kernel_forward.h:1152-1298`):

#### Step 2.1: 缩放

```cpp
constexpr float kLog2e = 1.4426950408889634074f;  // log2(e)
// 将 scale (= 1/sqrt(d)) 和 log2(e) 合并，后续用 exp2f 代替 expf (更快)
frag = scaling * kLog2e * frag;
// 此时 frag[i] = S_ij * scale * log2(e)
// 后续 exp2f(frag[i] - mi) = exp(S_ij * scale - mi / log2(e))
```

**为什么用 `exp2f` 代替 `expf`?**
`exp2f` 在硬件上比 `expf` 快约 2x (直接映射到 PTX 的 `ex2.approx.ftz.f32` 指令)。

#### Step 2.2: 行最大值 (Row Max)

```cpp
// 使用 AccumLambdaIterator 遍历 accum 中属于每一行的元素
accum_t max_val;
LambdaIterator::iterateRows(
    lane_offset,
    [&](int accum_m) {  // 行开始
        max_val = -infinity;
    },
    [&](int accum_m, int accum_n, int idx) {  // 每个元素
        if (accum_n < max_col) {
            max_val = cutlass::fast_max(max_val, frag[idx]);
        }
    },
    [&](int accum_m) {  // 行结束 — 写入 SMEM
        atomicMaxFloat(&mi[accum_m], max_val);
    }
);
__syncthreads();
```

**关于 `AccumLambdaIterator`**:
这个工具类把 Tensor Core 累加器 fragment 中扁平的索引 (`idx`) 映射回二维的 (行, 列)。因为 Tensor Core 的数据布局是非平凡的（不是简单的行优先），所以需要这个迭代器。

**为什么用 `atomicMaxFloat`?**
一行的元素可能分布在不同 warp 的不同线程中。`atomicMax` 是最简单的跨 warp 归约方式。

#### Step 2.3: Rescale 旧输出 + 更新统计量

```cpp
// 只有部分线程需要执行 (每个 warp 处理几行)
if (lane_id < kLinesPerWarp) {
    int id = warp_id * kLinesPerWarp + lane_id;

    auto m_prime_id = m_prime[id];   // 上一轮的 max
    auto mi_id = mi[id];             // 当前轮的 max (刚刚 atomicMax 算出来的)

    if (m_prime_id < mi_id) {        // max 增大了, 需要 rescale
        auto rescale = exp2f(m_prime_id - mi_id);
        out_rescale[id] = rescale;
        s_prime[id] *= rescale;      // 旧的 sum 也要 rescale
    } else {
        out_rescale[id] = 1.0f;      // max 没变, 不需要 rescale
    }
}
__syncthreads();

// Rescale 之前积累的输出 O
LambdaIterator::iterateRows(
    lane_offset,
    [&](int accum_m) { line_rescale = out_rescale[accum_m]; },
    [&](int accum_m, int accum_n, int idx) {
        frag_o[idx] *= line_rescale;
    },
    [&](int accum_m) {}
);
```

#### Step 2.4: 计算 exp 并累加 sum

```cpp
LambdaIterator::iterateRows(
    lane_offset,
    [&](int accum_m) { mi_row = mi[accum_m]; },
    [&](int accum_m, int accum_n, int idx) {
        frag[idx] = (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : 0.0f;
    },
    [&](int accum_m) {}
);

// 累加每行的 sum
LambdaIterator::iterateRows(
    lane_offset,
    [&](int accum_m) { total_row = 0.0; },
    [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
    [&](int accum_m) {
        // warp 内归约
        if (LambdaIterator::reduceSameRow(lane_id, total_row, plus)) {
            addition_storage[accum_m + kQueriesPerBlock * warp_col] = total_row;
        }
    }
);
__syncthreads();

// 汇总所有 warp 的部分 sum
if (lane_id < kLinesPerWarp) {
    int id = warp_id * kLinesPerWarp + lane_id;
    accum_t total = s_prime[id];
    for (int i = 0; i < WarpCount::kN; ++i) {
        total += addition_storage[id + kQueriesPerBlock * i];
    }
    s_prime[id] = total;
    m_prime[id] = mi[id];  // 更新 m_prime 供下一轮使用
}
```

**验证点**: 在单个 K block 的情况下，验证 softmax 结果与 CPU 参考一致。

---

### Phase 3: Accumulator -> SMEM 传递

**目标**: 将 softmax 之后的 attention scores (P) 从寄存器写入 SMEM，以便 GEMM1 使用。

这一步使用 CUTLASS 的 **Back-to-Back GEMM (B2bGemm)** 机制:

```cpp
// 参考 kernel_forward.h:908-909
// 从 examples/41_fused_multi_head_attention/gemm/mma_from_smem.h

// 确定当前 warp 在 output tile 中的坐标
int warp_idx_mn = my_warp_id % (WarpCount::kM * WarpCount::kN);
auto output_tile_coords = MatrixCoord{
    warp_idx_mn % WarpCount::kM,
    warp_idx_mn / WarpCount::kM
};

// 将寄存器中的 accum 写到 SMEM 的 AccumulatorSharedStorage
MM0::B2bGemm::accumToSmem(
    shared_storage.after_mm0.si,  // SMEM 目标
    accum,                         // 寄存器源 (softmax 后的 P)
    my_lane_id,
    output_tile_coords
);
__syncthreads();
```

**`AccumulatorSharedStorage`** 的布局:

```cpp
// gemm/mma_from_smem.h:79
template <typename Shape, typename Element, typename Layout, typename Padding>
struct AccumulatorSharedStorage {
    using AccumulatorLayout = Layout;  // 通常 RowMajor
    cutlass::AlignedBuffer<Element, Shape::kM * Shape::kN + Padding> buffer;
    // Shape = (kQueriesPerBlock, kKeysPerBlock)
    // Element = scalar_t (fp16/bf16) — 注意不是 float!
    // 写入时会从 float 降精度到 scalar_t
};
```

**关键**: 从 `float` 累加器到 `scalar_t` SMEM 的精度转换发生在这一步。

---

### Phase 4: 实现 GEMM1 (Attn @ V) — 从 SMEM 读 A

**目标**: P (在 SMEM 中) @ V (从 Global Memory) = O_partial (在寄存器中)。

GEMM1 的特殊之处: **operand A (即 P) 不是从全局内存加载的，而是直接从 SMEM 读取**。

CUTLASS 的 `MmaFromSharedMemory` 类专门为此设计:

```cpp
// 参考 kernel_forward.h:519-531
using WarpIteratorA = typename cutlass::gemm::threadblock::
    DefaultWarpIteratorAFromSharedMemory<
        typename DefaultGemm::Mma::Policy::Operator::Shape,        // warp op shape
        typename DefaultGemm::Mma::Policy::Operator::InstructionShape,
        typename DefaultGemm::Mma::Policy::Operator::IteratorA,
        typename DefaultGemm::Mma::Policy>::WarpIterator;

using DefaultMmaFromSmem =
    typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
        typename DefaultGemm::Mma,
        kKeysPerBlock,           // kMaxK — SMEM 中 P 矩阵的 K 维度
        WarpIteratorA,
        false                    // kScaleOperandA = false
    >;
using Mma = typename DefaultMmaFromSmem::Mma;
```

**`WarpIteratorFromSmem`** (`iterators/warp_iterator_from_smem.h`):
- 使用 `ldmatrix` (LDMATRIX PTX) 指令从 SMEM 高效加载到寄存器
- 支持自动转置（如果需要）
- 处理 SMEM 中 accumulator 布局到 MMA 输入布局的转换

GEMM1 的执行：

```cpp
// V 的 iterator (从全局内存)
typename MM1::IteratorB iterator_V(
    typename MM1::IteratorB::Params{typename MM1::LayoutB(v_strideM)},
    value_ptr + iter_key_start * v_strideM,
    {problem_size_1_k, problem_size_1_n},  // [Bc, d_v]
    thread_id(),
    {0, 0}
);

// GEMM1 的 MMA — A 从 SMEM 读, B 从 Global 读
typename MM1::Mma mma_pv(
    shared_storage.after_mm0.mm1,  // SMEM (与 si union 共享)
    shared_storage.after_mm0.si,   // P 的 SMEM
    thread_id(),
    warp_id,
    lane_id
);

mma_pv(gemm_k_iterations_1, accum_o, iterator_V, accum_o);
// accum_o 累加了 P @ V_j 的结果
```

**注意**: `accum_o` 是**跨循环累加的**。每轮 K block 的结果都 += 到同一个 `accum_o` 中（因为 softmax 的 rescale 已经在 Phase 2 中处理了）。

**验证点**: 对单个 K block, 验证 `softmax(Q@K^T) @ V` 的结果正确。

---

### Phase 5: 自定义 Epilogue — Rescale 输出

**目标**: 循环结束后，将 `accum_o / s_prime` 写回全局内存。

需要自定义的 `OutputOp`:

```cpp
// 参考 epilogue/epilogue_rescale_output.h
template <typename ElementOutput, typename ElementSource, int Count,
          typename ElementAccumulator, typename ElementCompute,
          bool isFirst, bool isLast, typename FragmentS_prime>
struct MemoryEfficientAttentionNormalize {
    FragmentS_prime const& s_prime;
    FragmentS_prime const& out_rescale;

    // 每个 output 元素的计算:
    CUTLASS_DEVICE FragmentOutput operator()(
        int row,
        FragmentAccumulator const& accum,
        FragmentSource const& source) const {

        FragmentOutput result;
        for (int i = 0; i < Count; ++i) {
            if (isFirst) {
                // 第一个 K block: 不需要 rescale 之前的值
                result[i] = accum[i] / s_prime[row];
            } else if (isLast) {
                // 最后一个 K block: rescale 并归一化
                result[i] = (source[i] * out_rescale[row] + accum[i]) / s_prime[row];
            } else {
                // 中间的 K block: rescale 之前的值并加上新值
                result[i] = source[i] * out_rescale[row] + accum[i];
                // 不除以 s_prime, 因为后续还会 rescale
            }
        }
        return result;
    }
};
```

**注意**: Example 41 的实际实现有两种路径:
- `kKeepOutputInRF = true`: 输出一直保留在寄存器中（如果 `kMaxK` 小，即 `head_dim` 小），循环结束后一次性 epilogue
- `kKeepOutputInRF = false`: 每轮循环将中间结果写回全局内存的 accumulator buffer，下一轮再读回来 rescale

对于常见的 head_dim=128, 输出通常保留在寄存器中。

---

### Phase 6: 组装 Kernel 主循环

把 Phase 1-5 组装到一起：

```cpp
template <typename AttentionKernel>
__global__ void __launch_bounds__(
    AttentionKernel::kNumThreads, AttentionKernel::kMinBlocksPerSm)
attention_kernel(typename AttentionKernel::Params p) {

    // 使用动态共享内存
    extern __shared__ char smem_buffer[];
    auto& shared_storage = *reinterpret_cast<
        typename AttentionKernel::SharedStorage*>(smem_buffer);

    // 1. 计算当前 CTA 负责的 batch/head/query block
    // 2. 调整指针偏移

    // 3. 初始化 online softmax 统计量
    if (thread_id() < kQueriesPerBlock) {
        shared_storage.m_prime[thread_id()] = -infinity;
        shared_storage.s_prime[thread_id()] = 0.0f;
    }

    typename MM1::Mma::FragmentC accum_o;
    accum_o.clear();

    // 4. 主循环
    for (int32_t iter_key_start = 0;
         iter_key_start < p.num_keys;
         iter_key_start += kKeysPerBlock) {

        __syncthreads();

        // --- Phase 1: GEMM0 (Q @ K^T) ---
        auto accum = run_gemm0(p, shared_storage, iter_key_start);

        // --- (可选) Phase 7: Causal Masking ---
        apply_causal_mask(accum, iter_key_start, ...);

        // --- Phase 2: Online Softmax ---
        iterative_softmax(accum_o, accum, mi, m_prime, s_prime, ...);

        // --- Phase 3: Accum -> SMEM ---
        accumToSmem(shared_storage.si, accum, ...);
        __syncthreads();

        // --- Phase 4: GEMM1 (P @ V) ---
        run_gemm1(accum_o, shared_storage, p, iter_key_start);
    }

    // 5. Epilogue: O = accum_o / s_prime -> Global Memory
    // --- Phase 5 ---
    run_epilogue(accum_o, shared_storage, p);

    // 6. (可选) 写 logsumexp: log(s_prime) + mi / log2(e)
    write_logsumexp(p, mi, s_prime);
}
```

**验证点**: 完整 kernel 在小尺寸 (Sq=64, Sk=64, d=64) 上与 CPU reference 一致。

---

### Phase 7: Causal Masking 支持

**目标**: 支持 causal (下三角) mask，使得每个 query 只能 attend 到它之前的 key。

```cpp
// 在 GEMM0 结果上应用 mask — 将被 mask 的位置设为 -inf
// 参考 kernel_forward.h:852-882

if (custom_mask_type != 0 &&
    min(iter_key_start + kKeysPerBlock, num_keys) >= query_start + offset) {

    auto lane_offset = LambdaIterator::get_lane_offset(lane_id, warp_id, ...);
    int32_t last_col;

    LambdaIterator::iterateRows(
        lane_offset,
        [&](int accum_m) {
            // 对于第 accum_m 行(绝对行号 = query_start + accum_m)
            // 允许 attend 到的最后一列 = query_start + accum_m + offset - iter_key_start
            last_col = query_start + accum_m
                     + causal_diagonal_offset - iter_key_start;
        },
        [&](int accum_m, int accum_n, int idx) {
            if (accum_n > last_col) {
                accum[idx] = -infinity;  // mask 掉
            }
        },
        [&](int accum_m) {}
    );
}
```

**优化**: 只在 mask boundary 与当前 tile 有交集时才执行 masking。完全在 mask 内或完全在 mask 外的 tile 不需要逐元素检查。

---

## 第五章：关键代码深度解析

### 5.1 iterative_softmax 逐行解读

文件: `examples/41_fused_multi_head_attention/kernel_forward.h:1152-1298`

```
函数签名:
  iterative_softmax<WarpIteratorC>(
      frag_o,           // [in/out] 之前累加的输出, 需要 rescale
      frag,             // [in/out] 当前 GEMM0 的 accum, 变成 softmax 结果
      mi,               // [SMEM] 当前轮的行 max
      m_prime,          // [SMEM] 上一轮的行 max
      s_prime,          // [SMEM] 上一轮的行 sum
      out_rescale,      // [SMEM] rescale 因子 = exp(m_old - m_new)
      addition_storage, // [SMEM] 跨 warp 的部分 sum 暂存
      ...
  )
```

**执行流程**:

```
Step 1: frag *= scale * log2(e)
        | 全部在寄存器中
        v
Step 2: 遍历 frag, 找每行 max -> atomicMaxFloat -> mi[SMEM]
        | __syncthreads()
        v
Step 3: 计算 out_rescale = exp2f(m_prime - mi)
        更新 s_prime *= out_rescale
        | __syncthreads()
        v
Step 4: frag_o *= out_rescale  (rescale 旧输出)
        | 全部在寄存器中
        v
Step 5: frag[i] = exp2f(frag[i] - mi[row])
        | 全部在寄存器中
        v
Step 6: 每行 sum -> warp 内归约 -> addition_storage[SMEM]
        | __syncthreads()
        v
Step 7: s_prime += sum(addition_storage)
        m_prime = mi (更新供下一轮使用)
```

**为什么有 3 次 `__syncthreads()`?**
1. 第一次: 确保所有 warp 的 `atomicMaxFloat` 都完成，mi[] 是最终值
2. 第二次: 确保 `out_rescale[]` 被计算完毕，才能用于 rescale `frag_o`
3. 第三次: 确保 `addition_storage[]` 被所有 warp 写完，才能汇总 sum

### 5.2 AccumLambdaIterator 的设计哲学

文件: `examples/41_fused_multi_head_attention/gemm/mma_accum_lambda_iterator.h`

**问题**: Tensor Core 的累加器 fragment 是一个扁平数组 `Array<float, N>`，但其元素在 (row, col) 空间的分布是由 MMA 指令的 register mapping 决定的。不同线程持有不同位置的元素。

**解决**: `AccumLambdaIterator` 提供了三个回调的遍历接口:

```cpp
static void iterateRows(
    LaneOffset lane_offset,
    BeginRowFn begin_row,       // (int accum_m) -> 行开始时调用
    BodyFn body,                // (int accum_m, int accum_n, int idx) -> 每个元素
    EndRowFn end_row            // (int accum_m) -> 行结束时调用
);
```

内部会根据 warp MMA 的具体布局，正确地将 fragment 中的第 `idx` 个元素映射到 `(accum_m, accum_n)` 坐标。

**为什么这很重要?** Softmax 需要逐行操作（rowmax, rowsum），但元素的物理分布与逻辑行不一致。`AccumLambdaIterator` 屏蔽了这个复杂性。

### 5.3 MmaFromSharedMemory — 背靠背 GEMM 的桥梁

文件: `examples/41_fused_multi_head_attention/gemm/mma_from_smem.h`

标准 CUTLASS MMA 的 operand A 从全局内存加载 (via `cp.async`)。`MmaFromSharedMemory` 修改了这个流程:

```
标准 MMA:
  Global -> [cp.async] -> SMEM_A -> [LDMATRIX] -> Registers -> mma.sync

MmaFromSharedMemory:
  SMEM_A (已有数据) -> [LDMATRIX / WarpIteratorFromSmem] -> Registers -> mma.sync
```

关键改动:
- 不需要 `cp.async` 来加载 operand A
- 用 `WarpIteratorFromSmem` 替代标准的 warp-level A iterator
- Operand B (V) 仍然通过 `cp.async` 从全局内存加载

`WarpIteratorFromSmem` (`iterators/warp_iterator_from_smem.h`):
- 使用 `ldmatrix.sync.aligned.x4.m8n8` PTX 指令从 SMEM 高效加载
- 处理 accumulator 的 RowMajor 布局到 MMA 要求的特定布局的转换

### 5.4 EpiloguePipelined + MemoryEfficientAttentionNormalize

文件:
- `examples/41_fused_multi_head_attention/epilogue/epilogue_pipelined.h`
- `examples/41_fused_multi_head_attention/epilogue/epilogue_rescale_output.h`

`EpiloguePipelined` 是标准 CUTLASS Epilogue 的变体:
- 增加了一个 source tile iterator (用于读取之前迭代的中间结果)
- OutputOp 接收 `row_id` 参数（用于查表 `s_prime[row]` 和 `out_rescale[row]`）

`MemoryEfficientAttentionNormalize` 的核心逻辑:

```
if (isLast):   output = accumulator / s_prime[row]
               // 最终归一化: O_final = O_accum / l
```

当 `kKeepOutputInRF = true` 时，`isFirst = true, isLast = true`，整个循环只有一次 epilogue，公式简化为 `output = accum_o / s_prime[row]`。

---

## 第六章：性能优化清单

按优先级从高到低:

### 1. 保证 Tensor Core 利用率
- 确保 Q, K, V 的 head_dim 维度是 128-bit aligned (FP16 的话 8 个元素)
- ThreadblockShape 和 WarpShape 的选择要匹配 Tensor Core 的 InstructionShape

### 2. SMEM 优化
- 使用 swizzled SMEM layout 避免 bank conflict
  - CUTLASS 自动处理: `ColumnMajorTensorOpMultiplicandCongruous64b` 等
- 合理使用 `union` 复用 SMEM 空间，减少总 SMEM 占用，提高 occupancy

### 3. Global Memory 访问
- 确保合并访问 (coalesced access)
- Q 可以预加载（在整个 K 循环中 Q 不变，只加载一次到 SMEM）
- V 可以预加载 (`prologueV`): 在 GEMM0 + softmax 执行期间异步加载下一轮的 V

### 4. 流水线重叠
- `cp.async` + `cp.async.wait_group` 实现 Global->SMEM 的异步传输
- 多 stage (3-4) 的环形缓冲区实现计算-访存重叠
- V 的预加载要在 GEMM0 期间发出:
  ```cpp
  auto prologueV = [&](int blockN) {
      // 在 GEMM0 开始前发出 V 的异步加载
      MM1::Mma::prologue(shared_storage.mm1, iterator_V, thread_id(), ...);
  };
  ```

### 5. 寄存器压力管理
- `kKeepOutputInRF = true` 节省 Global Memory 带宽但增加寄存器压力
- 如果 head_dim 很大 (>128), 可能需要 `kKeepOutputInRF = false`
- 使用 `__launch_bounds__` 限制线程数以保证足够的寄存器

### 6. 减少 `__syncthreads()` 调用
- 在 softmax 中有 3 次 barrier — 这是不可避免的
- 确保 GEMM0 和 GEMM1 之间的 barrier 不多于必要

### 7. 算术优化
- `exp2f` 代替 `expf` (2x faster on GPU)
- `atomicMaxFloat` 利用浮点到整数的位映射实现原子 max
- `fmaf` (fused multiply-add) — CUTLASS 自动使用

### 8. Tile Size 选择
- `kQueriesPerBlock` (Br) 和 `kKeysPerBlock` (Bc) 的选择要平衡:
  - 更大的 tile -> 更好的 Tensor Core 利用率，但 SMEM 占用更多
  - 常见选择: Br=Bc=64 或 Br=Bc=128
  - 受限于 SMEM 容量: Ampere 最多 164KB per SM (需要 `cudaFuncSetAttribute`)

---

## 第七章：调试与验证

### 7.1 CPU Reference 实现

```cpp
// 简单的 CPU reference
void attention_reference(
    const half* Q, const half* K, const half* V, half* O,
    int Sq, int Sk, int d, float scale) {

    // 1. S = Q @ K^T
    std::vector<float> S(Sq * Sk);
    for (int i = 0; i < Sq; i++)
        for (int j = 0; j < Sk; j++) {
            float sum = 0;
            for (int k = 0; k < d; k++)
                sum += float(Q[i*d+k]) * float(K[j*d+k]);
            S[i*Sk+j] = sum * scale;
        }

    // 2. P = softmax(S)
    std::vector<float> P(Sq * Sk);
    for (int i = 0; i < Sq; i++) {
        float max_val = *std::max_element(&S[i*Sk], &S[(i+1)*Sk]);
        float sum = 0;
        for (int j = 0; j < Sk; j++) {
            P[i*Sk+j] = expf(S[i*Sk+j] - max_val);
            sum += P[i*Sk+j];
        }
        for (int j = 0; j < Sk; j++)
            P[i*Sk+j] /= sum;
    }

    // 3. O = P @ V
    for (int i = 0; i < Sq; i++)
        for (int j = 0; j < d; j++) {
            float sum = 0;
            for (int k = 0; k < Sk; k++)
                sum += P[i*Sk+k] * float(V[k*d+j]);
            O[i*d+j] = half(sum);
        }
}
```

### 7.2 分阶段验证策略

| 阶段 | 验证方法 |
|------|---------|
| Phase 1 (GEMM0) | 将 accum 写到 global memory, 对比 Q@K^T |
| Phase 2 (Softmax) | 验证 exp 值之和约等于 s_prime, max 值 = mi |
| Phase 3 (Accum->SMEM) | 将 SMEM 内容拷贝到 global, 验证与 Phase 2 输出一致 |
| Phase 4 (GEMM1) | 单轮循环: 验证 softmax(Q@K^T) @ V |
| Phase 5 (Epilogue) | 验证 O / s_prime |
| Phase 6 (Full) | 多轮循环: 对比 CPU reference |
| Phase 7 (Causal) | 对比带 mask 的 CPU reference |

### 7.3 常见 Bug 和调试技巧

1. **数值精度**: FP16 的 softmax 容易溢出。确保 online softmax 的 max 减法在 `exp` 之前。累加器要用 float。

2. **SMEM 越界**: 使用 `compute-sanitizer --tool=memcheck` 检查:
   ```bash
   compute-sanitizer --tool=memcheck ./my_flash_attention
   ```

3. **Race Condition**: `__syncthreads()` 缺失或位置不对。特别是 softmax 中的 3 个 barrier。

4. **Warp Divergence**: 用 `AccumLambdaIterator` 时确保 boundary check 正确。

5. **调试打印**: 在 kernel 中用条件打印:
   ```cpp
   if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
       printf("mi[0] = %f, s_prime[0] = %f\n", float(mi[0]), float(s_prime[0]));
   }
   ```

---

## 第八章：扩展方向

当你完成了基础的 Flash Attention forward pass 之后，可以考虑以下扩展:

### 8.1 Backward Pass
- 需要存储 logsumexp 供反向使用: `LSE[i] = mi[i] / log2(e) + log(s_prime[i])`
- 反向传播涉及 dQ, dK, dV 三个梯度，每个需要两个 GEMM
- 参考 `examples/41_fused_multi_head_attention/kernel_backward.h`

### 8.2 Variable Sequence Length (Grouped FMHA)
- 不同 batch 内的序列长度不同
- 使用 "problem visitor" 模式动态分配 CTA 到不同的 (batch, head) 对
- 参考 `examples/41_fused_multi_head_attention/fmha_grouped.h`

### 8.3 Multi-Query / Grouped-Query Attention (MQA/GQA)
- K/V 的 head 数量少于 Q
- 修改 head 索引映射: `kv_head_id = q_head_id / num_q_per_kv_head`

### 8.4 Dropout
- 在 P (softmax 结果) 上应用 dropout
- 使用 Philox RNG 保证 forward/backward 的随机数一致
- 参考 `kernel_forward.h:926-971`

### 8.5 Attention Bias
- 支持 ALiBi 等位置编码方式
- 在 GEMM0 结果上加 bias: `S_ij += bias_ij`
- 参考 `kernel_forward.h:820-850`

### 8.6 迁移到 CUTLASS 3.x / CuTe
- CUTLASS 3.x 使用 CuTe 库重新设计了抽象层
- 更灵活的 layout 和 tensor 描述
- 参考 `examples/77_blackwell_fmha/` (SM100) 和 `examples/python/CuTeDSL/ampere/flash_attention_v2.py`

---

## 附录A：CUTLASS Example 41 文件索引

```
examples/41_fused_multi_head_attention/
|-- kernel_forward.h                    # 主 kernel: AttentionKernel struct
|   |-- Params (line 170)              # kernel 参数
|   |-- MM0 (line 387)                 # GEMM0 (Q@K^T) 类型定义
|   |-- MM1 (line 470)                 # GEMM1 (P@V) 类型定义
|   |-- SharedStorage (line 553)       # SMEM 布局
|   |-- attention_kernel() (line 630)  # 主 kernel body
|   +-- iterative_softmax() (line 1152)# online softmax 实现
|
|-- kernel_backward.h                   # 反向 kernel
|-- fmha_grouped.h                      # 变长序列版本
|
|-- gemm/
|   |-- custom_mma.h                   # 自定义 MMA (限制 K 迭代次数)
|   |-- custom_mma_multistage.h        # 自定义多阶段 MMA
|   |-- custom_mma_base.h              # 自定义 MMA base class
|   |-- find_default_mma.h             # MMA 类型选择器
|   |-- mma_from_smem.h               # 从 SMEM 读 A 的 MMA
|   |   |-- AccumulatorSharedStorage   # SMEM 中的 P 矩阵存储
|   |   |-- MmaBaseFromSharedMemory    # base class
|   |   +-- DefaultMmaFromSharedMemory # 默认配置
|   +-- mma_accum_lambda_iterator.h    # 累加器遍历工具
|
|-- epilogue/
|   |-- epilogue_pipelined.h           # 流水线 epilogue (支持 source + row_id)
|   |-- epilogue_rescale_output.h      # 归一化 OutputOp: O / s_prime
|   +-- epilogue_thread_apply_logsumexp.h  # logsumexp 计算
|
|-- iterators/
|   +-- warp_iterator_from_smem.h      # 从 SMEM 读 warp tile (LDMATRIX)
|
|-- transform/
|   +-- tile_smem_loader.h             # Bias tile 加载器
|
|-- debug_utils.h                       # 调试工具
|-- gemm_kernel_utils.h                 # 通用工具函数
|
|-- fused_multihead_attention_fixed_seqlen.cu    # 定长入口
|-- fused_multihead_attention_variable_seqlen.cu # 变长入口
+-- fused_multi_head_attention_backward.cu       # 反向入口
```

---

## 附录B：参考资料

### 论文
- **Flash Attention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
- **Flash Attention 2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023

### CUTLASS
- CUTLASS GitHub: https://github.com/NVIDIA/cutlass
- Example 00 (Basic GEMM): `examples/00_basic_gemm/`
- Example 13 (B2B GEMM): `examples/13_two_tensor_op_fusion/` — 背靠背 GEMM 的基础
- Example 41 (FMHA): `examples/41_fused_multi_head_attention/` — 本文主要参考

### 其他实现
- xFormers: https://github.com/facebookresearch/xformers — Example 41 的工业化版本
- Flash Attention Official: https://github.com/Dao-AILab/flash-attention — Tri Dao 的官方实现
- CuTe DSL Flash Attention: `examples/python/CuTeDSL/ampere/flash_attention_v2.py`

---

> **建议的学习路径**:
>
> 1. 先跑通 `examples/00_basic_gemm`，理解 CUTLASS 的编译和运行
> 2. 阅读 `examples/13_two_tensor_op_fusion`，理解 B2B GEMM 的概念
> 3. 编译并运行 `examples/41_fused_multi_head_attention`，观察输入输出
> 4. 按照本文 Phase 0-7 的顺序，从零实现你自己的版本
> 5. 每个 Phase 完成后都做数值验证，不要跳步
