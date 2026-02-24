# 融合算子实现计划: Conv3x3+ReLU+MaxPool+Conv1x1+ReLU+MaxPool

## Status

按照第 11 节的实现顺序，当前进度如下:

| Step | 文件 | 状态 |
|------|------|------|
| Step 1 | `epilogue/predicated_tile_iterator_pooled_output.h` | done |
| Step 2 | `epilogue/default_epilogue_tensor_op_with_pool.h` | done |
| Step 3 | `kernel/default_conv2d_fprop_with_pool.h` | done |
| Step 4 | `kernel/conv2d_fprop_with_pool.h` | done |
| Step 5 | `device/conv2d_fprop_with_pool.h` | done |
| Step 6 | `conv_relu_pool_run.h` | done |
| Step 7 | `fused_conv_relu_pool_f16_sm80.cu` | done |
| Step 8 | `CMakeLists.txt` + 编译验证 | done (已修复编译错误，待验证) |
| Step 9 | 性能 profiling | TODO |

所有文件位于 `examples/conv_pool_fused/` 下。

---

## 编译错误分析与修复 (Step 8)

### 错误现象

```
mma_tensor_op_policy.h(58): error: incomplete type
    "cutlass::arch::Mma<GemmShape<16,8,16>, 32, half_t, RowMajor,
     half_t, ColumnMajor, float, RowMajor, cutlass::arch::OpClassTensorOp>"
    is not allowed
    detected during instantiation of "MmaTensorOpPolicy<...>"
    at default_epilogue_tensor_op_with_pool.h(83)
```

### 根因分析

**关键区别: `arch::OpClassTensorOp` vs `arch::OpMultiplyAdd`**

CUTLASS 中有两个不同用途的 operator tag:

| Tag | 作用 | 用法 |
|-----|------|------|
| `arch::OpClassTensorOp` | **操作类选择器** (OpClass selector) | 传给 `DefaultMmaCore` 选择 TensorOp 专特化路径 |
| `arch::OpMultiplyAdd` | **实际指令操作符** (Instruction operator) | 传给 `DefaultMmaTensorOp` 和 `arch::Mma<..., OpMultiplyAdd>` |

**实例化链追踪:**

```
fused_conv_relu_pool_f16_sm80.cu:119
  DefaultConv2dFpropWithPool<..., MathOperatorTag=OpClassTensorOp>
    |
    v
default_conv2d_fprop_with_pool.h:85
  DefaultMmaCore<..., arch::OpClassTensorOp, Stages, MathOperatorTag=OpClassTensorOp>
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                      Operator_ = OpClassTensorOp (错误!)
    |
    v
default_mma_core_sm80.h (SM80 specialization)
  DefaultMmaTensorOp<WarpShape, InstrShape, ElementA, ..., Operator_=OpClassTensorOp>
    |
    v  (无 OpClassTensorOp 特化, 使用 base template)
default_mma_tensor_op.h:99 (base template)
  Policy = MmaTensorOpPolicy<arch::Mma<GemmShape<16,8,16>, 32, half_t, RowMajor,
                                       half_t, ColumnMajor, float, RowMajor,
                                       arch::OpClassTensorOp>>  <-- INCOMPLETE!
    |
    v
arch/mma.h:162
  struct Mma;   // 只有前向声明, 没有 OpClassTensorOp 的特化 -> INCOMPLETE TYPE
```

**为什么标准 CUTLASS 示例不报错?**

标准 `examples/16_ampere_tensorop_conv2dfprop/ampere_tensorop_conv2dfprop.cu:295`:

```cpp
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ...
  MMAOp,                          // = arch::OpClassTensorOp (OpClass selector)
  SmArch,                         // = Sm80
  ...
  cutlass::arch::OpMultiplyAdd,   // MathOperatorTag = OpMultiplyAdd (正确!)
  ...
>::Kernel;
```

标准示例传入 `arch::OpMultiplyAdd` 作为 `MathOperatorTag`, 而我们的代码传入了 `arch::OpClassTensorOp`。

使用 `arch::OpMultiplyAdd` 时, `DefaultMmaTensorOp<..., arch::OpMultiplyAdd>` 命中 `mma_sm80.h` 中的完整特化:

```cpp
// arch/mma_sm80.h (COMPLETE specialization):
struct Mma<GemmShape<16, 8, 16>, 32, half_t, RowMajor, half_t, ColumnMajor,
           float, RowMajor, OpMultiplyAdd> {
  using Shape = GemmShape<16, 8, 16>;
  // ... complete definition
};
```

### 修复

**`fused_conv_relu_pool_f16_sm80.cu:119`** — 将 `MathOperatorTag` 从 `OpClassTensorOp` 改为 `OpMultiplyAdd`:

```cpp
// 修复前 (错误):
cutlass::arch::OpClassTensorOp   // OpClass selector ≠ instruction operator

// 修复后 (正确):
cutlass::arch::OpMultiplyAdd     // actual SM80 FP16 tensor core instruction operator
```

---

## 1. 背景与目标

基于 CUTLASS 2.x (Implicit GEMM 架构) 实现一个融合的卷积管线:

```
Input[N,H,W,C_in] -> Conv3x3 -> ReLU -> MaxPool2x2 -> Conv1x1 -> ReLU -> MaxPool2x2 -> Output[N,H/4,W/4,C_out]
```

核心目标是**减少中间 tensor 的全局内存读写**，将 6 个独立 kernel 压缩到 2 个 fused kernel:

```
Kernel 1: Conv3x3 + ReLU + MaxPool2x2  (自定义 Epilogue)
Kernel 2: Conv1x1 + ReLU + MaxPool2x2  (自定义 Epilogue)
```

每个 kernel 是标准 CUTLASS Conv2d implicit GEMM，但 Epilogue 的 **OutputTileIterator** 被替换为自定义的池化输出迭代器。

### 约束与假设

| 参数 | 值 |
|---|---|
| 目标架构 | SM80 (Ampere), Tensor Core |
| 数据类型 | FP16 输入/输出, FP32 累加器 |
| Layout | NHWC |
| MaxPool | 2x2, stride=2, 无 padding |
| Conv3x3 | padding=1 (same padding) |
| Conv1x1 | padding=0 |
| 约束 | P, Q 必须为偶数 |

---

## 2. CUTLASS 标准 Conv2d 模板链 (我们需要改造的对象)

一个标准的 CUTLASS Conv2d Fprop 由以下模板链组装:

```
                           用户指定
                              |
                              v
                    DefaultConv2dFprop<...>
                    (default_conv2d_fprop.h:112)
                              |
              +---------------+----------------+
              |               |                |
              v               v                v
           MmaCore       IteratorA/B     DefaultEpilogueTensorOp
     (DefaultMmaCore)   (Analytic/Opt)   (default_epilogue_tensor_op.h:526)
              |                                |
              v                    +-----------+-----------+
             Mma                   |           |           |
   (ImplicitGemmMultistage)        v           v           v
                           OutputTileIter  WarpTileIter SharedLoadIter
                           (Predicated     (TensorOp)   (SharedLoad
                            TileIterator)               Iterator)
                                           |
              +----------------------------+
              v
           Epilogue                    <-- 标准 Epilogue 模板
     (epilogue/threadblock/epilogue.h:87)
              |
              v
    ImplicitGemmConvolution            <-- Kernel 入口
     (implicit_gemm_convolution.h:68)
              |
              v
    ImplicitGemmConvolution (device)   <-- Device 接口, launch kernel
     (device/implicit_gemm_convolution.h:53)
```

**我们的改造策略: 只替换 `OutputTileIterator`，其余全部复用。**

对应源码中的关键位置:

```cpp
// default_epilogue_tensor_op.h:554-560 -- 这就是我们要替换的
using PackedOutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
    OutputTileThreadMap,
    ElementOutput,
    ScatterD,
    PermuteDLayout,
    UseCUDAStore
>;
```

替换为:

```cpp
using OutputTileIterator = PredicatedTileIteratorPooledOutput<
    OutputTileThreadMap,
    ElementOutput
>;
```

---

## 3. 全局数据流

```
                     Kernel 1                                 Kernel 2
  +--------------------------------------+  +---------------------------------------+
  |                                      |  |                                       |
  | Global Mem: A0[N,H,W,Cin]           |  | Global Mem: A1[N,P/2,Q/2,Cmid]       |
  | Global Mem: B0[Cmid,3,3,Cin]        |  | Global Mem: B1[Cout,1,1,Cmid]        |
  |      |                              |  |      |                               |
  |      v cp.async (3-stage pipeline)  |  |      v cp.async (3-stage pipeline)   |
  | Shared Mem: [smem_A0, smem_B0]      |  | Shared Mem: [smem_A1, smem_B1]       |
  |      |                              |  |      |                               |
  |      v warp-level MMA (m16n8k16)    |  |      v warp-level MMA (m16n8k16)     |
  | Registers: accum[FP32]              |  | Registers: accum[FP32]                |
  |      |                              |  |      |                               |
  |      v  自定义 Epilogue             |  |      v  自定义 Epilogue              |
  | +- ReLU: max(0, a*acc + b*bias)     |  | +- ReLU: max(0, a*acc + b*bias)      |
  | | 坐标重映射: m->(n,p,q)->(n,p/2,q/2)|  | | 坐标重映射: m->(n,p,q)->(n,p/2,q/2) |
  | | atomicMax 写入池化位置             |  | | atomicMax 写入池化位置              |
  | +-----------+----------------------+  | +-----------+------------------------+
  |             v                        |  |             v                         |
  | Global Mem: D0[N,P/2,Q/2,Cmid]      |  | Global Mem: D1[N,P/4,Q/4,Cout]       |
  +--------------------------------------+  +---------------------------------------+
```

### 数值示例 (ResNet-like)

```
Kernel 1: Conv3x3 + ReLU + MaxPool2x2
  Input:   [32, 56, 56, 64]   = 6,422,528 elements (FP16 = 12.25 MB)
  Filter:  [128, 3, 3, 64]    = 73,728 elements
  Conv输出: [32, 56, 56, 128]  = 12,845,056 elements  <-- 标准流程要写入全局内存
  Pooled:  [32, 28, 28, 128]  = 3,211,264 elements   <-- 我们实际只写这个 (1/4)

Kernel 2: Conv1x1 + ReLU + MaxPool2x2
  Input:   [32, 28, 28, 128]  = 3,211,264 elements
  Filter:  [256, 1, 1, 128]   = 32,768 elements
  Conv输出: [32, 28, 28, 256]  = 6,422,528 elements   <-- 标准流程要写入全局内存
  Pooled:  [32, 14, 14, 256]  = 1,605,632 elements   <-- 我们实际只写这个 (1/4)

节省的全局内存写入量: (12,845,056 + 6,422,528) - (3,211,264 + 1,605,632)
                    = 14,450,688 elements 省下
                    = ~27.6 MB (FP16) 的全局内存写流量
```

---

## 4. 核心组件: PredicatedTileIteratorPooledOutput

这是整个方案的 **唯一核心创新点**。替换标准 `PredicatedTileIterator` 的 `store()` 方法。

### 4.1 设计思路

标准 Epilogue 中，每个线程在 `store()` 时知道自己负责的线性化 GEMM 输出坐标 `(m, k)`:

- `m` = 线性化的空间索引 (对应 N\*P\*Q 中的某个位置)
- `k` = 输出通道

我们的自定义迭代器在 store 时做 **坐标重映射 + 原子取最大值**:

```
m -> (n, p, q)  ->  (n, p/2, q/2)  ->  atomicMax 到 pooled output
```

### 4.2 CUTLASS 已有先例

`predicated_tile_iterator.h:482-548` 的 `downsample_load_with_byte_offset` 中已有完全相同的坐标分解逻辑:

```cpp
// predicated_tile_iterator.h:505-512
int output_row = row_offset + thread_start_row_;
int output_N = output_row / (convolution_P * convolution_Q);
int output_PQ = output_row % (convolution_P * convolution_Q);
int output_P = output_PQ / convolution_Q;
int output_Q = output_PQ % convolution_Q;
```

我们需要的是方向相反的操作: 从全分辨率 **store** 到 1/2 分辨率的 pooled 输出。

### 4.3 Params 结构

```cpp
struct Params : PredicatedTileIteratorParams {  // 继承标准 stride/advance 参数

    // 池化相关参数
    int N;                // batch size
    int P;                // conv 输出高度
    int Q;                // conv 输出宽度
    int K;                // 输出通道数
    int P_pool;           // = P / 2
    int Q_pool;           // = Q / 2
    float *pooled_ptr;    // pooled output buffer (FP32, 用于 atomicMax)

    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(
        Layout const &layout,
        Conv2dProblemSize const &problem_size,
        float *pooled_ptr_
    ) :
        PredicatedTileIteratorParams(
            layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
            make_OutputTileThreadMapDesc<ThreadMap>()
        ),
        N(problem_size.N),
        P(problem_size.P), Q(problem_size.Q),
        K(problem_size.K),
        P_pool(problem_size.P / 2),
        Q_pool(problem_size.Q / 2),
        pooled_ptr(pooled_ptr_)
    {}
};
```

### 4.4 store() 伪代码

```cpp
CUTLASS_DEVICE
void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) const {

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
        CUTLASS_PRAGMA_UNROLL
        for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
            CUTLASS_PRAGMA_UNROLL
            for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

                int frag_row_idx = (row + ThreadMap::Iterations::kRow *
                    (group + ThreadMap::Iterations::kGroup * cluster));

                int row_offset = row * ThreadMap::Delta::kRow
                    + group * ThreadMap::Delta::kGroup
                    + cluster * ThreadMap::Delta::kCluster;

                // 1. 线性化 M 索引
                int m = row_offset + thread_start_row_;

                // 2. 分解为空间坐标
                int n = m / (P_ * Q_);
                int pq = m % (P_ * Q_);
                int p = pq / Q_;
                int q = pq % Q_;

                // 3. 计算池化后坐标
                int p_pool = p / 2;
                int q_pool = q / 2;

                // 4. 边界检查
                bool row_guard = (m < extent_row_);

                CUTLASS_PRAGMA_UNROLL
                for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
                    bool guard = row_guard && mask_.predicates[column];

                    int k = thread_start_column_
                        + column * ThreadMap::Delta::kColumn;

                    if (guard && k < K_) {
                        // 5. 计算 pooled output 地址
                        int pooled_offset =
                            ((n * P_pool_ + p_pool) * Q_pool_ + q_pool) * K_ + k;

                        // 6. 取 fragment 中对应的值
                        AccessType const *frag_ptr =
                            reinterpret_cast<AccessType const *>(&frag);

                        for (int v = 0; v < kElementsPerAccess; ++v) {
                            float val = float(frag_ptr[frag_row_idx *
                                ThreadMap::Iterations::kColumn + column][v]);

                            // 7. atomicMax (利用非负浮点的位模式序)
                            atomicMax(
                                reinterpret_cast<int*>(pooled_ptr_ + pooled_offset + v),
                                __float_as_int(val));
                        }
                    }
                }
            }
        }
    }
}
```

### 4.5 为什么 atomicMax 对非负浮点有效

IEEE-754 单精度浮点格式: `[sign(1)][exponent(8)][mantissa(23)]`

对于 `val >= 0`: sign bit = 0, 剩余 31 位的整数比较与浮点比较结果一致。指数在高位，指数越大整数值越大；指数相同时尾数越大整数值也越大。

**ReLU 保证所有输出 >= 0，所以这个技巧在这里总是正确的。**

### 4.6 atomicMax 的竞争分析

MaxPool 2x2 stride 2: 每个 pooled 输出位置被 **4 个线程** 竞争写入 (来自 2x2 窗口中的 4 个空间位置)。

SM80 `atomicMax(int)` 对 L2 cache 中的地址延迟 ~20 cycles。4 次竞争最坏 ~80 cycles 序列化。相比 conv mainloop 数千 cycles, **开销可忽略**。

---

## 5. 完整模板链组装

### 5.1 标准 CUTLASS 模板链 vs 我们的修改

```
标准 CUTLASS:                             我们的修改:
==========                                ==========

DefaultConv2dFprop                        DefaultConv2dFpropWithPool  (新)
  |                                         |
  +-> MmaCore          (不变)               +-> MmaCore          (复用)
  +-> IteratorA/B       (不变)               +-> IteratorA/B       (复用)
  +-> DefaultEpilogueTensorOp               +-> DefaultEpilogueTensorOpWithPool  (新)
       |                                         |
       +-> OutputTileThreadMap (不变)             +-> OutputTileThreadMap (复用)
       +-> PredicatedTileIterator               +-> PredicatedTileIteratorPooledOutput (新, 核心)
       +-> WarpTileIterator    (不变)             +-> WarpTileIterator    (复用)
       +-> SharedLoadIterator  (不变)             +-> SharedLoadIterator  (复用)
       +-> AccumFragIterator   (不变)             +-> AccumFragIterator   (复用)
       |                                         |
       v                                         v
       Epilogue<..., PredicatedTileIter>         Epilogue<..., PooledOutputIter> (复用模板, 换类型参数)
       |                                         |
       v                                         v
  ImplicitGemmConvolution (kernel)          ImplicitGemmConvolutionWithPool (新, 扩展 Args/Params)
       |                                         |
       v                                         v
  ImplicitGemmConvolution (device)          Conv2dFpropWithPool (新, 扩展 launch 逻辑)
```

**新增文件: 3 个。核心创新代码: 1 个 (PooledOutputIterator)。其余均为模板参数透传。**

### 5.2 DefaultEpilogueTensorOpWithPool 详细设计

```cpp
// epilogue/epilogue_with_pool.h

template <
    typename Shape_,
    typename WarpMmaTensorOp_,
    int PartitionsK,
    typename OutputOp_,
    int ElementsPerAccess
>
struct DefaultEpilogueTensorOpWithPool {

    // ---- 以下全部复用标准 DefaultEpilogueTensorOp 的类型 ----

    using OutputTileThreadMap = typename DefaultThreadMapTensorOp<
        Shape_, typename WarpMmaTensorOp_::Shape,
        PartitionsK, typename OutputOp_::ElementOutput,
        ElementsPerAccess
    >::Type;

    using AccumulatorFragmentIterator = FragmentIteratorTensorOp<...>;  // 复用
    using WarpTileIterator = TileIteratorTensorOp<...>;                 // 复用
    using SharedLoadIterator = SharedLoadIterator<...>;                  // 复用

    // ---- 唯一替换: OutputTileIterator ----

    using OutputTileIterator = PredicatedTileIteratorPooledOutput<
        OutputTileThreadMap,
        typename OutputOp_::ElementOutput
    >;

    using Padding = MatrixShape<0, 64 / sizeof_bits<ElementAccumulator>::value * 4>;

    // 组装 Epilogue (同标准路径, 只是 OutputTileIterator 不同)
    using Epilogue = epilogue::threadblock::Epilogue<
        Shape_,
        WarpMmaTensorOp_,
        PartitionsK,
        OutputTileIterator,
        AccumulatorFragmentIterator,
        WarpTileIterator,
        SharedLoadIterator,
        OutputOp_,
        Padding
    >;
};
```

### 5.3 ImplicitGemmConvolutionWithPool (kernel 级)

需要扩展 `Arguments` 和 `Params` 来传递池化参数:

```cpp
// kernel/conv2d_fprop_with_pool.h

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct ImplicitGemmConvolutionWithPool {

    // 扩展的 Arguments: 增加 pooled output 指针和 pool 参数
    struct Arguments {
        Conv2dProblemSize problem_size;
        TensorRefA ref_A;
        TensorRefB ref_B;
        TensorRefC ref_C;
        TensorRefC ref_D;                // 标准输出 (本方案中不实际使用)
        typename EpilogueOutputOp::Params output_op;
        float *pooled_output_ptr;        // 新增: pooled 输出指针 (FP32)
        int P_pool, Q_pool;              // 新增: pooled 空间维度
    };

    // 扩展的 Params: 将 pool 信息传递给 epilogue output iterator
    struct Params {
        // ... 标准字段 ...
        typename Epilogue::OutputTileIterator::Params iterator_D;  // 含 pool params
        float *pooled_output_ptr;
        // ...

        Params(Arguments const &args, int *semaphore = nullptr)
            : // 标准初始化...
              // 关键: 构造 OutputTileIterator::Params 时传入 pool 参数
              iterator_D(
                  ConvOutputIteratorParameter::layout(args.ref_D),
                  args.problem_size,
                  args.pooled_output_ptr
              )
        { }
    };

    // operator() 中: Epilogue 构造时使用扩展的 iterator_D params
};
```

---

## 6. Tile Size 与资源预算

### 6.1 Tile Size 选择

```cpp
// Kernel 1: Conv3x3 + ReLU + MaxPool2x2
using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape0        = cutlass::gemm::GemmShape<64, 32, 32>;   // 2x2 warp grid
using InstructionShape  = cutlass::gemm::GemmShape<16, 8, 16>;    // SM80 tensor core
static int const kStages = 3;

// Kernel 2: Conv1x1 + ReLU + MaxPool2x2
using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 128, 32>; // 1x1 conv 用更大 N tile
using WarpShape1        = cutlass::gemm::GemmShape<64, 64, 32>;   // 2x2 warp grid
```

### 6.2 Shared Memory 预算

无额外开销 (池化在全局内存写出时完成，不占用 smem):

| 组件 | 大小 |
|---|---|
| Operand A (3 stages) | ~128 x 96 x 2B = ~24 KB |
| Operand B (3 stages) | ~96 x 64 x 2B = ~12 KB |
| Epilogue smem (union) | ~8 KB |
| **Total** | **~36 KB** (< 48 KB default, 无需 opt-in) |

### 6.3 输出 Buffer 初始化

每次 kernel launch 前必须初始化:

```cpp
cudaMemsetAsync(pooled_output_ptr, 0, pooled_output_bytes, stream);
```

ReLU 输出 >= 0, 所以 0 是有效的初始"最小值"。`atomicMax(0, relu_val)` 正确计算最大值。

---

## 7. Kernel 执行流程

```
1. Host: cudaMemsetAsync(pooled_output, 0, size)

2. Prologue (多阶段预填充)
   +-- cp.async: 加载前 (kStages-1) 个 A/B tile 到 smem
   +-- 标准 CUTLASS multi-stage pipeline

3. Mainloop (遍历 K 维度: R*S*C, 每次 K_tile=32)
   +-- cp.async: 异步加载下一个 A/B tile
   +-- warp-level MMA: tensor core 计算当前 tile
   +-- 累加到 FP32 fragment registers
   +-- 软件 pipeline, kStages=3

4. Epilogue (自定义部分)
   +-- Fragment iterator: 提取累加器 fragments
   +-- WarpTileIterator: 写入 epilogue smem staging buffer
   +-- __syncthreads()
   +-- SharedLoadIterator: 从 smem 读取到线程寄存器
   +-- OutputOp (LinearCombinationRelu): output = ReLU(a * accum + b * bias)
   +-- PooledOutputIterator:
       +-- 分解 m -> (n, p, q)
       +-- 计算 (p_pool, q_pool) = (p/2, q/2)
       +-- atomicMax((int*)&pooled[n][p_pool][q_pool][k], __float_as_int(output))
```

---

## 8. 关键挑战与解决方案

### 8.1 atomicMax 的类型问题

| 方案 | 优势 | 劣势 |
|---|---|---|
| **FP32 中间tensor + int atomicMax** (推荐) | SM80原生atomicMax(int), 正确高效 | 中间tensor 2x大小 (但pool后1/4, 净1/2) |
| FP16 + CAS 自旋 | 最小内存 | atomicCAS loop 性能差, 高竞争 |
| 不用atomic, tile对齐pool window | 零 atomic 开销 | 限制tile size, 不通用 |

推荐: FP32 中间 tensor。经过 MaxPool 缩小到 1/4, 即使 FP32 也只有 FP16 中间 tensor 的 1/2。

### 8.2 Kernel 2 的输入类型适配

Kernel 1 输出 FP32 pooled tensor, Kernel 2 需要读入:

1. **推荐: 在 Kernel 2 前插入轻量 conversion kernel** (FP32->FP16), Kernel 2 标准读取 FP16
2. 或: 修改 Kernel 2 的 activation iterator 直接从 FP32 加载

选择 1 保持代码干净, conversion kernel 是带宽受限的简单操作。

### 8.3 OutputTileIterator 接口兼容性

标准 `Epilogue` 模板 (`epilogue.h:87`) 通过以下模板参数使用 OutputTileIterator:

```cpp
template <typename OutputTileIterator_, ...>
class Epilogue {
    // 在 operator() 中调用:
    //   iterator_D.store(frag)   -- 写输出 D
    //   ++iterator_D             -- 前进到下一个 tile 位置
    //   iterator_C.load(frag)    -- 读源 C (用于 beta*C)
};
```

我们的 `PredicatedTileIteratorPooledOutput` 必须满足:
- `store(Fragment const &frag)` -- 实现池化写出
- `operator++()` -- 前进 (可以空操作, 因为我们通过坐标计算地址)
- `Params` 构造函数兼容 -- 从 kernel 的 Params 中获取池化参数
- `Fragment` 类型 -- 与标准相同 (不变)

`load()` 方法 (用于读 source C) 可以退化为空操作/返回零, 因为 ReLU epilogue 通常 beta=0 不需要读 C。

---

## 9. 文件结构与实现清单

所有新文件放在 `examples/XX_fused_conv_relu_pool/` 下。

```
examples/XX_fused_conv_relu_pool/
+-- CMakeLists.txt
+-- fused_conv_relu_pool_f16_sm80.cu              # 主示例文件
+-- conv_relu_pool_run.h                           # 测试 harness
+-- device/
|   +-- conv2d_fprop_with_pool.h                  # Device 级接口
+-- kernel/
|   +-- conv2d_fprop_with_pool.h                  # Kernel 级 (扩展 Args/Params)
|   +-- default_conv2d_fprop_with_pool.h          # Kernel 组装元函数
+-- epilogue/
    +-- predicated_tile_iterator_pooled_output.h   # 核心: 池化输出迭代器
    +-- default_epilogue_tensor_op_with_pool.h     # Epilogue 组装元函数
```

### 文件 1: `epilogue/predicated_tile_iterator_pooled_output.h`

**核心组件, 最难, 最重要。**

- 基于 `PredicatedTileIterator` (`predicated_tile_iterator.h:79-794`) 改造
- 新增 `Params`: 含 N, P, Q, K, P_pool, Q_pool, pooled_ptr
- 重写 `store_with_byte_offset()`: 坐标分解 + atomicMax (见 4.4 节伪代码)
- `load()` / `load_with_byte_offset()`: 退化为零填充 (beta=0 不需要读 C)
- `operator++()`: 维护 `thread_start_row_` 前进 (与标准相同)
- `Fragment`, `AccessType`, `Mask`: 与标准相同, 直接复用类型定义

### 文件 2: `epilogue/default_epilogue_tensor_op_with_pool.h`

- 基于 `DefaultEpilogueTensorOp` (`default_epilogue_tensor_op.h:526-620`)
- 将 `OutputTileIterator` 替换为 `PredicatedTileIteratorPooledOutput`
- 其他所有 type alias 完全复用: OutputTileThreadMap, AccumulatorFragmentIterator, WarpTileIterator, SharedLoadIterator, Padding
- 组装 `Epilogue<..., PredicatedTileIteratorPooledOutput, ...>`

### 文件 3: `kernel/default_conv2d_fprop_with_pool.h`

- 基于 `DefaultConv2dFprop` (`default_conv2d_fprop.h:112-207`)
- MmaCore, IteratorA, IteratorB, SmemIteratorA/B, Mma: 全部复用
- Epilogue: 使用文件 2 的 `DefaultEpilogueTensorOpWithPool`
- 组装 `ImplicitGemmConvolutionWithPool<Mma, Epilogue, Swizzle>`

### 文件 4: `kernel/conv2d_fprop_with_pool.h`

- 基于 `ImplicitGemmConvolution` (`implicit_gemm_convolution.h:68-244`)
- 扩展 `Arguments`: 增加 `float *pooled_output_ptr, int P_pool, int Q_pool`
- 扩展 `Params` 构造: 将 pool 参数传给 `Epilogue::OutputTileIterator::Params`
- `operator()` 中 epilogue 调用逻辑保持不变 (类型系统自动传递)

### 文件 5: `device/conv2d_fprop_with_pool.h`

- 基于 `ImplicitGemmConvolution` device (`device/implicit_gemm_convolution.h:53-150`)
- `initialize()` 中: 调用 `cudaMemsetAsync` 清零 pooled output buffer
- `can_implement()` 中: 增加 P%2==0 && Q%2==0 检查
- `run()`: 标准 kernel launch

### 文件 6: `fused_conv_relu_pool_f16_sm80.cu`

主示例, 实例化两个 kernel 并运行:

```cpp
// Kernel 1 类型定义
using Conv3x3WithPool = cutlass::conv::device::Conv2dFpropWithPool<
    cutlass::half_t, LayoutNHWC,     // A
    cutlass::half_t, LayoutNHWC,     // B
    cutlass::half_t, LayoutNHWC,     // C
    float,                           // Accumulator
    arch::OpClassTensorOp,
    arch::Sm80,
    GemmShape<128, 64, 32>,          // ThreadblockShape
    GemmShape<64, 32, 32>,           // WarpShape
    GemmShape<16, 8, 16>,            // InstructionShape
    LinearCombinationRelu<half_t, 8, float, float>,
    SwizzleConv2dFpropNHWC,
    3                                // Stages
>;

// Kernel 2 类似, ThreadblockShape = <128, 128, 32>
```

### 文件 7: `conv_relu_pool_run.h`

测试 harness:
- `NonFusedRun`: Conv2d reference + elementwise ReLU + `pooling_nhwc(..., poolingType=1)`
- `FusedRun`: 我们的 fused kernel
- 参考: `examples/13_two_tensor_op_fusion/b2b_conv2d_run.h`

---

## 10. 验证策略

### 10.1 Golden Reference

```
对 Kernel 1:
  1. cutlass::reference::device::Conv2d(input, filter_3x3, conv_output)
  2. elementwise ReLU(conv_output)
  3. cutlass::pooling_nhwc(relu_output, pooled_output, poolingType=1)

对 Kernel 2:
  4. cutlass::reference::device::Conv2d(pooled_output, filter_1x1, conv_output2)
  5. elementwise ReLU(conv_output2)
  6. cutlass::pooling_nhwc(relu_output2, final_output, poolingType=1)
```

MaxPool reference: `tools/util/include/cutlass/util/device_nhwc_pooling.h`

### 10.2 测试用例

| 用例 | N | H | W | C_in | C_mid | C_out | 说明 |
|---|---|---|---|------|-------|-------|------|
| Sanity | 1 | 8 | 8 | 16 | 32 | 16 | 最小, 可手动验证 |
| Medium | 4 | 32 | 32 | 64 | 128 | 256 | 中等规模 |
| ResNet-like | 32 | 56 | 56 | 64 | 64 | 256 | 接近实际网络 |
| Non-square | 2 | 28 | 14 | 64 | 64 | 128 | 非方形输入 |

### 10.3 性能对比

| 配置 | Kernel 数量 | 中间内存流量 |
|---|---|---|
| Non-fused | 6 (Conv+ReLU+Pool) x 2 | 全分辨率 tensor x 4 次读写 |
| **Fused (本方案)** | **2 (+1 FP32->FP16 convert)** | **1/4 分辨率 tensor x 1 次读写** |

预期加速: **1.2x - 1.5x**

---

## 11. 实现顺序

```
Step 1: 实现 PredicatedTileIteratorPooledOutput        (核心, 最难)
        +-- 编写 Params 结构, store() 方法
        +-- 单独写单元测试验证坐标映射和 atomicMax 正确性

Step 2: 实现 DefaultEpilogueTensorOpWithPool            (模板组合, 复用为主)
        +-- 替换 OutputTileIterator 类型

Step 3: 实现 DefaultConv2dFpropWithPool                 (模板组合, 复用为主)
        +-- 替换 Epilogue 类型

Step 4: 实现 ImplicitGemmConvolutionWithPool (kernel)   (扩展 Args/Params)

Step 5: 实现 Conv2dFpropWithPool (device)               (扩展 launch, 加 memset)

Step 6: 编写 conv_relu_pool_run.h                       (Non-fused Reference + Harness)

Step 7: 编写 fused_conv_relu_pool_f16_sm80.cu           (主示例)

Step 8: 编译验证正确性 (Sanity -> Medium -> ResNet-like)

Step 9: 性能 profiling (nsys timeline + ncu kernel analysis)
```

---

## 12. 关键参考文件

| 文件 | 行号 | 作用 |
|---|---|---|
| `include/cutlass/epilogue/threadblock/predicated_tile_iterator.h` | 79-794 | 基类, store() at 386-471, downsample参考 at 482-548 |
| `include/cutlass/epilogue/threadblock/default_epilogue_tensor_op.h` | 526-620 | DefaultEpilogueTensorOp, OutputTileIterator 组装 at 554-560 |
| `include/cutlass/epilogue/threadblock/epilogue.h` | 87-100 | Epilogue 模板参数定义 |
| `include/cutlass/epilogue/thread/linear_combination_relu.h` | 79 | ReLU OutputOp (直接复用) |
| `include/cutlass/conv/kernel/default_conv2d_fprop.h` | 112-207 | Kernel 组装: MmaCore+Iters+Epilogue+Kernel |
| `include/cutlass/conv/kernel/implicit_gemm_convolution.h` | 68-244 | Kernel 入口, Args at 145, Params at 197 |
| `include/cutlass/conv/device/implicit_gemm_convolution.h` | 53-150 | Device 接口, can_implement + initialize + run |
| `examples/13_two_tensor_op_fusion/fused_two_convs_f16_sm80_rf.cu` | - | 顶层使用模式参考 |
| `tools/util/include/cutlass/util/device_nhwc_pooling.h` | 80-144 | MaxPool golden reference |
