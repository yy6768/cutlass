# CUTLASS 学习记 (4) —— Shared Memory 与 Warp MMA

# 前言

- 上一篇我们从 CUDA Debug 的原点出发，剖析了 analytic iterator 如何将卷积坐标映射为 GEMM 坐标，并通过 `cp.async` 将数据从 Global Memory 异步搬运到 Shared Memory。
- 本篇将继续沿着数据流向下游(Claude code的指挥)走：从上篇的Thread Block的mma出发，继续沿着数据流探索。

## Claude：数据流全景回顾

上一篇我们走完了数据流的第一段——从 Global Memory 到 Shared Memory。本篇覆盖剩余的两段：

```
                        Blog 3                          Blog 4
              ┌──────────────────────┐    ┌──────────────────────────────────────┐
              │                      │    │                                      │
Global Memory ──(cp.async)──→ Shared Memory ──(ldmatrix)──→ Registers ──(mma.sync)──→ Accumulator
                                │                              │                          │
                        swizzle layout                   warp tile iterator            blog5将介绍
                     避免 bank conflict               32线程协作加载               
```

具体来说，本篇需要回答个问题：

1. **Shared Memory 如何组织数据？** (NV blog的部分)
2. **如何从 Shared Memory 加载到寄存器？** —— Warp Tile Iterator 使用 `ldmatrix` 指令，32 个线程协作，一次加载 4 个 8×8 矩阵块



# Shared Memory 数据组织

上一篇中 `cp.async` 已经把数据搬进了 shared memory，但搬进去的**排布方式**并不是简单的行优先/列优先——而是经过精心设计的 **swizzle layout**，目的是同时满足两个需求：

1. **写入时无 bank conflict**：cp.async 写入 smem 时，同一 warp 内 32 个线程不能访问同一 bank
2. **读取时无 bank conflict**：后续 `ldmatrix` 从 smem 加载到寄存器时，也不能冲突

这里需要注意的点包括：

- `ldmatrix`是warp-cooperative 指令，加载32行/列，128 bits长的数据
- bank conflict需要同一个warp内的线程不访问相同的8行bank，因此
  - T{0-7} 不能同时读取相同128bank
  - T{8-15} 不能同时读取相同128bank
  - T{16-23} 不能同时读取相同128bank
  - T{24-31} 不能同时读取相同128bank


![ALT](https://docs.nvidia.com/cutlass/latest/_images/tensor-op-permuted-smem-layout-TN.png)

## Bank Conflict 与 Swizzle

Shared memory 由 **32 个 bank** 组成，每个 bank 宽 4 字节，一行 128 字节恰好覆盖全部 32 个 bank。如果一个 warp 中的多个线程同时访问同一个 bank 的不同地址，就会产生 **bank conflict**，访问将被串行化。

以朴素的行优先布局为例，如果连续的若干行在 smem 中紧密排列，那么当 `ldmatrix` 让 32 个线程同时读取不同行的同一列位置时，它们可能落在同一个 bank 上——这就是 conflict 的来源。

### XOR Swizzle 原理

CUTLASS 常规的swizzle方案是 **XOR 置换**，XOR 的效果是类似于NV blog示意图（上图）：将

具体参数（fp16, K=64）

```
kElementSize       = 16 (bits, fp16)
kElementsPerAccess = 128 / 16 = 8       （一次 128-bit 访问读 8 个 fp16）
kCrosswise         = 64                  （K 维度大小）
kTileShapeContiguous = 128 / (128/8) = 8 （一个 tile 的连续维度：8 个 vector）
kFactor            = 8 * 8 / 64 = 1
kTileShapeStride   = max(8/1, 32/8) = 8  （一个 tile 的跨步维度：8 个 vector）

TileShape     = <8, 8>   （以 vector 为单位的基本 tile）
PartitionShape = <4, 4>  （以 vector 为单位的 partition）
```

一个 8×8 的 vector tile 在物理上占 8×8×8×2 = 1024 字节 = 8 个 128B cache line。通过两级 XOR，这 8 行数据在 bank 分布上完全错开，保证无 conflict。

官方博客给的伪代码如下：

```
  // lane_id是线程在warp里的编号
  int store_column = (lane_id % 8) ^ (lane_id / 8);
```

直观理解来看

```
  朴素布局（有 bank conflict）：       Swizzle 布局（无 bank conflict）：

  row 0: [ v0 v1 v2 v3 v4 v5 v6 v7 ]   row 0: [ v0 v1 v2 v3 v4 v5 v6 v7 ]
  row 1: [ v0 v1 v2 v3 v4 v5 v6 v7 ]   row 1: [ v1 v0 v3 v2 v5 v4 v7 v6 ]  ← XOR 1
  row 2: [ v0 v1 v2 v3 v4 v5 v6 v7 ]   row 2: [ v2 v3 v0 v1 v6 v7 v4 v5 ]  ← XOR 2
  row 3: [ v0 v1 v2 v3 v4 v5 v6 v7 ]   row 3: [ v3 v2 v1 v0 v7 v6 v5 v4 ]  ← XOR 3
    ↑                                     ↑
  同列=同bank，纵向读取conflict!        每列分散到不同bank，无conflict!
```

这种 swizzle 不改变数据本身的值，只改变数据在 smem 中的**物理存放位置**。`ldmatrix` 读取时，硬件会通过同样的地址变换找到正确的数据。

### SmemIterator：写入 smem 的地址计算

cp.async 写入 smem 的目标地址由 `SmemIteratorA`（`RegularTileAccessIterator`，定义在 `transform/threadblock/regular_tile_access_iterator_tensor_op.h`）计算。它的 `get()` 方法内部调用了上述 swizzle layout 的 `operator()`，把逻辑坐标 (row, col) 转换为经过 XOR 置换的物理地址。

```
cp.async 写入流程：
  逻辑坐标 (m, k) → SmemLayout::operator() → XOR swizzle → smem 物理地址
                                                                ↓
                                              cp.async.ca.shared.global [smem_addr], [gmem_addr]
```

## ldmatrix 指令

`ldmatrix` 是 Turing (SM75) 引入的 **warp 协同指令**，一次调用让 32 个线程协作从 shared memory 加载数据到各自的寄存器中。与普通的 shared memory load 不同，`ldmatrix` 天然理解矩阵的 8×8 分块结构，能够一次性把数据重新分配到各线程的寄存器里，为后续 `mma.sync` 做好准备。

CUTLASS 在 `arch/memory_sm75.h` 中封装了这条指令：

```cpp
// ldmatrix.sync.aligned.x4.m8n8.shared.b16
asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
    : "r"(smem_addr));
```

- **`.sync`**：warp 内所有 32 个线程同步参与
- **`.x4`**：同时加载 4 个 8×8 矩阵块
- **`.m8n8.b16`**：每个块是 8 行 × 8 列 × 16bit 元素
- 每个线程提供一个 smem 地址（指向它要读取的那一行），指令把数据重新分配到 4 个 32-bit 寄存器中
- 还有 `.trans` 变体（ColumnMajor 版本）：在加载时做隐式转置

# 回到代码

还是和之前的博客一样我们还是对照着代码一步步看。

## ThreadBlock 层：Smem 数据组织与加载

### 调用栈

利用claude生成的调用栈如下

```
ImplicitGemmMultistage 构造函数
│  (conv/threadblock/implicit_gemm_multistage.h:168)
│  继承自 MmaBase<Shape, MmaPolicy, Stages>
│       (gemm/threadblock/mma_base.h:80)
│
├─ MmaBase 构造（基类, mma_base.h:214）
│   │  SharedStorage 定义了 smem buffer（mma_base.h:140）：
│   │    operand_A: AlignedBuffer [M=128, K*kStages=192]
│   │    operand_B: AlignedBuffer [K*kStages=192, N=128]
│   │
│   ├─ warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx)
│   │    类型：MmaPolicy::Operator::IteratorA（= MmaTensorOpMultiplicandTileIterator）
│   │    (gemm/warp/mma_tensor_op_tile_iterator.h:2291)
│   │    → 读取端：用 lane_idx(0-31) 初始化，warp 级 ldmatrix 操作
│   └─ warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx)
│
├─ smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx)
│    类型：SmemIteratorA_（= RegularTileAccessIterator）
│    (transform/threadblock/regular_tile_access_iterator_tensor_op.h)
│    → 写入端：用 thread_idx(0-255) 初始化，threadblock 级 cp.async 操作
├─ smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
│
└─ warp_tile_iterator 加上 per-warp 偏移（line 195）
     → add_tile_offset({warp_idx_m, ...})

---

ImplicitGemmMultistage::operator()  Prologue / Mainloop
│  (conv/threadblock/implicit_gemm_multistage.h:266)
│
├─ cp.async 写入 smem 时：
│   ├─ smem_iterator_A_.get()
│   │    (regular_tile_access_iterator_tensor_op.h:174)
│   │    → 通过 SmemLayoutA::operator() 计算 swizzle 后的物理地址
│   │    → 返回 smem 写入目标指针
│   └─ cp_async_zfill(smem_dst, global_src, is_valid)
│        (arch/memory_sm80.h:151)
│
└─ warp_tile_iterator 从 smem 读取时：
    ├─ warp_tile_iterator_A_.load(warp_frag_A)
    │    (gemm/warp/mma_tensor_op_tile_iterator.h:2594)
    │    → 内部调用 ldmatrix.x4 从 smem 加载到寄存器
    └─ warp_tile_iterator_B_.load(warp_frag_B)
         → 同理
```



### MMA Core

其实在Iterator映射完数据后很多部分都来自于传统的**GEMM管线**了（红框标记的部分）

![GEMM API 管线](https://typora-yy.oss-cn-hangzhou.aliyuncs.com/Typora-img/image-20260220190414983.png)

由于指定了SM80架构，`ImplicitGemmMultistage`继承的base来自于`default_mma_core_sm80.h`，经过 `DefaultConv2dFprop` 传递给 `ImplicitGemmMultistage`：

```
DefaultConv2dFprop (conv/kernel/default_conv2d_fprop.h:133)
│
├─ MmaCore = DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape, ...>
│    (gemm/threadblock/default_mma_core_sm80.h:1395)
│    │
│    ├─ SmemLayoutA  = RowMajorTensorOpMultiplicandCrosswise<16, 64>     ← smem 排布
│    ├─ SmemLayoutB  = ColumnMajorTensorOpMultiplicandCrosswise<16, 64>
│    │
│    ├─ SmemIteratorA = RegularTileAccessIterator<..., SmemLayoutA, ...>  ← 写入端
│    ├─ SmemIteratorB = RegularTileAccessIterator<..., SmemLayoutB, ...>  (line 1466)
│    │
│    ├─ MmaTensorOp  = DefaultMmaTensorOp<WarpShape, ..., SmemLayoutA, SmemLayoutB, ...>
│    │    └─ IteratorA/B = MmaTensorOpMultiplicandTileIterator<...>       ← 读取端
│    │
│    └─ MmaPolicy（包含 MmaTensorOp + padding 策略）            (line 1492)
│
├─ SmemIteratorA = MmaCore::SmemIteratorA   ← 直接取用 (line 152)
├─ SmemIteratorB = MmaCore::SmemIteratorB                    (line 165)
├─ MmaPolicy    = MmaCore::MmaPolicy                        (line 169)
│
└─ Mma = ImplicitGemmMultistage<..., SmemIteratorA, SmemIteratorB, MmaPolicy, ...>
         │                                                   (line 177)
         └─ 继承 MmaBase<..., MmaPolicy, Stages>
              → SharedStorage（smem buffer, 由 SmemLayout 决定排布）
              → warp_tile_iterator_A/B（来自 MmaPolicy::Operator::IteratorA/B）
```

**layout 排布**

在 `default_mma_core_sm80.h:1447`，我们的配置（fp16, ThreadblockShape 128×128×64）选择的 smem 布局：

```cpp
// A矩阵 (Activation): [M=128, K=64] — 行优先 Crosswise
using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
    sizeof_bits<ElementA>::value,  // 16 (fp16)
    Shape::kK>;                    // 64

// B矩阵 (Filter): [K=64, N=128] — 列优先 Crosswise
using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
    sizeof_bits<ElementB>::value,  // 16
    Shape::kK>;                    // 64
```

Q:这里的 **Crosswise** 是什么意思？

Claude：在 GEMM 的 A[M, K] 中，K 是"横向"维度（contiguous），M 是"纵向"维度（strided）。Crosswise 表示 **K 维度是 smem 中连续存储的方向**——这与 `ldmatrix` 指令的访问模式匹配。

这两个类型最终都委托给同一个底层布局类 `TensorOpMultiplicand<ElementSize=16, Crosswise=64>`（定义在 `layout/tensor_op_multiplicand_sm75.h:60`），只是坐标轴的映射不同（行优先交换 row/column）。

---



**Smem Iterator**

首先上篇说到，Prologue 的 iteratorA/B 访问和 `copy_tiles_and_advance` 内部，数据的 destination 都是 Smem IteratorA/B 为基指针的。
当前 Smem Iterator 是 ImplicitGemmMultistage 继承 MmaBase(DefaultMmaCore) 对应的 `RegularTileAccessIterator`。

对于我们的 SmemIteratorA（`RowMajorTensorOpMultiplicandCrosswise<16, 64>`），实际走到的是 `regular_tile_access_iterator_tensor_op.h:730` 的 RowMajor Crosswise 特化。但这只是一层薄包装，内部委托给 PitchLinear 版本的 `TensorOpMultiplicandCrosswise<16, 64>` 特化（line 435）。

**构造函数**（line 521）：

```cpp
RegularTileAccessIterator(TensorRef ref, int thread_id)
    : sections_(ref.stride(0) / kCrosswise),              // 总section数
      sections_per_stage_(Shape::kContiguous / kCrosswise), // 每个stage的section数
      stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
      byte_offset_(0)
{
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);          // 1. 256线程各自的起始偏移

    for (int i = 0; i < kPointerCount; ++i) {
        layout::PitchLinearCoord thread_offset_in_threadblock_tile =
            thread_offset_base +
            PitchLinearCoord{0, WarpThreadArrangement::kStrided * i};

        pointer_[i] = reinterpret_cast<AccessType *>(ref.data()) +
                      ref.offset(thread_offset_in_threadblock_tile) /  // 2. thread_offset(computed swizzle)
                          Layout::kElementsPerAccess;
    }
}
```


**get()**（line 564）：

```cpp
AccessType *get() const {
    AccessType *access_ptr = pointer_[iteration_strided_ & 1];   // 取预计算的基指针
    int stride_idx = (iteration_strided_ & ~1);

    int access_offset =
        stride_idx * ThreadMap::Delta::kStrided * stride_ / Layout::kFactor +
        iteration_contiguous_ * (ThreadMap::Delta::kContiguous / kCrosswise) *
            Layout::TileShape::kContiguous;                      // 纯指针算术，无swizzle

    char *access_byte_ptr = reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);
}
```

`get()` 内部只做指针偏移计算。因为 contiguous 维度的步进是按 `kCrosswise` 对齐的整数倍跳（每跳一个 section 恰好是 `TileShape::kContiguous` 个 vector）。



### XOR 置换代码

**XOR swizzle 发生在上述注释的2 处**：`ref.offset()` 调用的是 `TensorOpMultiplicandCrosswise::operator()`（`tensor_op_multiplicand_sm75.h:150`），它内部做了两级 XOR 置换。所以 `pointer_` 在构造时就已经被 bake 了 swizzle 后的物理地址——后续 `get()` 不需要再做任何 swizzle 运算。

```cpp
// 两级 XOR 置换：
// 第一级：partition 内部的 vector 级别
int permuted_vec = partition_contiguous_residual ^ (partition_strided_residual % 4);

// 第二级：tile 内部的 partition 级别
int permuted_partition = partition_contiguous_idx ^ (partition_strided_idx % 2);
```

---

## Warp 层：MMA 计算

### 调用栈

```
ImplicitGemmMultistage::operator()  Mainloop 内部
│
└─ for warp_mma_k in [0, kWarpGemmIterations):
      │
      ├─ warp_tile_iterator_A_.load(warp_frag_A[warp_mma_k % 2])
      │    (gemm/warp/mma_tensor_op_tile_iterator.h:2594)
      │    → load_with_byte_offset() 内部循环 4 次 ldmatrix.x4
      │    → 填充 FragmentA = Array<half_t, 32>（每线程 32 个 half）
      │
      ├─ warp_tile_iterator_B_.load(warp_frag_B[warp_mma_k % 2])
      │    → 同理，填充 FragmentB
      │
      ├─ ++warp_tile_iterator_A_ / ++warp_tile_iterator_B_
      │    (mma_tensor_op_tile_iterator.h:2522)
      │    → XOR byte_offset_ 步进到 K 维度下一组
      │
      └─ warp_mma(accum, warp_frag_A, warp_frag_B, accum)
           │  (gemm/warp/mma_tensor_op.h:287)
           │
           └─ 双层循环：for m in [0, MmaIterations::kRow=4):
                          for n in [0, MmaIterations::kColumn=8):
                            mma(ptr_D[...], ptr_A[m], ptr_B[n], ptr_D[...])
                              (arch/mma_sm80.h:275)
                              → PTX: mma.sync.aligned.m16n8k16.row.col.f16
              共 4 × 8 = 32 次 mma.sync 指令
```

### Warp Tile Iterator 详解

**类型链路：**

```
DefaultMmaCore (default_mma_core_sm80.h:1487)
  └─ MmaTensorOp = DefaultMmaTensorOp<WarpShape, InstructionShape, ...>::Type
       (gemm/warp/default_mma_tensor_op.h:108)
       └─ MmaTensorOp<WarpShape=64x64x64, ...>
            (gemm/warp/mma_tensor_op.h:231)
            ├─ IteratorA = MmaTensorOpMultiplicandTileIterator<
            │      MatrixShape<64, 64>, kA, half, RowMajorCrosswise<16,64>,
            │      MatrixShape<16, 16>, ...>
            │      → 委托给 PitchLinear 版本 (mma_tensor_op_tile_iterator.h:2149)
            └─ IteratorB = MmaTensorOpMultiplicandTileIterator<
                   MatrixShape<64, 64>, kB, half, ColumnMajorCrosswise<16,64>,
                   MatrixShape<16, 8>, ...>
                   → 委托给 PitchLinear 版本 (mma_tensor_op_tile_iterator.h:2149)
```

**具体参数推导（fp16, kCrosswise=64）：**

底层都走到 `TensorOpMultiplicandCrosswise<16, 64>` 这个特化版本（line 2149）。关键参数：

```
Policy（决定 ldmatrix 的调用方式）：
  kLdsmOpOuter            = 8     （= kElementsPerAccess）
  kLdsmOpInner            = 8     （固定值）

  LdsmShapeContiguous     = InstructionShape::kContiguous / 8
                          = 16 / 8 = 2
  LdsmShapeStrided        = 4 / 2 = 2
  LdsmShape               = <2, 2>   → kCount = 4 → ldmatrix.x4

  LdsmIterations          = <1, Shape::kStrided / 8 / 2>
                          = <1, 4>   → 4 次 ldmatrix 调用

  kGroupsPerTile          = 8 / 1 / 2 = 4

Fragment（每个线程持有的数据量）：
  FragmentA = Array<half_t, 64 * 16 / 32> = Array<half_t, 32>  → 64 字节
  FragmentB = Array<half_t, 32>                                 → 同理
```

**load() 的完整流程：**

```cpp
// mma_tensor_op_tile_iterator.h:2598
void load_with_byte_offset(Fragment &frag, Index byte_offset) const {

    Array<unsigned, 4> *fetch_ptr = ...;   // 指向 Fragment 的 4-int 块

    for (int s = 0; s < 4; ++s) {          // LdsmIterations::kStrided = 4
        for (int c = 0; c < 1; ++c) {      // LdsmIterations::kContiguous = 1

            AccessType const *source_ptr =
                pointer_                                // 基地址（构造时根据 lane_id 计算）
                + 2 * c                                 // LdsmShape::kContiguous 偏移
                + 8 * 2 * s * stride_;                  // 沿 strided 方向步进

            ldsm<RowMajor, 4>(fetch_ptr[s], source_ptr + byte_offset_);
        }
    }
}
```

每次 `ldsm<RowMajor, 4>` 加载 4 × 32bit = 8 个 half 到当前线程的寄存器。4 次迭代加载 4 × 8 = 32 个 half，恰好填满 FragmentA。

**构造函数中的线程映射：**

构造函数（`mma_tensor_op_tile_iterator.h:2291`）根据 `lane_id` 计算每个线程的 smem 起始地址。对于我们的配置（kFactor=1, LdsmShape=<2,2>, Operand::kA）走的是 line 2418 分支：

```cpp
// "Matrix multiply 16816 A" 分支
// Q0 Q2
// Q1 Q3
partition_contiguous_idx = (lane_in_quad_pair >> 2);      // lane 0-3→0, 4-7→1
access_contiguous_idx   = (quad_quad ^ lane_in_quad);     // XOR 避免 bank conflict
access_strided_idx      = lane_in_quad_quad;              // 0-15
```

32 个线程被分为两个 quad-quad（各 16 线程），每组负责 smem tile 的不同区域。XOR 操作确保同一时刻不同线程读取不同的 bank。

**operator++()：K 维度上的步进：**

warp tile iterator 的 `++` 不是简单的地址递增，而是通过 **XOR byte_offset_** 来跳转（line 2522）：

```cpp
// 对于 kGroupsPerTile=4, kPartitionsK=1, mask=1:
if (((k_group_idx_ & mask) % 2) == 0)
    byte_offset_ ^= 1 * LdsmShape::kContiguous * ...;    // XOR 翻转
else if ((k_group_idx_ & mask) == 1)
    byte_offset_ ^= 3 * LdsmShape::kContiguous * ...;    // XOR 回原位

k_group_idx_++;
if (k_group_idx_ == 4) {  // 一个 tile 遍历完毕
    k_group_idx_ = 0;
    add_tile_offset({kGroupsPerTile, 0});  // 跳到下一个 tile
}
```

这种 XOR 步进与 smem 的 swizzle 布局配合：不需要重新计算地址，只需翻转 byte_offset_ 的特定 bit，就能跳到 K 维度上的下一组数据。每 4 次 `++` 走完一个 tile（kGroupsPerTile=4），然后跳到下一个 stage。

### Warp MMA operator 详解

`MmaTensorOp::operator()`（`gemm/warp/mma_tensor_op.h:287`）把 Fragment 切成多个 mma.sync 指令所需的小片段，双层循环遍历 warp tile：

```cpp
void operator()(FragmentC &D, FragmentA const &A, FragmentB const &B, FragmentC const &C) {
    D = C;
    MmaOperandA const *ptr_A = ...;   // FragmentA 切成 MmaIterations::kRow = 4 份
    MmaOperandB const *ptr_B = ...;   // FragmentB 切成 MmaIterations::kColumn = 8 份
    MmaOperandC *ptr_D = ...;         // Accumulator 切成 4 × 8 = 32 份

    for (int m = 0; m < 4; ++m) {              // WarpShape::kM / InstructionShape::kM = 64/16
        for (int n = 0; n < 8; ++n) {          // WarpShape::kN / InstructionShape::kN = 64/8
            int n_serpentine = (m % 2) ? (7 - n) : n;   // 蛇形遍历优化寄存器复用
            mma(ptr_D[m + n_serpentine * 4],
                ptr_A[m],                       // 8 个 half
                ptr_B[n_serpentine],             // 4 个 half
                ptr_D[m + n_serpentine * 4]);    // 4 个 half (累加)
        }
    }
}
```

**MmaIterations 的含义：**

```
WarpShape = 64 × 64 × 64
InstructionShape = 16 × 8 × 16

MmaIterations::kRow    = 64 / 16 = 4    （M 方向需要 4 次）
MmaIterations::kColumn = 64 / 8  = 8    （N 方向需要 8 次）

总计 4 × 8 = 32 次 mma.sync.m16n8k16
```

**蛇形遍历（serpentine）** 是一个优化：偶数行正向遍历 N（0→7），奇数行反向遍历 N（7→0）。这样相邻两次 MMA 共享 A operand（`ptr_A[m]` 不变），减少寄存器切换开销。



# 后记

## 小结

TODO

# Reference

- [CUTLASS Convolution — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/implicit_gemm_convolution.html)
- [CUTLASS GEMM API — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html#cutlass-gemm-api)
- [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)