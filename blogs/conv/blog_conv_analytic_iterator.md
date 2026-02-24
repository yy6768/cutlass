# CUTLASS 学习记 (3) —— 数据加载流 （Kernel层/ThreadBlock层 iterator部分）

# 前言

- 本篇发表在丙午年的农历春节大年初三，祝大家新年快乐；

- 为了更好的叙述整个系列和利用示例代码（实际上是不想自己抄到自己写的部分），读者根据claude的建议将整体的叙述顺序从自下而上更改为自上而下；**本篇将从CUDA Debug的原点入手，重点剖析 convolution的 analytic iterator的数据映射**；
- 撰写本篇时，笔者尝试使用**claude code opus 4.6 + cuda skill的方式进行辅助编写**，所以部分叙述ai味将比前几篇重。但为了适应新时代的ai workflow，笔者觉得这是值得尝试的一环。如果大家有比较好的使用方式也可以在评论区交流。



# 数据读取/映射

## 回忆Implicit GEMM

回忆第一篇的介绍：

- Implicit GEMM 的核心思想是将这个卷积的数据坐标映射成一个 GEMM 问题，而非创建新的空间展开成卷积：

```
GEMM: C = A × B

其中：
  A (Activation) 的逻辑形状为 [NPQ, CRS]
  B (Filter)     的逻辑形状为 [CRS, K]
  C (Output)     的逻辑形状为 [NPQ, K]
```

这里 NPQ = N × P × Q 是输出张量的空间维度展平，CRS = C × R × S 是 filter 的通道和空间维度展平。

但与标准 GEMM 不同的是，矩阵 A 并不真正存在于内存中——它是从输入张量 `Input(N, H, W, C)` 通过坐标映射"虚拟"构造出来的。这就是 iterator 需要解决的核心问题。



## Implicit GEMM 多级划分

Nvidia 官方博客这张图非常直观的展现一个典型的案例（C=128， R=S=3）

- **Kernel层**(implicit_gemm_convolution.h) 会负责GEMM的K维度上的划分，每个block理论上负责一个tile（比如M × N × K = 128 x 128 x 64, 超参数决定）。 每个Activation Tile和Filter tile都相互对应；每个 block 内部沿 CRS 维度串行迭代，迭代次数（gemm_k_iterations）的计算由Kernel 层级的超参数决定（后面代码层面分析）；
- **ThreadBlock层** 会有专门的iterator负责加载和映射对应的数据。
  - `at`函数：在指定的thread block上，根据GEMM的行和列，逆向计算原始需要加载的Activation和Filter的坐标；
  - `operator ++` 操作符: ++被用于一个线程内的数据偏移迭代，这些内存访问操作发生在两个维度上：连续的（contiguous）维度和跨步的（strided）维度。
    - contiguous是C通道方向上的偏移，由于每个线程只需要访问输入通道一次，所以iteration_contiguous_永远为0。
    - strided 维度对应 tile 的行方向（NPQ/输出位置方向）。ThreadMap::Iterations::kStrided
        表示每个线程在这个方向上需要访问多少行。因为一个 tile 有 128 行（M=128），而 threadblock
        里有若干线程分摊，每个线程可能需要负责多行，所以需要迭代 kStrided 次。

  - `advance`函数：advance函数在每一次Tile结束访问之后被调用，所有指针偏移会指向下一个tile。官方的实现是按照SRC的顺序（C通道是按照32对齐）




![ALT](https://docs.nvidia.com/cutlass/latest/_images/conv2d-fprop-int4.png)



## Claude：一图总结映射关系

```
GEMM 视角:          卷积视角:              内存访问:

A[NPQ, CRS]  →  Input[N, H, W, C]  →  pointer + layout(n, h, w, c)
  │  行idx          ↑                      ↑
  │  ──────→  (n,p,q) ──┐                  │
  │  列idx            (r,s,c)              │
  │  ──────→          ──┘                  │
  │               h = p*stride - pad + r*dilation
  │               w = q*stride - pad + s*dilation
  │                                        │
B[CRS, K]    →  Filter[K, R, S, C]  →  pointer + layout(k, r, s, c)
                  (直接索引，无需变换)
```



# 回到代码

## Kernel层 : 关于之前调试问题的补充

前情提要：

- 在更换了一个本地环境后，我发现自己在开启Nsight Debug插件后没办法进入CUTLASS 的Kernel了；

- 另外一个群友在学习cutlass的时候也遇到了类似的问题（他在WSL上使用cuda gdb也无法进入Kernel Launch的点）

后续发现在之前的环境里（当时配环境用了cursor没留意）需要在编译的时候添加cuda的Debug信息flag。

根据[nvcc文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html?highlight=lineinfo#device-debug-g) 需要添加device debug flags和行号信息才能进入具体的kernel

在根目录下的`CMakeLists.txt`添加-G和 -lineinfo：

```cmake
# Cache the flags so they are available when the function below is called anywhere globally.

# Enable device-level debugging for Debug builds (required by Nsight CUDA Debugger)
list(APPEND CUTLASS_CUDA_NVCC_FLAGS_DEBUG -G -lineinfo)
```



添加完之后使用nsight debug就可以进入kernel的断点了。

![进入调试](https://typora-yy.oss-cn-hangzhou.aliyuncs.com/Typora-img/image-20260217142512075.png)



## Kernel层:Implicit GEMM的问题划分

claude生成的从 kernel 入口到 iterator 的完整调用链：

```
1. ImplicitGemmConvolution::operator()
   │  (conv/kernel/implicit_gemm_convolution.h:281)
   │
   ├─ 构造 iterator_A (ActivationTileAccessIteratorAnalytic)
   │    (conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h:137)
   │    → 分解 threadblock_offset 为 (n, p, q)
   │    → 初始化 filter_r_ = 0, filter_s_ = 0, filter_c_ = threadblock_offset.column()
   │
   ├─ 构造 iterator_B (FilterTileAccessIteratorAnalytic)
   │    (conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h:137)
   │    → 记录 offset_k_[] 和 filter_c_
   │
   └─ 调用 mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, ...)

```

我们逐行打上断点，查看一下核心的逻辑。

-----------------

step 1:**Calculate threadblock_tile_idx**

这里只需要了解到**只有当 K 较大（比如 K=512，N 方向有 4 个 tile）时 swizzle 才会生效**。 其余时间blockIdx.x = M, blockIdx.y = N

```c++
 // Compute threadblock location
 ThreadblockSwizzle threadblock_swizzle;

 cutlass::gemm::GemmCoord threadblock_tile_idx =
     threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
```

swizzle我们后续会介绍，这里的CRS也比较小，不会触发swizzle；

----



step 2：**计算threadBlock的position**

Mma会提供超参数， 只需要拿到tile_idx.k, 计算列偏移即可（我们不涉及group_conv和channel-wise conv）

```c++
 int iterator_A_column_offset = threadblock_tile_idx.k() * Mma::Shape::kK;
```



step 3：**创建iterator A和B**

为了探索iterator A和B的映射计算，所以我们需要把optimized iterator换成analytic iterator。

在 `ampere_tensorop_conv2dfprop.cu`将 第 266 行改为：

```cpp
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kAnalytic;
```

然后我们把断点打在

conv2d_fprop_activation_tile_access_iterator_analytic.h的137行。

```C++
CUTLASS_HOST_DEVICE
Conv2dFpropActivationTileAccessIteratorAnalytic(
  Params const &params, 
  ...       
):
  params_(params), 
  ....
{
  // 计算thread block的起始偏移
  layout::PitchLinearCoord thread_coord = ThreadMap::initial_offset(thread_idx);

  // filter channel 映射偏移
  filter_c_ = threadblock_offset.column() + thread_coord.contiguous();

  // Group / Channel-wise
  if (kGroupMode != conv::GroupMode::kNone) {
   ...
  }
  // 计算npq的偏移
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
    ....
  }
}
```

这个函数主要有3个功能：

- ThreadMap::initial_offset(thread_idx)   计算当前线程在 tile 内的起始位置，返回一个二维坐标 (contiguous, strided)， 这里的contiguous是channel 通道一定offset是1， strided是NPQ方向的偏移；
- **filter_c**初始化：threadblock_offset 是当前thread_block的偏移，加上thread_coord当前的偏移
- **NPQ 坐标分解循环**
  - kStride会根据当前ThreadBlock对应的Tile长度（现在超参定了是64）和Delta::kStride（kWarpCount * WarpThreadArrangement::kStrided =
   8 * 2 = 16）计算出64/ 16 = 4
  - 根据s * ThreadMap::Delta::kStrided 计算 第 s 次迭代的步进（npq方向） offset_npq
  - n = offset_npq / (P * Q)
  - p = (offset_npq % (P * Q)) / Q
  - q = offset_npq % Q



对于B iterator，代码结构也是类似：

- init thread block offset (一致)

- **filter_c**初始化：注意Filter矩阵是[CRS, K] filter_c_ = threadblock_offset.row() + ...（行方向 = CRS）

- offset_k_[] 循环：（K只有一个维度，不需要折叠）

```
offset_k_[s] = threadblock_offset.column() + thread_coord.strided() + s * ThreadMap::Delta::kStrided;    
```



## Thread Block层: 关键调用栈

claude生成的核心调用栈，本期只关注到Main loop的数据加载部分。warp_mma的数据这里标出，但是下一期再看

```
2. ImplicitGemmMultistage::operator()
   │  (conv/threadblock/implicit_gemm_multistage.h:266)
   │
   ├─ Prologue: 预加载前 kStages-1 个 tile 到 shared memory
   │   │  (implicit_gemm_multistage.h:289)
   │   ├─ 对每个 stage：
   │   │   ├─ 循环调用 ++iterator_A / ++iterator_B  → 遍历 tile 内的所有访问位置
   │   │   │   (implicit_gemm_multistage.h:312 / 338)
   │   │   │   内部调用：
   │   │   │     iterator_A.get()   → at() 计算坐标 → 返回 global memory 指针
   │   │   │     iterator_A.valid() → 判断是否越界（padding 区域）
   │   │   │     cp_async_zfill()   → 异步复制到 smem，越界则填零
   │   │   │     ++smem_iterator_A_ → smem 写指针推进
   │   │   └─ iterator_A.advance() / iterator_B.advance()
   │   │       → 推进到下一个 (r,s,c) tile
   │   │       (activation_tile_access_iterator_analytic.h:198)
   │   │       (filter_tile_access_iterator_analytic.h:198)
   │   └─ cp_async_fence() → 给本 stage 的所有 cp.async 打标记
   │
   └─ Mainloop: for 每个 gemm_k_iteration
         │
         └─ for warp_mma_k in [0, kWarpGemmIterations):
               ├─ warp_tile_iterator.load()
               ├─ copy_tiles_and_advance()
               │   (implicit_gemm_multistage.h:202)
               │   → 发射下一个 tile 的「一段」cp.async（与 MMA 交叠）
               │   → 内部同样调用 get() / valid() / cp_async_zfill() / ++iterator
               │   注意：advance() 不在这里调用
               ├─ warp_mma()
               │   → 用寄存器中的数据执行 Tensor Core MMA
               │
               └─ 仅在 warp_mma_k + 2 == kWarpGemmIterations 时：
                   ├─ cp_async_fence()
                   ├─ cp_async_wait<kStages-2>()  → 流水线同步
                   ├─ iterator_A.advance()         → 切换到下一个 (r,s,c)
                   │   (activation_tile_access_iterator_analytic.h:198)
                   └─ iterator_B.advance()
                       (filter_tile_access_iterator_analytic.h:198)
```

----

## Thread Block层: Prologue

在初始化iteratorA和B之后很快就进入到ThreadBlock的mma operator（Convolution对应的是implicit_gemm_multistage）， 对于threadblock mma，按照cutlass的设计都是 prologue -> MainLoop -> Epilogue

为了一探究竟， 首先断点打到之前分析的 conv/threadblock/implicit_gemm_multistage.h:312

首先看调用栈： operator++在312行被multistage调用，这里实际上是 停在了Prologue阶段

![Iterator调用栈](https://typora-yy.oss-cn-hangzhou.aliyuncs.com/Typora-img/image-20260218180212621.png)

**Q:**  implicit_gemm_multistage 中 Prologue做了什么？

**Claude：** Prologue 的整体目标：提前把前 kStages - 1 个 tile 加载进 shared memory，供 mainloop 流水使用。

-  第一层循环（stage 循环）：每次循环加载一个完整的 CRS tile（一个 advance() 对应的数据），同时把 gemm_k_iterations 减 1——表示这个 tile已经"消耗"掉，mainloop 少跑一次。
- 第二层循环（AsyncCopyIterationsPerStageA）：AsyncCopyIterationsPerStageA 表示一个 stage 内需要多少次 smem 写入。smem_iterator_A_ 负责管理 shared memory的写入位置，每次 ++ 推进到 smem 里的下一个 slot。
- 第三层循环（kAccessesPerVector）+ cp_async_zfill：cp_async_zfill 是核心：它对 Ampere 硬件指令 cp.async 的封装。它做两件事：
  1. valid() == true：从 global memory 异步复制 kSrcBytes 字节到 shared memory，不阻塞当前线程（硬件 DMA 搬运）   
  2. valid() == false：向 shared memory 写入零（zero-fill），这就是隐式 padding的实现——越界的位置自动填零，不需要任何 if 判断

假设kStages=3整体流程图如下：

```
  Prologue:
    stage=0: 加载 CRS tile[0] → smem slot[0]  → fence[0]
             advance() → 指向 CRS tile[1]
    stage=1: 加载 CRS tile[1] → smem slot[1]  → fence[1]
             advance() → 指向 CRS tile[2]

  cp_async_wait<1>()    ← 等待至少 fence[0] 完成
```



Q: **cp.async 是什么？**
Claude: Ampere (SM80) 引入了 `cp.async` 指令，允许线程发起一次从 Global Memory 到 Shared Memory 的**异步拷贝**。与传统的 "global → register → shared" 两步搬运不同，cp.async 由硬件 DMA 完成，**不占用寄存器，也不阻塞当前线程的计算指令**。

CUTLASS 在 `include/cutlass/arch/memory_sm80.h` 中对这些 PTX 指令做了封装：

**1. cp_async_zfill —— 带零填充的异步拷贝**（memory_sm80.h:151）

```cpp
// 核心 PTX（以 CacheOperation::Always, 16字节为例）：
asm volatile(
    "cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
    :: "r"(smem_int_ptr),   // shared memory 目标地址
       "l"(global_ptr),      // global memory 源地址
       "n"(SizeInBytes),     // 拷贝字节数（编译期常量：4/8/16）
       "r"(src_in_bytes));   // 实际拷贝字节数（运行时）
```

零填充的秘密在第 4 个参数 `src_in_bytes`：
- `pred_guard == true` 时，`src_in_bytes = SizeInBytes`（正常拷贝 16 字节）
- `pred_guard == false` 时，`src_in_bytes = 0`（拷贝 0 字节，硬件自动将 smem 目标区域填零）

这就是隐式 padding 的实现——不需要 if/else 分支，一条指令同时处理有效数据和越界填零。

**2. cp_async_fence —— 异步拷贝分组标记**（memory_sm80.h:440）

```cpp
asm volatile("cp.async.commit_group;\n" ::);
```

`commit_group` 把之前发射的所有 cp.async 指令打包成一个 **group**。每调用一次 fence，就创建一个新的 group。这使得后续可以按 group 粒度等待完成。

**3. cp_async_wait\<N\> —— 等待异步拷贝完成**（memory_sm80.h:452）

```cpp
asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
// N=0 时特化为：
asm volatile("cp.async.wait_all;\n" ::);
```

`wait_group N` 的语义：等待直到**最多还有 N 个 group 未完成**。例如：
- `cp_async_wait<0>()`：等待所有 group 完成
- `cp_async_wait<1>()`：允许最新的 1 个 group 还在飞行中，但更早的必须完成

在 kStages=3 的流水线中，`cp_async_wait<kStages-2>()` 即 `wait<1>`：确保最老的那个 stage 数据已经到位，可以被 MMA 消费，同时最新发射的那个 stage 还在异步搬运中——这就是流水线的核心。

---

额……我相信claude讲的已经够好了（虽然我还是一头雾水），因为本篇主要还是讲数据加载，multistage以及shared memory我们后续再探究。

F11跳进iteratorA++函数

**operator ++ **

两个iterator的**operator++**函数都相似，大概都符合下面的伪代码。也就是只是推移offset

```cpp
 每次 ++：
    vector++
    如果 vector 没溢出 → 结束
    否则 vector 归零，contiguous++
      如果 contiguous 没溢出 → 结束
      否则 contiguous 归零，strided++
        如果 strided 没溢出 → 结束
        否则 strided 归零（整个 tile 遍历完毕）
```

  对于 Activation iterator（ThreadMap Shape <128, 64>）：
  - kAccessesPerVector = 1（128-bit 访问 cp.async，8 个 half，恰好等于 AccessType 大小）
  - Iterations::kContiguous = 1（static_assert 强制为 1） 
  - Iterations::kStrided = 4

因为前两个值计算出来是1，所以实际上递增只有kStride会增加；

filter iterator的 operator ++ 也类似；

-----

**advance()**函数

 `advance()` 方法逐步遍历 filter 的空间维度：

```cpp
void advance() {
    ++filter_s_;                          // 先遍历 S（filter 宽度）
    if (filter_s_ < problem_size_.S) return;
    filter_s_ = 0;

    ++filter_r_;                          // 再遍历 R（filter 高度）
    if (filter_r_ < problem_size_.R) return;
    filter_r_ = 0;

    filter_c_ += Shape::kColumn * split_k_slices;  // 最后推进 C（通道），步长为 tile 宽度
}
```

遍历顺序是 S → R → C。每次 `advance()` 对应 GEMM 主循环的一次迭代，沿 K 维度前进一个 tile。

---

## Thread Block层: Mainloop

接下来看Mainloop， 主循环会多次调用iterator推进。

断点打到 `implicit_gemm_multistage.h:447`（copy_tiles_and_advance 调用处）。

**Q：Mainloop 的copy_tiles_and_advance 和 Prologue 的数据处理最大的区别是什么？**

**Claude：** Prologue 里 iterator 每加载完一个完整 tile 就立刻 advance；Mainloop 里为了让 MMA 计算和 global→smem 的数据搬运完全重叠，把一个 tile 的 cp.async 拆成了 `kWarpGemmIterations` 份，每次 MMA 迭代只发射其中一份，等到 `warp_mma_k` 接近末尾时才统一 advance。

也就是说：**Mainloop 的 `copy_tiles_and_advance` 不是在搬"当前" tile，而是在提前搬"下一个" tile 的一部分**。

**Q：warp_mma_k + 2 == kWarpGemmIterations 这个条件块的作用是？**

**Claude：** 在 `warp_mma_k + 2 == kWarpGemmIterations` 的条件块里（`implicit_gemm_multistage.h:477`）：

```cpp
if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
    cutlass::arch::cp_async_fence();          // 给本轮所有 cp.async 打标记
    arch::cp_async_wait<Base::kStages - 2>(); // 等待流水线
    __syncthreads();

    iterator_A.advance();   // ← 这里才真正切换到下一个 (r,s,c)
    iterator_B.advance();
    ...
}
```

注意是倒数第二次 `warp_mma_k` 迭代，不是最后一次。这样 advance 之后紧接着最后一次 `warp_mma_k` 还能继续发射下一轮 tile 的第一批 cp.async，流水不断。

**copy_tiles_and_advance 内部：**

```cpp
void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                             int group_start_A = 0, int group_start_B = 0) {

    // 从 group_start 对应的位置开始（不是从头开始）
    iterator_A.set_iteration_index(group_start_A * kAccessesPerVector);
    smem_iterator_A_.set_iteration_index(group_start_A);

    for (int j = 0; j < kAccessesPerGroupA; ++j) {
        if (group_start_A + j < AsyncCopyIterationsPerStageA) {
            dst_ptr = smem_iterator_A_.get();   // smem 写入目标

            for (int v = 0; v < kAccessesPerVector; ++v) {
                cp_async_zfill(dst_ptr + v,
                               iterator_A.get(),    // at() → 坐标 → 指针
                               iterator_A.valid()); // 越界则填零
                ++iterator_A;
            }
            ++smem_iterator_A_;
        }
    }
    // iterator_B 同理 ...
}
```

`group_start` 决定从 tile 的哪个位置开始发射。`warp_mma_k = 0` 时 `group_start = kAccessesPerGroupA`（跳过第 0 组，因为第 0 组在上一轮末尾已经发射了），最后一次 `warp_mma_k` 时 `group_start = 0`（从头开始发射下一个 tile）。

---

接下来是`iterator_A.get()`调用的三个函数 

**`at()` —— 逆向计算卷积坐标**（`activation_tile_access_iterator_analytic.h:232`）

```cpp
TensorCoord at() const {
    // 1. 从预计算数组里取出当前 strided 迭代对应的输出坐标
    int n = offset_n_[iteration_strided_];
    int p = offset_p_[iteration_strided_];
    int q = offset_q_[iteration_strided_];

    // 2. 取当前 filter 位置（由 advance() 维护）
    int r = filter_r_;
    int s = filter_s_;

    // 3. 如果是 true convolution反转坐标，默认这个 if 不会进入
    if (problem_size_.mode == Mode::kConvolution) {
        r = (problem_size_.R - 1 - filter_r_);
        s = (problem_size_.S - 1 - filter_s_);
    }

    // 4. 核心公式：从输出坐标 (p,q) 和 filter 位置 (r,s) 逆推输入坐标 (h,w)
    int h = p * stride_h - pad_h + r * dilation_h;
    int w = q * stride_w - pad_w + s * dilation_w;

    // 5. 通道：filter_c_ 是当前 C tile 的起始，加上 vector 内偏移
    int c = filter_c_ + iteration_vector_ * AccessType::kElements;

    return TensorCoord(n, h, w, c);
}
```

这里第 4 步就是 Implicit GEMM 的灵魂所在。正向卷积的定义是：

```
Output(n, p, q, k) = Σ Input(n, h, w, c) * Filter(k, r, s, c)
其中 h = p * stride - pad + r * dilation
     w = q * stride - pad + s * dilation
```

`at()` 做的就是把这个关系反过来用：**已知当前要计算的输出位置 (p, q) 和当前 filter 核位置 (r, s)，求需要加载的输入位置 (h, w)**。不需要构造任何中间矩阵，直接一行公式搞定。

**`valid()` —— 判断是否是 padding 区域**（`activation_tile_access_iterator_analytic.h:255`）

```cpp
bool valid() const {
    TensorCoord coord = at();
    return coord.n() < problem_size_.N &&
           coord.h() >= 0 && coord.h() < problem_size_.H &&
           coord.w() >= 0 && coord.w() < problem_size_.W &&
           coord.c() < problem_size_.C;
}
```

当 `h < 0` 或 `h >= H`（或 w 越界）时，说明这个位置对应的是 padding——物理上不存在，应该当作 0 处理。`valid()` 返回 false，`cp_async_zfill` 就会在 smem 对应位置写零，而不是去读非法内存。

**`get()` —— 把坐标转换成内存地址**（`activation_tile_access_iterator_analytic.h:267`）

```cpp
AccessType const *get() const {
    TensorCoord coord = at();
    LongIndex offset = params_.layout(coord);   // NHWC: n*H*W*C + h*W*C + w*C + c
    return reinterpret_cast<AccessType const *>(
        pointer_ + offset * sizeof_bits<Element>::value / 8);
}
```

`params_.layout(coord)` 是 NHWC 布局的线性化公式，把 `(n, h, w, c)` 转成相对于 tensor 基地址的字节偏移。`pointer_` 是构造函数里传进来的 global memory 基地址，两者相加就是最终要 `cp.async` 的源地址。

**claude：一次 cp_async_zfill 的完整流程**

```
++iterator_A 推进 iteration_strided_ / iteration_vector_
    ↓
iterator_A.get()
    ↓ 调用 at()
    (p, q, r, s, c) → h = p*stride - pad + r*dilation
                     → w = q*stride - pad + s*dilation
                     → 返回 TensorCoord(n, h, w, c)
    ↓ 调用 layout(coord)
    → 线性字节偏移 → pointer_ + offset → global memory 源地址
    ↓
iterator_A.valid() → h/w 是否在 [0, H) / [0, W) 内
    ↓
cp_async_zfill(smem_dst, global_src, is_valid)
    → is_valid=true:  硬件异步搬运 16 字节 global → smem
    → is_valid=false: smem 对应位置填零（隐式 padding）
```

filter iterator 的 `at()` 就简单得多，直接返回 `TensorCoord(k, filter_r_, filter_s_, c)`，不需要任何坐标变换，因为 filter 本身就按 KRSC 存储的。

**整体时序（kStages=3, kWarpGemmIterations=9 为例）：**

```
 完整时序（kStages=3，共 9 个 tile）

  smem 有 3 个槽位：slot[0], slot[1], slot[2]（循环复用）

  Prologue:
    tile[0] → cp.async → slot[0]    fence
    tile[1] → cp.async → slot[1]    fence
    wait slot[0] 完成
    （此时 slot[0] 的数据可用，slot[1] 正在搬运中）

  Mainloop（共 9-2=7 次迭代）:
    iter 0: MMA 消费 slot[0] (tile[0])  ← 计算
            同时 cp.async tile[2] → slot[2]  ← 搬运下一个

    iter 1: MMA 消费 slot[1] (tile[1])
            同时 cp.async tile[3] → slot[0]  ← slot[0] 已被消费完，可以复用

    iter 2: MMA 消费 slot[2] (tile[2])
            同时 cp.async tile[4] → slot[1]

    iter 3: MMA 消费 slot[0] (tile[3])
            同时 cp.async tile[5] → slot[2]

    ...以此类推...

    iter 6: MMA 消费 slot[0] (tile[6])
            同时 cp.async tile[8] → slot[2]

    最后两次不再发射新的 cp.async，只做 MMA：
    iter 7: MMA 消费 slot[1] (tile[7])
    iter 8: MMA 消费 slot[2] (tile[8])
```

好的到这里，所有卷积数据映射到GEMM排布的相关代码解析完了。

# 后记

## Claude code: Analytic vs Optimized

CUTLASS 提供了两种 iterator 实现：

| 特性     | Analytic                    | Optimized                  |
| -------- | --------------------------- | -------------------------- |
| 坐标计算 | 每次访问实时计算 (n,h,w,c)  | 预计算指针偏移，用增量推进 |
| 边界检查 | 计算坐标后逐维度检查        | 预计算 bitmask，位运算判断 |
| 适用范围 | 任意 R, S, stride, dilation | R ≤ 32, S ≤ 32             |
| 性能     | 较低（更多算术指令）        | 较高（指针算术 + bitmask） |
| 可读性   | 高，逻辑清晰                | 低，大量预计算和位操作     |

---

## 小结

本篇主要解析了 CUTLASS 卷积中 analytic iterator 的工作原理（包括Kernel层和Thread Block的Prologue和部分MainLoop）。核心要点：

- 根据复习了Implicit GEMM算法，了解了相关的数据映射；
- Kernel层的调试问题
- Kernel层：Iterator的初始化，重点是offset相关的计算
- Thread Block层 Prologue层（加载kStage -1组数据进入smem slot）
- Thread Block层 Mainloop层（除了最后两个stage，每次计算时MMA消费一个slot数据，并填充一个slot的新数据）

下一篇我们将深入 shared memory 和warp 级别的 MMA（Matrix Multiply-Accumulate），看数据如何shared memory 的数据如何组织并传递给寄存器兵最终喂给Tensor Core。

# Reference

- [CUTLASS Convolution — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/media/docs/cpp/implicit_gemm_convolution.html)
- [IDE Setup for CUTLASS Development — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/ide_setup.html)
- [CUDA Compiler Driver NVCC — device-debug](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html?highlight=lineinfo#device-debug-g)
- [1. Introduction — PTX ISA 9.1 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- CIS 5650 CUDA Debugging Lab.mp4 - Google Drive （Upenn CIS 5650课程讲述了如何使用VS Nsight扩展调试CUDA程序，推荐入门者观看）