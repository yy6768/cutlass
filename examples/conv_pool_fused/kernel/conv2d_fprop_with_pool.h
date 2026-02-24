/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Implicit GEMM convolution kernel with fused MaxPool2x2.

  Extends ImplicitGemmConvolution with:
    - Arguments: replaces ref_D with float *pooled_output_ptr
    - Params:    builds iterator_D via Params(problem_size, pooled_ptr),
                 so pool coordinates are baked into the iterator at launch time
    - operator(): identical main loop; epilogue dispatches atomicMax via the
                  custom OutputTileIterator (PredicatedTileIteratorPooledOutput)

  Constraints:
    - split_k_slices must be 1 (split-K is incompatible with atomicMax pooling)
    - The caller must zero-initialize pooled_output_ptr before launch
      (atomicMax(0, x) = x for non-negative x, matching ReLU output)
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                                  ///< Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,                             ///< Epilogue (uses PredicatedTileIteratorPooledOutput)
  typename ThreadblockSwizzle_,                   ///< Threadblock swizzling function
  conv::Operator ConvOperator,                    ///< Must be kFprop
  typename ConvProblemSize_ = Conv2dProblemSize,
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone
>
struct ImplicitGemmConvolutionWithPool {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA  = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB  = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueOutputOp::ElementOutput;
  using LayoutC  = LayoutA;

  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ElementCompute     = typename EpilogueOutputOp::ElementCompute;

  using WarpMmaOperator  = typename Mma::Policy::Operator;
  using ArchMmaOperator  = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator     = typename ArchMmaOperator::Operator;
  using OperatorClass    = typename WarpMmaOperator::OperatorClass;
  using ArchTag          = typename WarpMmaOperator::ArchTag;

  using ThreadblockShape = typename Mma::Shape;
  using WarpShape        = typename WarpMmaOperator::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static int const kStages = Mma::kStages;
  static IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm;
  static StrideSupport const kStrideSupport = Mma::IteratorA::kStrideSupport;

  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

  static_assert(Mma::IteratorA::kConvDim == Mma::IteratorB::kConvDim,
    "Convolution on different dimensions is not supported");
  static int const kConvDim = Mma::IteratorA::kConvDim;

  using ConvProblemSize = ConvProblemSize_;
  static conv::GroupMode const kGroupMode = GroupMode_;

  static int const kWgradCStrideIdx =
    platform::is_same<LayoutC, cutlass::layout::TensorNHWC>::value ? 2 : 3;
  static int const kTensorCStrideIdx =
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradCStrideIdx : 0);

  using ConvOutputIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,
    typename Epilogue::OutputTileIterator::Layout,
    TensorRefC,
    ConvOperator,
    ConvProblemSize>;

  //
  // Arguments — replaces ref_D with a raw pooled output pointer (FP32)
  //

  struct Arguments {

    ConvProblemSize problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefC ref_C;               ///< Source tensor (unused when beta=0)
    float *pooled_output_ptr;       ///< FP32 pooled output; must be zero-initialized before launch
    typename EpilogueOutputOp::Params output_op;
    SplitKMode split_k_mode;        ///< Must be kSerial with split_k_slices == 1

    CUTLASS_HOST_DEVICE
    Arguments() {}

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const &problem_size_,
      TensorRefA const &ref_A_,
      TensorRefB const &ref_B_,
      TensorRefC const &ref_C_,
      float *pooled_output_ptr_,
      typename EpilogueOutputOp::Params const &output_op_,
      SplitKMode split_k_mode_ = SplitKMode::kSerial
    ) :
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      pooled_output_ptr(pooled_output_ptr_),
      output_op(output_op_),
      split_k_mode(split_k_mode_)
    {}
  };

  //
  // Params — iterator_D is built via Params(problem_size, pooled_ptr)
  //

  struct Params {
    ConvProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    gemm::GemmCoord implicit_gemm_problem_size;
    int swizzle_log_tile;

    int gemm_k_iterations;
    int gemm_k_iterations_per_channel;

    typename Mma::IteratorA::Params iterator_A;
    typename Mma::IteratorA::Element const *ptr_A;

    typename Mma::IteratorB::Params iterator_B;
    typename Mma::IteratorB::Element const *ptr_B;

    /// iterator_C params: compatibility constructor (load() returns zeros with beta=0)
    typename Epilogue::OutputTileIterator::Params iterator_C;
    typename Epilogue::OutputTileIterator::Element *ptr_C;

    /// iterator_D params: pool-aware constructor stores N/P/Q/K + pooled_ptr
    typename Epilogue::OutputTileIterator::Params iterator_D;
    // No ptr_D — pool addresses are computed from coordinates inside the iterator

    typename EpilogueOutputOp::Params output_op;
    int *semaphore;
    SplitKMode split_k_mode;

    CUTLASS_HOST_DEVICE
    Params() : swizzle_log_tile(0), gemm_k_iterations(0) {}

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore_ = nullptr
    ) :
      problem_size(args.problem_size),
      implicit_gemm_problem_size(
        cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
      iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
      ptr_A(args.ref_A.data()),
      iterator_B(args.problem_size, args.ref_B.layout()),
      ptr_B(args.ref_B.data()),
      // iterator_C: use Layout-based compatibility constructor (zeros on load)
      iterator_C(
        ConvOutputIteratorParameter::layout(args.ref_C),
        implicit_gemm_tensor_c_extent(kConvolutionalOperator, args.problem_size)),
      ptr_C(args.ref_C.data()),
      // iterator_D: pool-aware — bakes N/P/Q/K and pooled_ptr into Params
      iterator_D(args.problem_size, args.pooled_output_ptr),
      output_op(args.output_op),
      semaphore(semaphore_),
      split_k_mode(args.split_k_mode)
    {
      gemm_k_iterations = implicit_gemm_k_iterations(
        kConvolutionalOperator,
        ThreadblockShape::kK,
        args.problem_size,
        kIteratorAlgorithm,
        kGroupMode,
        ThreadblockShape::kN);

      gemm_k_iterations_per_channel = implicit_gemm_k_iterations_per_channel(
        kConvolutionalOperator, args.problem_size, kIteratorAlgorithm);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_HOST_DEVICE
  ImplicitGemmConvolutionWithPool() {}

  /// Executes one fused Conv2dFprop + MaxPool2x2
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
      threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {
      return;
    }

    int thread_idx = threadIdx.x;
    int iterator_A_column_offset = threadblock_tile_idx.k() * Mma::Shape::kK;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.iterator_A,
      params.problem_size,
      params.ptr_A,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        iterator_A_column_offset));

    typename Mma::IteratorB iterator_B(
      params.iterator_B,
      params.problem_size,
      params.ptr_B,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN));

    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;
    accumulators.clear();

    mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators,
        params.gemm_k_iterations_per_channel);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * Mma::Shape::kM,
      threadblock_tile_idx.n() * Mma::Shape::kN);

    // Pooled output iterator: pointer arg is ignored; pool params come from iterator_D
    typename Epilogue::OutputTileIterator iterator_D(
      params.iterator_D,
      static_cast<typename Epilogue::OutputTileIterator::Element *>(nullptr),
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset);

    // Source iterator: with beta=0 + LinearCombinationRelu, load() returns zeros
    typename Epilogue::OutputTileIterator iterator_C(
      params.iterator_C,
      params.ptr_C,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset);

    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

    // Run epilogue — stores via atomicMax to pooled buffer
    epilogue(output_op, iterator_D, accumulators, iterator_C);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
