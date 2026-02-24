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
  \brief Device-level interface for Conv2dFprop + fused MaxPool2x2.

  Wraps ImplicitGemmConvolutionWithPool (the kernel struct) in the same
  host-side API pattern as cutlass::conv::device::ImplicitGemmConvolution.

  Usage:
    Conv2dFpropWithPool<...> op;
    op.initialize(args, nullptr, stream);   // or op(args, nullptr, stream)
    op.run(stream);

  Caller responsibilities:
    1. Zero-initialize args.pooled_output_ptr before every call
       (atomicMax needs a floor value of 0 since ReLU output >= 0)
    2. args.problem_size.split_k_slices must be 1
*/

#pragma once

#include <limits>
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Conv2dFpropWithPoolKernel_>
class Conv2dFpropWithPool {
public:

  using UnderlyingKernel = Conv2dFpropWithPoolKernel_;

  using ElementA            = typename UnderlyingKernel::ElementA;
  using LayoutA             = typename UnderlyingKernel::LayoutA;
  using ElementB            = typename UnderlyingKernel::ElementB;
  using LayoutB             = typename UnderlyingKernel::LayoutB;
  using ElementC            = typename UnderlyingKernel::ElementC;
  using LayoutC             = typename UnderlyingKernel::LayoutC;
  using ElementAccumulator  = typename UnderlyingKernel::ElementAccumulator;
  using ElementCompute      = typename UnderlyingKernel::ElementCompute;
  using OperatorClass       = typename UnderlyingKernel::OperatorClass;
  using ArchTag             = typename UnderlyingKernel::ArchTag;
  using ThreadblockShape    = typename UnderlyingKernel::ThreadblockShape;
  using WarpShape           = typename UnderlyingKernel::WarpShape;
  using InstructionShape    = typename UnderlyingKernel::InstructionShape;
  using ThreadblockSwizzle  = typename UnderlyingKernel::ThreadblockSwizzle;
  using EpilogueOutputOp    = typename UnderlyingKernel::EpilogueOutputOp;

  static int const kStages = UnderlyingKernel::kStages;
  static int const kConvDim = UnderlyingKernel::kConvDim;
  static cutlass::conv::Operator const kConvolutionalOperator =
    UnderlyingKernel::kConvolutionalOperator;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm =
    UnderlyingKernel::kIteratorAlgorithm;

  static int const kWarpCount =
    (ThreadblockShape::kM / WarpShape::kM) *
    (ThreadblockShape::kN / WarpShape::kN) *
    (ThreadblockShape::kK / WarpShape::kK);

  /// Argument structure (same as UnderlyingKernel::Arguments)
  using Arguments = typename UnderlyingKernel::Arguments;

private:

  typename UnderlyingKernel::Params params_;

public:

  Conv2dFpropWithPool() {}

  /// Checks problem constraints.
  static Status can_implement(Arguments const &args) {

    // Only kFprop with no split-K
    if (args.problem_size.split_k_slices != 1) {
      return Status::kErrorInvalidProblem;
    }
    if (args.pooled_output_ptr == nullptr) {
      return Status::kErrorInvalidProblem;
    }
    // P and Q must be even (MaxPool2x2 stride 2 assumption)
    if (args.problem_size.P % 2 != 0 || args.problem_size.Q % 2 != 0) {
      return Status::kErrorInvalidProblem;
    }

    Status status = UnderlyingKernel::Mma::IteratorA::can_implement(args.problem_size);
    if (status != Status::kSuccess) return status;

    status = UnderlyingKernel::Mma::IteratorB::can_implement(args.problem_size);
    if (status != Status::kSuccess) return status;

    // Check activation/filter/output sizes fit in 31-bit addressing
    if (args.problem_size.activation_size() * sizeof(ElementA) >= (1ull << 31) ||
        args.problem_size.filter_size()     * sizeof(ElementB) >= (1ull << 31)) {
      return Status::kErrorInvalidProblem;
    }

    // K alignment check
    static int const kAlignmentC =
      UnderlyingKernel::Epilogue::OutputTileIterator::kElementsPerAccess;
    if (args.problem_size.K % kAlignmentC) {
      return Status::kErrorMisalignedOperand;
    }

    // Grid dimensions must fit in uint16
    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid = threadblock_swizzle.get_grid_shape(
      threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices));

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max())) {
      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }

  /// No workspace needed (no split-K serial semaphore)
  static size_t get_workspace_size(Arguments const &) {
    return 0;
  }

  /// Initializes kernel params from arguments.
  Status initialize(
    Arguments const &args,
    void * /*workspace*/ = nullptr,
    cudaStream_t stream  = nullptr)
  {
    Status status = can_implement(args);
    if (status != Status::kSuccess) return status;

    params_ = typename UnderlyingKernel::Params(args, nullptr);

    int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(
        cutlass::Kernel<UnderlyingKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
      if (result != cudaSuccess) return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }

  /// Updates mutable pointers without recomputing grid shape.
  Status update(Arguments const &args, void * = nullptr) {
    params_.ptr_A      = args.ref_A.data();
    params_.ptr_B      = args.ref_B.data();
    params_.ptr_C      = args.ref_C.data();
    // Rebuild pool iterator_D params (carries the pooled_ptr)
    params_.iterator_D =
      typename UnderlyingKernel::Epilogue::OutputTileIterator::Params(
        args.problem_size, args.pooled_output_ptr);
    params_.output_op  = args.output_op;
    return Status::kSuccess;
  }

  /// Runs the kernel with the current params.
  Status run(cudaStream_t stream = nullptr) {

    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid  = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);
    int  smem  = int(sizeof(typename UnderlyingKernel::SharedStorage));

    cutlass::Kernel<UnderlyingKernel><<<grid, block, smem, stream>>>(params_);

    cudaError_t result = cudaGetLastError();
    return (result == cudaSuccess) ? Status::kSuccess : Status::kErrorInternal;
  }

  /// Combined initialize + run.
  Status operator()(
    Arguments const &args,
    void *workspace  = nullptr,
    cudaStream_t stream = nullptr)
  {
    Status status = initialize(args, workspace, stream);
    if (status == Status::kSuccess) status = run(stream);
    return status;
  }

  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
