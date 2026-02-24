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
  \brief Example: fused Conv2dFprop + ReLU + MaxPool2x2 on SM80 (A100/A10 etc.)

  Kernel configuration:
    - Input:       FP16, NHWC
    - Filter:      FP16, NHWC  (TensorNHWC stores filters in KRSC order for Fprop)
    - Accumulator: FP32
    - Epilogue:    LinearCombinationRelu (alpha=1, beta=0)
    - Output:      FP32 pooled tensor (atomicMax, MaxPool2x2 stride 2)
    - TensorCore:  SM80 16x8x16 HMMA
    - Pipeline:    3-stage multistage (AsyncCopy)
    - Threadblock: 128x128x64
    - Warp:         64x 64x64
    - Iterator:    Analytic (general conv parameters)

  Usage:
    ./fused_conv_relu_pool_f16_sm80 [--N=1] [--H=56] [--W=56] [--C=64] [--K=64]
                                    [--R=3] [--S=3] [--pad_h=1] [--pad_w=1]
                                    [--iterations=20] [--no-verify]
*/

#include <iostream>

// CUTLASS core
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

// Epilogue / output op
#include "cutlass/epilogue/thread/linear_combination_relu.h"

// Conv threadblock swizzle
#include "cutlass/conv/threadblock/threadblock_swizzle.h"

// Our custom headers (relative to this file's directory)
#include "kernel/default_conv2d_fprop_with_pool.h"
#include "device/conv2d_fprop_with_pool.h"
#include "conv_relu_pool_run.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

///
/// Define the fused Conv2dFprop+ReLU+MaxPool2x2 operator type.
///
/// The epilogue output op is LinearCombinationRelu:
///   output = max(0, alpha * accumulator + beta * source)
/// With alpha=1, beta=0 -> output = max(0, accumulator)
///
/// kCount=8 matches the vector width for FP16 with 128-bit memory access.
///

using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using LayoutA            = cutlass::layout::TensorNHWC;
using LayoutB            = cutlass::layout::TensorNHWC;
using LayoutC            = cutlass::layout::TensorNHWC;

using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationRelu<
  ElementC,           // output element type
  8,                  // vector width (8 x FP16 = 128 bits)
  ElementAccumulator, // accumulator type
  ElementAccumulator  // compute type
>;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape< 64,  64, 64>;
using InstructionShape = cutlass::gemm::GemmShape< 16,   8, 16>;

using ThreadblockSwizzle =
  cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>;

// Assemble the kernel via the DefaultConv2dFpropWithPool metafunction
using Conv2dFpropWithPoolKernel =
  typename cutlass::conv::kernel::DefaultConv2dFpropWithPool<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    /*Stages=*/3,
    cutlass::arch::OpMultiplyAdd   // actual SM80 FP16 tensor core instruction operator
  >::Kernel;

// Device-level wrapper
using FusedConvReluPool =
  cutlass::conv::device::Conv2dFpropWithPool<Conv2dFpropWithPoolKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **argv) {

  // Check SM version
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int sm = prop.major * 10 + prop.minor;
  if (sm < 80) {
    std::cerr << "This example requires SM80 or later (got SM" << sm << ").\n";
    return 1;
  }

  std::cout << "Device: " << prop.name << "  (SM" << sm << ")\n";

  cutlass::examples::ConvReluPoolOptions opts;
  if (!opts.parse(argc, argv)) return 0;

  if (!opts.valid()) {
    std::cerr << "Invalid problem: P and Q must be even, K and C must be multiples of 8.\n"
              << "  P=" << opts.P() << " Q=" << opts.Q()
              << " K=" << opts.K << " C=" << opts.C << "\n";
    return 1;
  }

  opts.print_problem();

  bool ok = cutlass::examples::run_conv_relu_pool<FusedConvReluPool>(opts);

  return ok ? 0 : 1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
