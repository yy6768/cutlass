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
  \brief Test harness for Conv2dFprop + fused MaxPool2x2.

  Provides:
    - ConvReluPoolOptions: parsed from command-line args
    - run_conv_relu_pool<Op>(): host function that
        1. Allocates and initialises input tensors
        2. Runs the fused kernel
        3. Runs a reference (host) implementation
        4. Compares outputs (FP32 pooled tensors)
        5. Prints throughput

  Reference implementation: unfused Conv + ReLU + MaxPool2x2 on host using
  cutlass::reference::host utilities.
*/

#pragma once

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/layout/tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace examples {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvReluPoolOptions {
  // Conv problem
  int N = 1;
  int H = 56, W = 56;
  int C = 64;
  int K = 64;
  int R = 3, S = 3;     // filter height, width (3x3 for first conv, 1x1 for second)
  int pad_h = 1, pad_w = 1;
  int stride_h = 1, stride_w = 1;
  int dilation_h = 1, dilation_w = 1;

  // Run options
  int iterations = 20;
  bool verify = true;

  // Parse from argc/argv.
  // Returns false and prints help if parsing fails.
  bool parse(int argc, char const **argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

      auto parse_int = [&](const char *key, int &val) -> bool {
        std::string prefix = std::string("--") + key + "=";
        if (arg.substr(0, prefix.size()) == prefix) {
          val = std::stoi(arg.substr(prefix.size()));
          return true;
        }
        return false;
      };

      if (arg == "--help" || arg == "-h") {
        print_usage();
        return false;
      }
      parse_int("N",          N)         ||
      parse_int("H",          H)         ||
      parse_int("W",          W)         ||
      parse_int("C",          C)         ||
      parse_int("K",          K)         ||
      parse_int("R",          R)         ||
      parse_int("S",          S)         ||
      parse_int("pad_h",      pad_h)     ||
      parse_int("pad_w",      pad_w)     ||
      parse_int("iterations", iterations)||
      (arg == "--no-verify" ? (verify = false, true) : false);
    }
    return true;
  }

  void print_usage() const {
    std::cout <<
      "Usage: fused_conv_relu_pool_f16_sm80 [options]\n"
      "\n"
      "  --N=<int>           Batch size (default 1)\n"
      "  --H=<int>           Input height (default 56)\n"
      "  --W=<int>           Input width  (default 56)\n"
      "  --C=<int>           Input channels (default 64)\n"
      "  --K=<int>           Output channels (default 64)\n"
      "  --R=<int>           Filter height (default 3)\n"
      "  --S=<int>           Filter width  (default 3)\n"
      "  --pad_h=<int>       Padding height (default 1)\n"
      "  --pad_w=<int>       Padding width  (default 1)\n"
      "  --iterations=<int>  Timing iterations (default 20)\n"
      "  --no-verify         Skip correctness check\n";
  }

  /// Compute output spatial dimensions given current settings.
  int P() const {
    return (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
  }
  int Q() const {
    return (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;
  }

  bool valid() const {
    // P and Q must be even for MaxPool2x2
    return (P() % 2 == 0) && (Q() % 2 == 0) && (K % 8 == 0) && (C % 8 == 0);
  }

  void print_problem() const {
    std::cout
      << "Problem: N=" << N
      << " H=" << H << " W=" << W << " C=" << C
      << " K=" << K << " R=" << R << " S=" << S
      << " -> P=" << P() << " Q=" << Q()
      << " pooled=(" << P()/2 << "x" << Q()/2 << ")\n";
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Host reference: conv2d (naive NHWC) + ReLU + MaxPool2x2 on float tensors.
/// Inputs/outputs are row-major NHWC float tensors.
inline void reference_conv_relu_pool(
  // Activation input:  [N, H, W, C]
  float const *input,
  // Filter:            [K, R, S, C]  (KRSC for Fprop)
  float const *filter,
  // Pooled output:     [N, P/2, Q/2, K]
  float *pooled,
  int N, int H, int W, int C,
  int K, int R, int S,
  int pad_h, int pad_w,
  int stride_h, int stride_w)
{
  int P = (H + 2*pad_h - R) / stride_h + 1;
  int Q = (W + 2*pad_w - S) / stride_w + 1;
  int P2 = P / 2, Q2 = Q / 2;

  // Initialize pooled to 0 (ReLU guarantees output >= 0)
  std::fill(pooled, pooled + N * P2 * Q2 * K, 0.0f);

  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int q_idx = 0; q_idx < Q; ++q_idx) {
        for (int k = 0; k < K; ++k) {

          // Convolution at (n, p, q_idx, k)
          float acc = 0.0f;
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
              int h_in = p * stride_h - pad_h + r;
              int w_in = q_idx * stride_w - pad_w + s;
              if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;
              for (int c = 0; c < C; ++c) {
                float a = input [((n * H + h_in) * W + w_in) * C + c];
                float b = filter[(k * R + r) * S * C + s * C + c];
                acc += a * b;
              }
            }
          }

          // ReLU
          float val = acc > 0.0f ? acc : 0.0f;

          // MaxPool2x2: write to pooled[n, p/2, q/2, k]
          int p2 = p / 2, q2 = q_idx / 2;
          float &dst = pooled[((n * P2 + p2) * Q2 + q2) * K + k];
          if (val > dst) dst = val;
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Run and verify the fused Conv2dFprop+ReLU+MaxPool2x2 op.
/// Template parameter Op must be a Conv2dFpropWithPool device op type.
template <typename Op>
bool run_conv_relu_pool(ConvReluPoolOptions const &opts) {
  using ElementA = typename Op::ElementA;   // FP16 activation
  using ElementB = typename Op::ElementB;   // FP16 filter
  using ElementC = typename Op::ElementC;   // FP16 (unused, beta=0)
  using LayoutA  = typename Op::LayoutA;    // TensorNHWC
  using LayoutB  = typename Op::LayoutB;    // TensorNHWC
  using LayoutC  = typename Op::LayoutC;    // TensorNHWC

  int P = opts.P(), Q = opts.Q();

  // Build problem size
  cutlass::conv::Conv2dProblemSize problem_size(
    opts.N, opts.H, opts.W, opts.C,
    opts.K, opts.R, opts.S, P, Q,
    opts.pad_h, opts.pad_w,
    opts.stride_h, opts.stride_w,
    opts.dilation_h, opts.dilation_w,
    cutlass::conv::Mode::kCrossCorrelation,
    /*split_k_slices=*/1);

  // Allocate device tensors
  cutlass::HostTensor<ElementA, LayoutA> tensor_A(
    cutlass::conv::implicit_gemm_tensor_a_extent(
      cutlass::conv::Operator::kFprop, problem_size));
  cutlass::HostTensor<ElementB, LayoutB> tensor_B(
    cutlass::conv::implicit_gemm_tensor_b_extent(
      cutlass::conv::Operator::kFprop, problem_size));
  cutlass::HostTensor<ElementC, LayoutC> tensor_C(
    cutlass::conv::implicit_gemm_tensor_c_extent(
      cutlass::conv::Operator::kFprop, problem_size));

  // Pooled output: [N, P/2, Q/2, K] in FP32
  int pooled_size = opts.N * (P / 2) * (Q / 2) * opts.K;

  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_pooled(
    {pooled_size, 1});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_pooled_ref(
    {pooled_size, 1});

  // Fill inputs
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_A.host_view(), /*seed=*/1, /*max=*/1.0, /*min=*/-1.0, /*bits=*/4);
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_B.host_view(), /*seed=*/2, /*max=*/1.0, /*min=*/-1.0, /*bits=*/4);
  cutlass::reference::host::TensorFill(tensor_C.host_view());  // zeros (beta=0)

  // Copy to device
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();

  // Zero-initialize pooled output buffers
  cudaMemset(tensor_pooled.device_data(),     0, pooled_size * sizeof(float));
  cudaMemset(tensor_pooled_ref.device_data(), 0, pooled_size * sizeof(float));

  // Build output op: alpha=1, beta=0, ReLU
  using EpilogueOp = typename Op::EpilogueOutputOp;
  typename EpilogueOp::Params output_op_params(
    ElementC(1),   // alpha
    ElementC(0));  // beta

  typename Op::Arguments args(
    problem_size,
    tensor_A.device_ref(),
    tensor_B.device_ref(),
    tensor_C.device_ref(),
    tensor_pooled.device_data(),
    output_op_params);

  // Check feasibility
  cutlass::Status status = Op::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "can_implement failed: "
              << cutlass::cutlassGetStatusString(status) << "\n";
    return false;
  }

  // Instantiate and run
  Op op;
  status = op.initialize(args, nullptr);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "initialize failed: "
              << cutlass::cutlassGetStatusString(status) << "\n";
    return false;
  }

  status = op.run();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "run failed: "
              << cutlass::cutlassGetStatusString(status) << "\n";
    return false;
  }
  cudaDeviceSynchronize();

  // -----------------------------------------------------------------------
  // Correctness check
  // -----------------------------------------------------------------------
  bool passed = true;

  if (opts.verify) {
    tensor_A.sync_host();
    tensor_B.sync_host();

    // Build float copies of A and B for the host reference
    std::vector<float> A_f32(tensor_A.size());
    std::vector<float> B_f32(tensor_B.size());
    for (size_t i = 0; i < A_f32.size(); ++i)
      A_f32[i] = float(tensor_A.host_data()[i]);
    for (size_t i = 0; i < B_f32.size(); ++i)
      B_f32[i] = float(tensor_B.host_data()[i]);

    std::vector<float> pooled_ref(pooled_size, 0.0f);
    reference_conv_relu_pool(
      A_f32.data(), B_f32.data(), pooled_ref.data(),
      opts.N, opts.H, opts.W, opts.C,
      opts.K, opts.R, opts.S,
      opts.pad_h, opts.pad_w,
      opts.stride_h, opts.stride_w);

    // Copy device result to host
    tensor_pooled.sync_host();
    float const *result = tensor_pooled.host_data();

    float max_err = 0.0f, max_ref = 0.0f;
    for (int i = 0; i < pooled_size; ++i) {
      float err = std::abs(result[i] - pooled_ref[i]);
      if (err > max_err) max_err = err;
      if (std::abs(pooled_ref[i]) > max_ref) max_ref = std::abs(pooled_ref[i]);
    }

    float rel_err = (max_ref > 0.0f) ? (max_err / max_ref) : max_err;
    bool ok = (rel_err < 1e-2f);  // 1% relative tolerance for FP16 accumulation

    std::cout << "Verification: " << (ok ? "PASSED" : "FAILED")
              << "  max_abs_err=" << max_err
              << "  max_rel_err=" << rel_err << "\n";
    passed = ok;
  }

  // -----------------------------------------------------------------------
  // Throughput measurement
  // -----------------------------------------------------------------------
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 3; ++i) {
      cudaMemset(tensor_pooled.device_data(), 0, pooled_size * sizeof(float));
      op.run();
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < opts.iterations; ++i) {
      cudaMemset(tensor_pooled.device_data(), 0, pooled_size * sizeof(float));
      op.run();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= opts.iterations;

    // FLOP count: 2 * N * P * Q * K * R * S * C  (multiply-add)
    double flops = 2.0 * opts.N * P * Q * opts.K * opts.R * opts.S * opts.C;
    double tflops = flops / (ms * 1e-3) * 1e-12;

    std::cout << "Runtime: " << ms << " ms  |  Throughput: " << tflops << " TFLOP/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace examples
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
