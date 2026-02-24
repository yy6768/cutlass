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
  \brief Epilogue assembly metafunction for Conv2d + MaxPool fusion.

  Replaces the standard OutputTileIterator with PredicatedTileIteratorPooledOutput.
  All other components (ThreadMap, AccumulatorFragmentIterator, WarpTileIterator,
  SharedLoadIterator, Padding) are reused from the standard DefaultEpilogueTensorOp.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "cutlass/platform/platform.h"

#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue.h"

#include "predicated_tile_iterator_pooled_output.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue assembly metafunction with fused MaxPool output.
///
/// This is identical to DefaultEpilogueTensorOp except that OutputTileIterator
/// is replaced with PredicatedTileIteratorPooledOutput which performs
/// coordinate remapping + atomicMax for MaxPool2x2 fusion.
///
template <
  typename Shape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess
>
struct DefaultEpilogueTensorOpWithPool {

  using Shape = Shape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  static int const kPartitionsK = PartitionsK;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map — same as standard DefaultEpilogueTensorOp
  //

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    Shape,
    typename WarpMmaTensorOp::Shape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  //
  // OutputTileIterator — the ONLY change: use our pooled output iterator
  //

  using OutputTileIterator = PredicatedTileIteratorPooledOutput<
    OutputTileThreadMap,
    ElementOutput
  >;

  //
  // AccumulatorFragmentIterator — reused from standard
  //

  using AccumulatorFragmentIterator = typename platform::conditional<is_complex<ElementOutput>::value,
                                    cutlass::epilogue::warp::FragmentIteratorComplexTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC>,
                                    cutlass::epilogue::warp::FragmentIteratorTensorOp<
                                        typename WarpMmaTensorOp::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC> >::type;

  //
  // WarpTileIterator and SharedLoadIterator — reused from standard
  //

  using DefaultIterators = detail::DefaultIteratorsTensorOp<
    ElementOutput,
    ElementAccumulator,
    kElementsPerAccess,
    Shape,
    typename WarpMmaTensorOp::Shape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename OutputTileThreadMap::CompactedThreadMap
  >;

  using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
  using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;

  /// Hard-coded padding elements added
  using Padding = cutlass::MatrixShape<0, 64 / sizeof_bits<ElementAccumulator>::value * 4>;

  static int const kFragmentsPerIteration = (kPartitionsK == 1 ? DefaultIterators::kFragmentsPerIteration : 1);

  //
  // Define the epilogue — same structure, but uses our OutputTileIterator
  //
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    SharedLoadIterator,
    OutputOp,
    Padding,
    kFragmentsPerIteration
  >;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
