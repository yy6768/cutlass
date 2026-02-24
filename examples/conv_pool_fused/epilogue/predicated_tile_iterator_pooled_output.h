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
  \brief Epilogue output tile iterator with fused MaxPool2x2 via atomicMax.

  Replaces the standard PredicatedTileIterator's store() with coordinate
  remapping + atomicMax to a pooled output buffer (FP32).

  Key idea:
    Linearized GEMM M index -> (n, p, q) spatial coords -> (n, p/2, q/2) pooled coords
    -> atomicMax((int*)&pooled[offset], __float_as_int(relu_output))

  IEEE-754 guarantees that for non-negative floats (ReLU output >= 0),
  integer comparison of bit patterns preserves float ordering.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/epilogue/threadblock/output_tile_thread_map.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_params.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/fast_math.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator that fuses MaxPool2x2 into the epilogue store via atomicMax.
///
/// Instead of writing to a full-resolution output tensor, this iterator
/// decomposes the linearized GEMM M index into (n, p, q) spatial coordinates,
/// maps them to (n, p/2, q/2) pooled coordinates, and uses atomicMax to write
/// the maximum value in each 2x2 pool window to an FP32 pooled output buffer.
///
/// Satisfies the OutputTileIterator concept required by Epilogue.
///
template <
  typename ThreadMap_,       ///< Thread map (concept: OutputTileThreadMap)
  typename Element_          ///< Element data type (typically float)
>
class PredicatedTileIteratorPooledOutput {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
  static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
  static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
  static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");

  /// Fragment object — identical shape to standard PredicatedTileIterator
  using Fragment = Array<
    Element,
    ThreadMap::Iterations::kColumn *
    ThreadMap::Iterations::kRow *
    ThreadMap::Iterations::kGroup *
    ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  struct Params {

    //
    // Data members
    //

    int N;           ///< Batch size
    int P;           ///< Conv output height (full resolution)
    int Q;           ///< Conv output width (full resolution)
    int K;           ///< Output channels
    int P_pool;      ///< Pooled output height (= P / 2)
    int Q_pool;      ///< Pooled output width (= Q / 2)
    float *pooled_ptr;  ///< Pointer to FP32 pooled output buffer

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : N(0), P(0), Q(0), K(0), P_pool(0), Q_pool(0), pooled_ptr(nullptr) {}

    /// Constructor for pooled output (used for iterator_D)
    CUTLASS_HOST_DEVICE
    Params(
      conv::Conv2dProblemSize const &problem_size,
      float *pooled_ptr_
    ) :
      N(problem_size.N),
      P(problem_size.P),
      Q(problem_size.Q),
      K(problem_size.K),
      P_pool(problem_size.P / 2),
      Q_pool(problem_size.Q / 2),
      pooled_ptr(pooled_ptr_)
    {}

    /// Compatibility constructor (for iterator_C, typically unused with beta=0)
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) : Params() {}

    /// Compatibility constructor with extent
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, cutlass::Tensor4DCoord const &) : Params() {}

    /// Compatibility constructor with extent
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, cutlass::Tensor5DCoord const &) : Params() {}

    /// Compatibility constructor with MatrixCoord extent
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout, MatrixCoord const &) : Params() {}
  };

  /// Mask object
  struct Mask {

    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< Efficiently enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

private:

  //
  // Data members
  //

  /// Pool parameters
  int N_;
  int P_;
  int Q_;
  int K_;
  int P_pool_;
  int Q_pool_;
  int PQ_;           ///< = P * Q, precomputed
  float *pooled_ptr_;

  /// Fast integer division constants (precomputed for P*Q and Q)
  uint32_t pq_mul_;
  uint32_t pq_shr_;
  uint32_t q_mul_;
  uint32_t q_shr_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows (M = N*P*Q)
  Index extent_row_;

  /// Extent of the matrix tile in columns (K)
  Index extent_column_;

  /// A thread's starting row position
  Index thread_start_row_;

  /// A thread's starting column position
  Index thread_start_column_;

  /// Internal state counter (for operator++ tracking)
  int state_[3];

public:

  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  PredicatedTileIteratorPooledOutput(
    Params const &params,
    Element *pointer,             ///< Not used for pooled store; kept for interface compatibility
    TensorCoord extent,           ///< (N*P*Q, K) — full resolution extent
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord(),
    int const *indices = nullptr  ///< Not used; kept for interface compatibility
  ) :
    N_(params.N),
    P_(params.P),
    Q_(params.Q),
    K_(params.K),
    P_pool_(params.P_pool),
    Q_pool_(params.Q_pool),
    pooled_ptr_(params.pooled_ptr)
  {
    PQ_ = P_ * Q_;

    // Precompute fast divmod constants for coordinate decomposition
    find_divisor(pq_mul_, pq_shr_, PQ_ > 0 ? PQ_ : 1);
    find_divisor(q_mul_, q_shr_, Q_ > 0 ? Q_ : 1);

    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent.row();
    extent_column_ = extent.column();

    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize column predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] = ((thread_offset.column()
        + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // If no pooled output buffer, disable all accesses
    if (!pooled_ptr_) {
      mask_.clear();
    }

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element (no-op for pooled output)
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    // No byte pointer to advance; pooled addresses are computed from coordinates
  }

  /// Loads a fragment from memory (returns zeros; source C is not needed for ReLU with beta=0)
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) const {
    frag.clear();
  }

  /// Loads a fragment from memory (returns zeros)
  CUTLASS_DEVICE
  void load(Fragment &frag) const {
    frag.clear();
  }

  /// Stores a fragment to the pooled output via coordinate remapping + atomicMax
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) const {
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow
            + group * ThreadMap::Delta::kGroup
            + cluster * ThreadMap::Delta::kCluster;

          // 1. Linearized GEMM M index
          int m = row_offset + thread_start_row_;

          bool row_guard = (m < extent_row_);

          // 2. Decompose m -> (n, p, q) using fast integer division
          int n, pq;
          fast_divmod(n, pq, m, PQ_, pq_mul_, pq_shr_);

          int p, q;
          fast_divmod(p, q, pq, Q_, q_mul_, q_shr_);

          // 3. Compute pooled coordinates
          int p_pool = p >> 1;   // p / 2
          int q_pool = q >> 1;   // q / 2

          // 4. Compute base offset in pooled output: [n, p_pool, q_pool, :]
          int pooled_base = ((n * P_pool_ + p_pool) * Q_pool_ + q_pool) * K_;

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            if (guard) {
              int k_base = thread_start_column_
                + column * ThreadMap::Delta::kColumn;

              // Get the access-width fragment for this (row, column) position
              AccessType const &access =
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
              Element const *elem_ptr = reinterpret_cast<Element const *>(&access);

              CUTLASS_PRAGMA_UNROLL
              for (int v = 0; v < kElementsPerAccess; ++v) {
                int k = k_base + v;

                if (k < K_) {
                  // Convert to float for atomicMax
                  float val = static_cast<float>(elem_ptr[v]);

                  // atomicMax using IEEE-754 bit trick for non-negative floats
                  // ReLU guarantees val >= 0, so __float_as_int preserves ordering
                  atomicMax(
                    reinterpret_cast<int*>(pooled_ptr_ + pooled_base + k),
                    __float_as_int(val));
                }
              }
            }
          }
        }
      }
    }
  }

  /// Stores a fragment to the pooled output
  CUTLASS_DEVICE
  void store(Fragment const &frag) const {
    store_with_byte_offset(frag, 0);
  }

  CUTLASS_DEVICE
  MatrixCoord thread_start() const {
    return MatrixCoord(thread_start_row_, thread_start_column_);
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const {
    return thread_start_row_;
  }

  /// Need to get the thread start column from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const {
    return thread_start_column_;
  }

  /// Extent of the matrix in rows
  CUTLASS_DEVICE
  Index extent_row() const {
    return extent_row_;
  }

  /// Extent of the matrix in columns
  CUTLASS_DEVICE
  Index extent_column() const {
    return extent_column_;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorPooledOutput &operator++() {

    ++state_[0];

    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {

      state_[0] = 0;
      ++state_[1];

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {

        state_[1] = 0;
        ++state_[2];

        thread_start_row_ += ThreadMap::Count::kGroup *
          ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;

          thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow
            * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
        }
      }
    }

    return *this;
  }

  /// Advances a number of positions to load or store
  CUTLASS_HOST_DEVICE
  PredicatedTileIteratorPooledOutput &operator+=(int increment) {
    // Row
    state_[0] += increment;
    int increment_row = state_[0] / ThreadMap::Count::kRow;
    state_[0] = state_[0] % ThreadMap::Count::kRow;
    thread_start_row_ += (ThreadMap::Shape::kRow * increment);

    // Group
    state_[1] += increment_row;
    int increment_group = state_[1] / ThreadMap::Count::kGroup;
    state_[1] = state_[1] % ThreadMap::Count::kGroup;
    thread_start_row_ +=
        (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow *
        ThreadMap::Count::kRow *
        increment_row;

    // Cluster
    state_[2] += increment_group;
    int increment_cluster = state_[2] / ThreadMap::Count::kCluster;
    state_[2] = state_[2] % ThreadMap::Count::kCluster;
    thread_start_row_ +=
        ThreadMap::Count::kGroup *
        ThreadMap::Shape::kGroup *
        ThreadMap::Count::kRow *
        ThreadMap::Shape::kRow *
        increment_group;

    // Tile
    thread_start_row_ +=
        ThreadMap::Shape::kGroup *
        ThreadMap::Shape::kRow *
        ThreadMap::Shape::kCluster *
        ThreadMap::Shape::kTile *
        increment_cluster;

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Gets the mask
  CUTLASS_DEVICE void get_mask(Mask &mask) const {
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
