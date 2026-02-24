# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cutlass_example_add_executable(
  conv_pool_fused
  fused_conv_relu_pool_f16_sm80.cu
  DISABLE_TESTS ON
)

# Add the example source tree so relative includes (kernel/..., device/...) resolve correctly.
target_include_directories(
  conv_pool_fused
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
