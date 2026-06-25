# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workload base presets for MiniMax-M2 performance configs.

Model: MiniMaxAI/MiniMax-M2
Architecture: 229.9B total params, ~9.8B activated per token, 62 layers,
256 experts, top-8 sigmoid routing with expert bias correction.
Default dispatcher: moe_flex_dispatcher_backend="deepep" (as set in the recipe).

Config naming convention::

    MINIMAX_M2_229B_A10B_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

All parallelism values below are HYPOTHESES — not benchmark-confirmed.
They were derived from architectural constraints and NVBug/Slack evidence:

- TP=1: required — MiniMax-M2 uses full-dimension QK norm (FullDimQNorm/
  FullDimKNorm) applied over ``num_heads * head_dim`` *before* TP-axis head
  splitting.  TP>1 would compute the norm over only ``num_heads/TP`` heads,
  producing incorrect normalisation values.  See minimax_m2.py for the
  architectural rationale.
- PP=2: 62 transformer layers; Megatron asserts ``num_layers % PP == 0`` at
  model construction (``transformer_block.py:149``).  Divisors of 62: 1, 2, 31.
  PP=2 is the only practical choice.
- EP=16: NVBug 6104911 (Slack #qz_hermes_agent, Apr-May 2026) reports a
  persistent ``RuntimeError: DeepEP error: timeout (dispatch CPU)`` at
  ep_size=32 for MiniMax-M2.1.  256 experts / EP=16 = 16 experts per rank.
  EP=32 must not be used until the NVBug is resolved.
- num_gpus: H100 = 1024 (128 nodes × 8 GPUs/node); all Blackwell targets
  (GB200, GB300, B200, B300) = 256 (32 nodes × 8 GPUs/node).

Use ``--config_variant`` to select a variant.
Use ``--list_config_variants`` to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


# ---------------------------------------------------------------------------
# Base template — fields shared by all MiniMax-M2 configs.
# ---------------------------------------------------------------------------
BASE_MINIMAX_M2_229B_A10B_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


# =============================================================================
# H100 — HYPOTHESIS: not benchmarked
# TP=1 (required), PP=2 (only clean divisor of 62), EP=16 (NVBug 6104911).
# 1024 GPUs = 128 nodes × 8 GPUs/node.
# 256 experts / EP=16 = 16 experts per EP rank.
# expert_data_parallel = 1024 / (ETP=1 × EP=16 × PP=2) = 32.
# =============================================================================

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_H100_V1 = replace(
    BASE_MINIMAX_M2_229B_A10B_CONFIG,
    # HYPOTHESIS: not benchmarked, topology constrained by architecture
    # 1024 GPUs = 128 nodes × 8 GPUs/node
    num_gpus=1024,
    # TP=1: full-dimension QK norm is TP-unsafe (see module docstring)
    tensor_model_parallel_size=1,
    # PP=2: only practical divisor of 62 transformer layers
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=None,
    # EP=16: NVBug 6104911 — avoid EP=32; 256 experts / 16 = 16 per rank
    expert_model_parallel_size=16,
    # CP=1: GQA is CP-compatible but untested at high EP; see module docstring
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    # Full recompute on H100 (80 GB HBM); switch to selective if memory permits
    recompute_modules=None,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    cuda_graph_impl=None,
)

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_H100_V1,
    recompute_num_layers=1,
)


# =============================================================================
# GB200 — HYPOTHESIS: not benchmarked
# TP=1, PP=2, EP=16, hybridep; NVL72 chassis.
# 256 GPUs = 32 nodes × 8 GPUs/node.
# 256 experts / EP=16 = 16 experts per EP rank.
# expert_data_parallel = 256 / (ETP=1 × EP=16 × PP=2) = 8.
# =============================================================================

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_GB200_V1 = replace(
    BASE_MINIMAX_M2_229B_A10B_CONFIG,
    # HYPOTHESIS: not benchmarked, topology constrained by architecture
    # 256 GPUs = 32 nodes × 8 GPUs/node
    num_gpus=256,
    # TP=1: full-dimension QK norm is TP-unsafe (see module docstring)
    tensor_model_parallel_size=1,
    # PP=2: only practical divisor of 62 transformer layers
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=None,
    # EP=16: NVBug 6104911 — avoid EP=32; 256 experts / 16 = 16 per rank
    expert_model_parallel_size=16,
    # CP=1: GQA is CP-compatible but untested at high EP; see module docstring
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    # hybridep on NVL72 chassis for cross-node expert dispatch
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
)

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_GB200_BF16_V1 = MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_GB200_V1


# =============================================================================
# GB300 — HYPOTHESIS: not benchmarked
# Same topology as GB200: TP=1, PP=2, EP=16, hybridep.
# GB300 has higher TFLOPs but the same NVLink domain topology as GB200.
# =============================================================================

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_GB300_V1 = replace(
    BASE_MINIMAX_M2_229B_A10B_CONFIG,
    # HYPOTHESIS: not benchmarked, topology constrained by architecture
    # 256 GPUs = 32 nodes × 8 GPUs/node
    num_gpus=256,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=16,
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
)

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_GB300_BF16_V1 = MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_GB300_V1


# =============================================================================
# B200 — HYPOTHESIS: not benchmarked
# TP=1, PP=2, EP=16, deepep; non-NVL72 Blackwell.
# 256 GPUs = 32 nodes × 8 GPUs/node.
# =============================================================================

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_B200_V1 = replace(
    BASE_MINIMAX_M2_229B_A10B_CONFIG,
    # HYPOTHESIS: not benchmarked, topology constrained by architecture
    # 256 GPUs = 32 nodes × 8 GPUs/node (same scale as GB200/GB300/B300)
    num_gpus=256,
    # TP=1: full-dimension QK norm is TP-unsafe (see module docstring)
    tensor_model_parallel_size=1,
    # PP=2: only practical divisor of 62 transformer layers
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=None,
    # EP=16: NVBug 6104911 — avoid EP=32; 256 experts / 16 = 16 per rank
    expert_model_parallel_size=16,
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
)

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_B200_BF16_V1 = MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_B200_V1


# =============================================================================
# B300 — HYPOTHESIS: not benchmarked
# TP=1, PP=2, EP=16, deepep; non-NVL72 Blackwell (higher TFLOPs than B200).
# 256 GPUs = 32 nodes × 8 GPUs/node.
# =============================================================================

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_B300_V1 = replace(
    BASE_MINIMAX_M2_229B_A10B_CONFIG,
    # HYPOTHESIS: not benchmarked, topology constrained by architecture
    # 256 GPUs = 32 nodes × 8 GPUs/node
    num_gpus=256,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=16,
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
)

MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_B300_BF16_V1 = MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_B300_V1


__all__ = [
    # V1 (initial HYPOTHESIS configs)
    "MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_B200_BF16_V1",
    "MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_B300_BF16_V1",
    "MINIMAX_M2_229B_A10B_PRETRAIN_CONFIG_H100_BF16_V1",
]
