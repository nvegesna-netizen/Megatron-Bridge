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

"""Workload base presets for Step-3.5-Flash performance configs.

Model: stepfun-ai/Step-3.5-Flash
Architecture: 196B total params, ~11B activated per token, 45 layers,
288 routed + 1 shared experts, top-8 routing, 3:1 SWA:full hybrid attention.
Default dispatcher: hybridep for NVL-72 chassis (GB300/GB200); deepep for B200/B300/H100.

Config naming convention::

    STEP35_196B_A11B_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

Parallelism follows the recipe author's recommendation (step35.py docstring):
    TP=1, PP=8, EP=8
Context parallelism (CP) is kept at 1 despite the recipe suggesting CP=8:
ring-attention with Step-3.5's per-layer sliding-window config deep-copies
produces incorrect causal masks across CP ranks and is blocked by the
validate_sweep_config guard in megatron_bridge_helpers.py.

45 layers is not divisible by PP=8 (45/8=5.625), so all configs carry an
explicit pp_layout: 5 stages of 6 decoder layers + 3 stages of 5 decoder
layers (lighter at the embedding/loss boundary stages).

Use ``--config_variant`` to select a variant.
Use ``--list_config_variants`` to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


# ---------------------------------------------------------------------------
# Base template — fields shared by all Step-3.5-Flash configs.
# The recipe already sets expert_tensor_parallel_size=1; keep it explicit here
# so get_workload_base_config() propagates the correct value.
# ---------------------------------------------------------------------------
BASE_STEP35_196B_A11B_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


# =============================================================================
# GB300 — TP=1, PP=8, EP=8 (recipe author recommendation)
# NVL-72 chassis; 256 GPUs for this topology.
# hybridep: NVL-72 uses NVLink fabric for intra-domain expert dispatch.
# EP=8: 288 experts / 8 = 36 per EP rank (evenly divisible).
# TP=1, PP=8 → DP = 256 / (1 × 8) = 32; EP=8 ≤ DP=32 ✓
# 45 layers / PP=8 = 5.625 → non-uniform pp_layout required.
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=None,
    # EP=8: 288 experts / 8 = 36 per rank (evenly divisible)
    expert_model_parallel_size=8,
    # CP=1: ring-attention + per-layer SWA config deep-copy produces incorrect
    # causal masks; blocked by validate_sweep_config guard.
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
    # 45 layers across 8 PP stages: 5 stages × 6 layers + 3 stages × 5 layers = 45.
    # Boundary stages (embedding, loss) carry fewer layers.
    pp_layout="Et*5|t*6|t*6|t*6|t*6|t*6|t*5|t*5L",
)

STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_V1


# =============================================================================
# GB200 — TP=1, PP=8, EP=8 (recipe author recommendation)
# Same node topology as GB300; precision variants differ.
# EP=8: 288 experts / 8 = 36 per EP rank (evenly divisible).
# TP=1, PP=8 → DP = 256 / (1 × 8) = 32; EP=8 ≤ DP=32 ✓
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=8,
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
    pp_layout="Et*5|t*6|t*6|t*6|t*6|t*6|t*5|t*5L",
)

STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_V1

# FP8 variant — same topology as BF16; recipe-level precision config handles the rest.
STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_V1


# =============================================================================
# B200 — TP=1, PP=8, EP=8 (recipe author recommendation)
# TP=1, PP=8 → DP = 256 / (1 × 8) = 32; EP=8 ≤ DP=32 ✓
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_B200_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=8,
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
    pp_layout="Et*5|t*6|t*6|t*6|t*6|t*6|t*5|t*5L",
)

STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B200_V1


# =============================================================================
# B300 — TP=1, PP=8, EP=8 (recipe author recommendation; same topology as B200)
# TP=1, PP=8 → DP = 256 / (1 × 8) = 32; EP=8 ≤ DP=32 ✓
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_B300_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=8,
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
    pp_layout="Et*5|t*6|t*6|t*6|t*6|t*6|t*5|t*5L",
)

STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B300_V1


# =============================================================================
# H100 — TP=1, PP=8, EP=8 (recipe author recommendation)
# 1024 GPUs; larger fleet needed for HBM capacity at TP=1.
# TP=1, PP=8 → DP = 1024 / (1 × 8) = 128; EP=8 ≤ DP=128 ✓
# More aggressive recompute on H100 due to lower HBM capacity vs Blackwell.
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_H100_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=8,
    context_parallel_size=1,
    global_batch_size=8192,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act", "mlp"],
    cuda_graph_impl=None,
    pp_layout="Et*5|t*6|t*6|t*6|t*6|t*6|t*5|t*5L",
)

STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_H100_V1


__all__ = [
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V1",
]
