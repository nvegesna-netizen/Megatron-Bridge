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
Default dispatcher: moe_flex_dispatcher_backend="deepep" (as set in the recipe).

Config naming convention::

    STEP35_196B_A11B_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

All parallelism values below are HYPOTHESES based on model architecture and
analogy to similarly-sized MoE models (DeepSeek-V3 / Kimi-K2).  They have
NOT been verified on actual hardware and must be tuned before production use.

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
# GB300 — HYPOTHESIS: TP=8, PP=2, EP=32, deepep backend
# NVL-72 chassis; 256 GPUs expected for this topology.
# HYPOTHESIS: same deepep backend as in the base recipe (no hybridep).
# EP=32: 288 experts / 32 = 9 experts per EP rank (evenly divisible).
# EP=64 was incorrect: 288 % 64 = 32 ≠ 0, causing an AssertionError in
# moe_layer.py at model init.
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    # HYPOTHESIS: 256 GPUs (4 NVL-72 nodes)
    num_gpus=256,
    # HYPOTHESIS: TP=8 fills NVLink domain; PP=2 keeps pipeline shallow
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=None,
    # HYPOTHESIS: EP=32 — 288 experts / 32 = 9 per rank (evenly divisible)
    expert_model_parallel_size=32,
    # HYPOTHESIS: CP=1 (4096 seq_len does not require CP splitting)
    context_parallel_size=1,
    # HYPOTHESIS: GBS=2048, MBS=1 is a typical starting point for MoE pretrain
    global_batch_size=2048,
    micro_batch_size=1,
    # deepep is the recipe default; keep it here so the config is self-describing
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    # HYPOTHESIS: selective recompute on MoE activation is a lightweight option
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
)

STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_V1


# =============================================================================
# GB200 — HYPOTHESIS: TP=8, PP=2, EP=32, deepep backend
# Same node topology as GB300; precision variants differ.
# EP=32: 288 experts / 32 = 9 experts per EP rank (evenly divisible).
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    # HYPOTHESIS: 256 GPUs
    num_gpus=256,
    # HYPOTHESIS: TP=8, PP=2 — mirrors GB300 BF16 shape
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=None,
    # HYPOTHESIS: EP=32 — 288 experts / 32 = 9 per rank (evenly divisible)
    expert_model_parallel_size=32,
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
)

STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_V1

# HYPOTHESIS: FP8 variant — same topology as BF16; recipe-level precision config handles the rest.
STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_V1


# =============================================================================
# B200 — HYPOTHESIS: TP=8, PP=4, EP=32, deepep backend
# Non-NVLink-72 Blackwell; smaller NVLink domain limits EP.
# HYPOTHESIS: unverified — derived by analogy from GB200 shape with reduced EP.
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_B200_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    # HYPOTHESIS: 256 GPUs
    num_gpus=256,
    # HYPOTHESIS: TP=8, PP=4 to compensate for smaller memory per GPU
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=None,
    # HYPOTHESIS: EP=32 (smaller NVLink domain than NVL-72)
    expert_model_parallel_size=32,
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
)

STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B200_V1


# =============================================================================
# B300 — HYPOTHESIS: same topology as B200
# HYPOTHESIS: unverified — B300 assumed similar to B200 until profiled.
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_B300_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    # HYPOTHESIS: 256 GPUs
    num_gpus=256,
    # HYPOTHESIS: TP=8, PP=4 (same as B200)
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=32,
    context_parallel_size=1,
    global_batch_size=2048,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    recompute_modules=["moe_act"],
    cuda_graph_impl=None,
)

STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_B300_V1


# =============================================================================
# H100 — HYPOTHESIS: TP=8, PP=4, EP=32, GBS=8192
# Requires more GPUs due to HBM capacity; full-recompute on select layers.
# HYPOTHESIS: unverified — numbers are rough estimates pending actual profiling.
# =============================================================================

STEP35_196B_A11B_PRETRAIN_CONFIG_H100_V1 = replace(
    BASE_STEP35_196B_A11B_CONFIG,
    # HYPOTHESIS: 1024 GPUs at TP=8, PP=4, EP=32
    num_gpus=1024,
    # HYPOTHESIS: TP=8, PP=4 needed for memory on 80 GB HBM
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=None,
    # HYPOTHESIS: EP=32 across nodes; deepep still valid on H100
    expert_model_parallel_size=32,
    context_parallel_size=1,
    # HYPOTHESIS: larger GBS to amortize pipeline bubbles over more GPUs
    global_batch_size=8192,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    # HYPOTHESIS: more aggressive recompute on H100 due to lower HBM capacity
    recompute_modules=["moe_act", "mlp"],
    cuda_graph_impl=None,
)

STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V1 = STEP35_196B_A11B_PRETRAIN_CONFIG_H100_V1


__all__ = [
    # V1 (initial HYPOTHESIS configs)
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_GB200_FP8_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B200_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_B300_BF16_V1",
    "STEP35_196B_A11B_PRETRAIN_CONFIG_H100_BF16_V1",
]
