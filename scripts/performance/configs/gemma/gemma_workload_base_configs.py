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

"""Workload base presets for Gemma 4 performance configs.

Model: google/gemma-4-E4B-it (Dense, ~3.8B parameters)
Architecture: 42 layers, hidden_size=2560, 8 attention heads, 2 KV heads (sliding),
2 KV heads (global), SWA hybrid attention (window_attn_skip_freq=6; 1 global every 6 layers),
Per-Layer Embeddings (PLE), dual RoPE.

Config naming convention::

    GEMMA4_E4B_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

All parallelism values below are HYPOTHESES based on model architecture and
analogy to similarly-sized dense models.  They have NOT been verified on actual
hardware and must be tuned before production use.

NOTE: SWA hybrid attention requires long sequence lengths for correctness; context
parallelism (CP) support with SWA has not been verified in this bridge.  Use
SEQ>=4096 and verify CP behaviour before enabling context parallelism.

Use ``--config_variant`` to select a variant.
Use ``--list_config_variants`` to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


# ---------------------------------------------------------------------------
# Base template — fields shared by all Gemma 4 E4B configs.
# The E4B recipe uses transformer_impl="local" (non-TE); keep EP=1 (dense).
# ---------------------------------------------------------------------------
BASE_GEMMA4_E4B_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


# =============================================================================
# H100 — HYPOTHESIS: TP=4, PP=1, EP=1, GBS=512, MBS=1
# Gemma4DenseProvider enforces PP=1 only (raises NotImplementedError for PP>1).
# HYPOTHESIS: not benchmarked; derived from model size and analogy to similar
# dense ~4B models on H100.
# NOTE: SWA hybrid attention requires long SEQ; CP support unverified.
# =============================================================================

GEMMA4_E4B_PRETRAIN_CONFIG_H100_V1 = replace(
    BASE_GEMMA4_E4B_CONFIG,
    # HYPOTHESIS: 4 GPUs (TP=4 * PP=1)
    num_gpus=4,
    # HYPOTHESIS: TP=4, PP=1 — Gemma4DenseProvider enforces PP=1 only
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    # Dense model: no expert parallelism
    expert_model_parallel_size=1,
    # HYPOTHESIS: CP=1 — SWA CP support unverified
    context_parallel_size=1,
    # HYPOTHESIS: GBS=512, MBS=1
    global_batch_size=512,
    micro_batch_size=1,
    cuda_graph_impl=None,
)

GEMMA4_E4B_PRETRAIN_CONFIG_H100_BF16_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_H100_V1


# =============================================================================
# GB200 — HYPOTHESIS: TP=4, PP=1, EP=1, GBS=512, MBS=2
# NVL-72 chassis; larger HBM allows PP=1 and MBS=2.
# HYPOTHESIS: not benchmarked.
# =============================================================================

GEMMA4_E4B_PRETRAIN_CONFIG_GB200_V1 = replace(
    BASE_GEMMA4_E4B_CONFIG,
    # HYPOTHESIS: 4 GPUs (TP=4 * PP=1)
    num_gpus=4,
    # HYPOTHESIS: TP=4, PP=1 — sufficient HBM on GB200
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    context_parallel_size=1,
    # HYPOTHESIS: GBS=512, MBS=2 — larger batch fits in GB200 HBM
    global_batch_size=512,
    micro_batch_size=2,
    cuda_graph_impl=None,
)

GEMMA4_E4B_PRETRAIN_CONFIG_GB200_BF16_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_GB200_V1


# =============================================================================
# GB300 — HYPOTHESIS: same topology as GB200
# HYPOTHESIS: unverified — GB300 assumed similar to GB200 until profiled.
# =============================================================================

GEMMA4_E4B_PRETRAIN_CONFIG_GB300_V1 = replace(
    BASE_GEMMA4_E4B_CONFIG,
    # HYPOTHESIS: 4 GPUs (TP=4 * PP=1)
    num_gpus=4,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    context_parallel_size=1,
    global_batch_size=512,
    micro_batch_size=2,
    cuda_graph_impl=None,
)

GEMMA4_E4B_PRETRAIN_CONFIG_GB300_BF16_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_GB300_V1


# =============================================================================
# B200 — HYPOTHESIS: TP=4, PP=1, EP=1, GBS=512, MBS=1
# Gemma4DenseProvider enforces PP=1 only. Non-NVLink-72 Blackwell.
# HYPOTHESIS: unverified — derived by analogy from H100 with Blackwell memory
# bandwidth; retune MBS once benchmarked.
# =============================================================================

GEMMA4_E4B_PRETRAIN_CONFIG_B200_V1 = replace(
    BASE_GEMMA4_E4B_CONFIG,
    # HYPOTHESIS: 4 GPUs (TP=4 * PP=1)
    num_gpus=4,
    # HYPOTHESIS: TP=4, PP=1 — Gemma4DenseProvider enforces PP=1 only
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    context_parallel_size=1,
    global_batch_size=512,
    micro_batch_size=1,
    cuda_graph_impl=None,
)

GEMMA4_E4B_PRETRAIN_CONFIG_B200_BF16_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_B200_V1


# =============================================================================
# B300 — HYPOTHESIS: same topology as GB200
# HYPOTHESIS: unverified — B300 assumed similar to GB200 until profiled.
# =============================================================================

GEMMA4_E4B_PRETRAIN_CONFIG_B300_V1 = replace(
    BASE_GEMMA4_E4B_CONFIG,
    # HYPOTHESIS: 4 GPUs (TP=4 * PP=1)
    num_gpus=4,
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=1,
    context_parallel_size=1,
    global_batch_size=512,
    micro_batch_size=2,
    cuda_graph_impl=None,
)

GEMMA4_E4B_PRETRAIN_CONFIG_B300_BF16_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_B300_V1


# =============================================================================
# FP8-CS (current-scaling) — Hopper and Blackwell
# Same parallelism topology as BF16; FP8 reduces memory pressure so MBS could
# increase, but conservative HYPOTHESIS configs reuse BF16 topology.
# Requires gemma4_e4b_fp8_pretrain_config() (transformer_impl="transformer_engine").
# =============================================================================

GEMMA4_E4B_PRETRAIN_CONFIG_H100_FP8_CS_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_H100_V1
GEMMA4_E4B_PRETRAIN_CONFIG_GB200_FP8_CS_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_GB200_V1
GEMMA4_E4B_PRETRAIN_CONFIG_GB300_FP8_CS_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_GB300_V1
GEMMA4_E4B_PRETRAIN_CONFIG_B200_FP8_CS_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_B200_V1
GEMMA4_E4B_PRETRAIN_CONFIG_B300_FP8_CS_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_B300_V1


# =============================================================================
# FP8-MX (MXFP8 block-scaling) — Blackwell only (SM100+)
# H100 does not support MXFP8; no H100 config here.
# =============================================================================

GEMMA4_E4B_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_GB200_V1
GEMMA4_E4B_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_GB300_V1
GEMMA4_E4B_PRETRAIN_CONFIG_B200_FP8_MX_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_B200_V1
GEMMA4_E4B_PRETRAIN_CONFIG_B300_FP8_MX_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_B300_V1


# =============================================================================
# NVFP4 — Blackwell only (SM100+)
# H100 does not support NVFP4; no H100 config here.
# =============================================================================

GEMMA4_E4B_PRETRAIN_CONFIG_GB200_NVFP4_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_GB200_V1
GEMMA4_E4B_PRETRAIN_CONFIG_GB300_NVFP4_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_GB300_V1
GEMMA4_E4B_PRETRAIN_CONFIG_B200_NVFP4_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_B200_V1
GEMMA4_E4B_PRETRAIN_CONFIG_B300_NVFP4_V1 = GEMMA4_E4B_PRETRAIN_CONFIG_B300_V1


__all__ = [
    # BF16 (Hopper and Blackwell)
    "GEMMA4_E4B_PRETRAIN_CONFIG_H100_BF16_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_B200_BF16_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_B300_BF16_V1",
    # FP8-CS (Hopper and Blackwell)
    "GEMMA4_E4B_PRETRAIN_CONFIG_H100_FP8_CS_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_GB200_FP8_CS_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_GB300_FP8_CS_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_B200_FP8_CS_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_B300_FP8_CS_V1",
    # FP8-MX (Blackwell only)
    "GEMMA4_E4B_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    # NVFP4 (Blackwell only)
    "GEMMA4_E4B_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "GEMMA4_E4B_PRETRAIN_CONFIG_B300_NVFP4_V1",
]
