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

"""Workload base presets for Kimi-K2 performance configs."""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_KIMI_K2_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


KIMI_K2_PRETRAIN_CONFIG_GB300 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=256,
    global_batch_size=4096,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    micro_batch_size=2,
    cuda_graph_scope=[],
    recompute_modules=["mla_up_proj"],
)
KIMI_K2_PRETRAIN_CONFIG_GB300_BF16 = KIMI_K2_PRETRAIN_CONFIG_GB300
KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_GB300
KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_GB300
KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4 = KIMI_K2_PRETRAIN_CONFIG_GB300


KIMI_K2_PRETRAIN_CONFIG_GB200 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    recompute_modules=["mla_up_proj"],
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)
KIMI_K2_PRETRAIN_CONFIG_GB200_BF16 = KIMI_K2_PRETRAIN_CONFIG_GB200
KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_GB200
KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_GB200

KIMI_K2_PRETRAIN_CONFIG_VR200_BF16 = KIMI_K2_PRETRAIN_CONFIG_GB200_BF16
KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX


KIMI_K2_PRETRAIN_CONFIG_B300 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=16,
    global_batch_size=4096,
    micro_batch_size=2,
    recompute_modules=["mla_up_proj"],
    moe_a2a_overlap=False,
)
KIMI_K2_PRETRAIN_CONFIG_B300_BF16 = KIMI_K2_PRETRAIN_CONFIG_B300
KIMI_K2_PRETRAIN_CONFIG_B300_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_B300
KIMI_K2_PRETRAIN_CONFIG_B300_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_B300


KIMI_K2_PRETRAIN_CONFIG_B200 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=16,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
    moe_a2a_overlap=False,
)
KIMI_K2_PRETRAIN_CONFIG_B200_BF16 = KIMI_K2_PRETRAIN_CONFIG_B200
KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_B200
KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX = KIMI_K2_PRETRAIN_CONFIG_B200


KIMI_K2_PRETRAIN_CONFIG_H100 = replace(
    BASE_KIMI_K2_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=16,
    virtual_pipeline_model_parallel_size=2,
    expert_model_parallel_size=64,
    global_batch_size=8192,
    recompute_modules=["mla_up_proj", "mlp"],
    moe_a2a_overlap=False,
    pp_layout="Et|(tt|)*30L",
)
KIMI_K2_PRETRAIN_CONFIG_H100_BF16 = KIMI_K2_PRETRAIN_CONFIG_H100
KIMI_K2_PRETRAIN_CONFIG_H100_FP8_CS = KIMI_K2_PRETRAIN_CONFIG_H100
KIMI_K2_PRETRAIN_CONFIG_H100_FP8_SC = KIMI_K2_PRETRAIN_CONFIG_H100


# =============================================================================
# Kimi K2 Pretrain - V2
# =============================================================================
# V2 naming convention follows DSv3: {MODEL}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}
# get_workload_base_config resolves "v2" -> versioned V2 name -> versioned V1 name -> unversioned.

# GB300 V2: GBS is already 4096 in the unversioned base (unlike DSv3 which started at 2048).
# cuda_graph_scope=[] was set deliberately in V1 — do not enable CUDA graphs for GB300
# without explicit validation. V2 aliases point to the same object.
KIMI_K2_PRETRAIN_CONFIG_GB300_BF16_V2 = KIMI_K2_PRETRAIN_CONFIG_GB300_BF16
KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS_V2 = KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS
KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX_V2 = KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX
KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4_V2 = KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4

# GB200 V2: bump GBS from 2048 to 4096, matching DSv3 GB200 V2.
# CUDA graphs were already enabled at V1 level
# (cuda_graph_impl="transformer_engine", scope=["moe_router", "moe_preprocess"]).
# moe_a2a_overlap and cutedsl_fused_grouped_mlp are excluded: no Kimi K2-specific
# validation exists for them at this time.
KIMI_K2_PRETRAIN_CONFIG_GB200_V2 = replace(
    KIMI_K2_PRETRAIN_CONFIG_GB200,
    global_batch_size=4096,
)
KIMI_K2_PRETRAIN_CONFIG_GB200_BF16_V2 = KIMI_K2_PRETRAIN_CONFIG_GB200_V2
KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS_V2 = KIMI_K2_PRETRAIN_CONFIG_GB200_V2
KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX_V2 = KIMI_K2_PRETRAIN_CONFIG_GB200_V2

# VR200 V2: alias GB200 V2 (same as VR200 V1 aliases GB200 V1).
# VR200 has 4 GPUs/node vs 8 for GB200; adjust num_nodes in the JET-LLM YAML accordingly.
# No VR200_FP8_CS_V2 — consistent with V1 which also has no VR200_FP8_CS entry.
KIMI_K2_PRETRAIN_CONFIG_VR200_BF16_V2 = KIMI_K2_PRETRAIN_CONFIG_GB200_BF16_V2
KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX_V2 = KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX_V2

# B300 V2: add CUDA graph coverage (cuda_graph_impl + cuda_graph_scope), matching
# DSv3 B300 V2 pattern. GBS is already 4096 in V1, so no GBS change needed.
# PP halving (16->8) from DSv3 B300 V2 is excluded: Kimi K2 uses EP=16 (vs DSv3
# B300 V1's EP=8) and the PP change is coupled with EP — no Kimi K2 validation exists.
# cuda_graph_scope uses ["moe_router", "moe_preprocess"] (NOT "attn") to match the
# scope validated in Kimi K2 GB200 V1. GB200 V1 deliberately excludes "attn" from
# cuda_graph_scope; the pending qk_clip fix (kimi_llm_pretrain.py:54) confirms an
# open MLA/attention MCORE issue. Until "attn" is confirmed safe for Kimi K2's
# attention on any Blackwell target, it is excluded here as well.
KIMI_K2_PRETRAIN_CONFIG_B300_V2 = replace(
    KIMI_K2_PRETRAIN_CONFIG_B300,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)
KIMI_K2_PRETRAIN_CONFIG_B300_BF16_V2 = KIMI_K2_PRETRAIN_CONFIG_B300_V2
KIMI_K2_PRETRAIN_CONFIG_B300_FP8_CS_V2 = KIMI_K2_PRETRAIN_CONFIG_B300_V2
KIMI_K2_PRETRAIN_CONFIG_B300_FP8_MX_V2 = KIMI_K2_PRETRAIN_CONFIG_B300_V2

# B200 V2: bump GBS from 2048 to 4096, matching DSv3 B200 V2 BF16/FP8_CS pattern.
# The deeper DSv3 B200 FP8_MX V2 changes (deepep dispatcher, PP 16->8, VP, EP->32,
# moe_a2a_overlap=True) are excluded: Kimi K2 B200 has no moe_flex_dispatcher_backend
# set in V1 and no Kimi K2-specific validation exists for that re-partitioning.
KIMI_K2_PRETRAIN_CONFIG_B200_V2 = replace(
    KIMI_K2_PRETRAIN_CONFIG_B200,
    global_batch_size=4096,
)
KIMI_K2_PRETRAIN_CONFIG_B200_BF16_V2 = KIMI_K2_PRETRAIN_CONFIG_B200_V2
KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS_V2 = KIMI_K2_PRETRAIN_CONFIG_B200_V2
KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX_V2 = KIMI_K2_PRETRAIN_CONFIG_B200_V2


__all__ = [
    "KIMI_K2_PRETRAIN_CONFIG_GB300_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_BF16_V2",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS_V2",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX_V2",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4_V2",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_BF16_V2",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS_V2",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX_V2",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_BF16_V2",
    "KIMI_K2_PRETRAIN_CONFIG_VR200_FP8_MX_V2",
    "KIMI_K2_PRETRAIN_CONFIG_B300_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_B300_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_B300_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_B300_BF16_V2",
    "KIMI_K2_PRETRAIN_CONFIG_B300_FP8_CS_V2",
    "KIMI_K2_PRETRAIN_CONFIG_B300_FP8_MX_V2",
    "KIMI_K2_PRETRAIN_CONFIG_B200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_B200_BF16_V2",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS_V2",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX_V2",
    "KIMI_K2_PRETRAIN_CONFIG_H100_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_H100_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_H100_FP8_SC",
]
