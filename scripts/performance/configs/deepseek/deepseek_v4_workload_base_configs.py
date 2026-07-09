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

"""DeepSeek-V4-Flash workload base configs (JET-LLM perf patch).

Naming/lookup convention mirrors ``deepseek_workload_base_configs.py`` (V3):
constant name = ``{MODEL}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}`` (uppercase),
resolved by ``get_workload_base_config(model_family_name="deepseek",
model_recipe_name="deepseek_v4", gpu, compute_dtype, "pretrain", config_variant)``.

These are a conservative, FUNCTIONAL-FIRST V1 starting point — NOT yet perf-tuned.
They intentionally match the recipe's Blackwell baseline so the golden path runs
without surprises; perf engineers should iterate (v1/v2, dispatcher, CUDA graphs,
recompute) once the DSv4-capable container is available.

Baseline: TP=1, PP=4, EP=8, ETP=1, VPP=None, CP=1, GBS=128, MBS=1, hybridep.
DeepSeek-V4 places hash-routed MoE in the first ``moe_n_hash_layers`` layers, which
must co-locate with the embedding on PP stage 0; PP=4 with VPP=None keeps that
layout simple (see set_deepseek_v4_pipeline_model_parallel_layout).
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig

BASE_DEEPSEEK_V4_CONFIG = WorkloadBaseConfig(
    num_gpus=256,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=4,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    global_batch_size=128,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_impl=None,
    cuda_graph_scope=None,
    recompute_modules=["moe_act"],
)

# Per-GPU bases (currently identical — distinct slots so future per-GPU tuning is
# a one-line ``replace`` without touching the dtype aliases below).
DEEPSEEK_V4_PRETRAIN_CONFIG_GB300_V1 = replace(BASE_DEEPSEEK_V4_CONFIG)
DEEPSEEK_V4_PRETRAIN_CONFIG_GB200_V1 = replace(BASE_DEEPSEEK_V4_CONFIG)
DEEPSEEK_V4_PRETRAIN_CONFIG_B300_V1 = replace(BASE_DEEPSEEK_V4_CONFIG)
DEEPSEEK_V4_PRETRAIN_CONFIG_B200_V1 = replace(BASE_DEEPSEEK_V4_CONFIG)

# Per-dtype aliases (BF16 and FP8-MX share parallelism; precision specifics are
# applied by the builder via get_precision_config + the recipe's quant recipe).
DEEPSEEK_V4_PRETRAIN_CONFIG_GB300_BF16_V1 = DEEPSEEK_V4_PRETRAIN_CONFIG_GB300_V1
DEEPSEEK_V4_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = DEEPSEEK_V4_PRETRAIN_CONFIG_GB300_V1
DEEPSEEK_V4_PRETRAIN_CONFIG_GB200_BF16_V1 = DEEPSEEK_V4_PRETRAIN_CONFIG_GB200_V1
DEEPSEEK_V4_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = DEEPSEEK_V4_PRETRAIN_CONFIG_GB200_V1
DEEPSEEK_V4_PRETRAIN_CONFIG_B300_BF16_V1 = DEEPSEEK_V4_PRETRAIN_CONFIG_B300_V1
DEEPSEEK_V4_PRETRAIN_CONFIG_B300_FP8_MX_V1 = DEEPSEEK_V4_PRETRAIN_CONFIG_B300_V1
DEEPSEEK_V4_PRETRAIN_CONFIG_B200_BF16_V1 = DEEPSEEK_V4_PRETRAIN_CONFIG_B200_V1
DEEPSEEK_V4_PRETRAIN_CONFIG_B200_FP8_MX_V1 = DEEPSEEK_V4_PRETRAIN_CONFIG_B200_V1


__all__ = [
    "BASE_DEEPSEEK_V4_CONFIG",
    "DEEPSEEK_V4_PRETRAIN_CONFIG_GB300_BF16_V1",
    "DEEPSEEK_V4_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "DEEPSEEK_V4_PRETRAIN_CONFIG_GB200_BF16_V1",
    "DEEPSEEK_V4_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "DEEPSEEK_V4_PRETRAIN_CONFIG_B300_BF16_V1",
    "DEEPSEEK_V4_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "DEEPSEEK_V4_PRETRAIN_CONFIG_B200_BF16_V1",
    "DEEPSEEK_V4_PRETRAIN_CONFIG_B200_FP8_MX_V1",
]
