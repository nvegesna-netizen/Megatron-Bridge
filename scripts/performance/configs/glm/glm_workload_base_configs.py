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

"""Workload base presets for GLM performance configs.

GLM-5.2 uses Dynamic Sparse Attention; Megatron-Core currently requires CP=1 and
RoPE fusion disabled for DSA, so those presets keep context_parallel_size=1.

GLM-5.2 PP divisibility: the HF config for zai-org/GLM-5.2 reports
num_hidden_layers=78.  78 is not divisible by 8 or 16, so all GLM-5.2 WBCs
carry an explicit pp_layout string that distributes the 78 layers non-uniformly
across pipeline stages:

  PP=8  (GB300/GB200): 6 stages of 10 layers + 2 stages of 9 layers.
    layout: Et*10|(t*10|)*5t*9|t*9L   (78 transformer layers total, 8 stages)

  PP=16 (B300/B200/H100): 14 stages of 5 layers + 2 stages of 4 layers.
    layout: Et*4|(t*5|)*14t*4L         (78 transformer layers total, 16 stages)

MCore validates that the count of LayerType.decoder entries equals
num_hidden_layers; E and L tokens are ignored from that count.

GLM-4.5 (355B-A32B) and GLM-4.5-Air (106B-A12B) use standard attention (GQA +
QK-norm) with the deepep MoE dispatcher.  These are v1 baseline configs; per-GPU
kernel tuning will land in v2 once throughput benchmarks are available.

"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_GLM52_753B_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
    context_parallel_size=1,
)


GLM52_753B_PRETRAIN_CONFIG_GB300 = replace(
    BASE_GLM52_753B_CONFIG,
    num_gpus=256,
    global_batch_size=4096,
    micro_batch_size=1,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=None,
    # 78 layers / 8 stages: 6 stages × 10 layers + 2 stages × 9 layers = 78
    pp_layout="Et*10|(t*10|)*5t*9|t*9L",
    # EP=16: max valid for TP=2, PP=8, 256 GPUs — TP×PP×EP = 2×8×16 = 256 = num_gpus.
    # DP = 256/16 = 16; EP=16 ≤ DP=16. 256 routed experts / EP=16 = 16 experts per rank.
    expert_model_parallel_size=16,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
    recompute_modules=["mla_up_proj"],
)
GLM52_753B_PRETRAIN_CONFIG_GB300_BF16 = GLM52_753B_PRETRAIN_CONFIG_GB300
GLM52_753B_PRETRAIN_CONFIG_GB300_FP8_CS = GLM52_753B_PRETRAIN_CONFIG_GB300
GLM52_753B_PRETRAIN_CONFIG_GB300_FP8_MX = replace(
    GLM52_753B_PRETRAIN_CONFIG_GB300,
    moe_a2a_overlap=True,
    cutedsl_fused_grouped_mlp=True,
)
GLM52_753B_PRETRAIN_CONFIG_GB300_NVFP4 = replace(
    GLM52_753B_PRETRAIN_CONFIG_GB300,
    moe_a2a_overlap=True,
    cutedsl_fused_grouped_mlp=True,
)


GLM52_753B_PRETRAIN_CONFIG_GB200 = replace(
    BASE_GLM52_753B_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    micro_batch_size=1,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=None,
    # 78 layers / 8 stages: 6 stages × 10 layers + 2 stages × 9 layers = 78
    pp_layout="Et*10|(t*10|)*5t*9|t*9L",
    # EP=16: max valid for TP=2, PP=8, 256 GPUs — TP×PP×EP = 2×8×16 = 256 = num_gpus.
    # DP = 256/16 = 16; EP=16 ≤ DP=16. 256 routed experts / EP=16 = 16 experts per rank.
    expert_model_parallel_size=16,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
    recompute_modules=["mla_up_proj"],
)
GLM52_753B_PRETRAIN_CONFIG_GB200_BF16 = GLM52_753B_PRETRAIN_CONFIG_GB200
GLM52_753B_PRETRAIN_CONFIG_GB200_FP8_CS = GLM52_753B_PRETRAIN_CONFIG_GB200
GLM52_753B_PRETRAIN_CONFIG_GB200_FP8_MX = replace(
    GLM52_753B_PRETRAIN_CONFIG_GB200,
    moe_a2a_overlap=True,
    cutedsl_fused_grouped_mlp=True,
)
GLM52_753B_PRETRAIN_CONFIG_GB200_NVFP4 = replace(
    GLM52_753B_PRETRAIN_CONFIG_GB200,
    moe_a2a_overlap=True,
    cutedsl_fused_grouped_mlp=True,
)


GLM52_753B_PRETRAIN_CONFIG_B300 = replace(
    BASE_GLM52_753B_CONFIG,
    num_gpus=256,
    global_batch_size=4096,
    micro_batch_size=1,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=16,
    virtual_pipeline_model_parallel_size=None,
    # 78 layers / 16 stages: 14 stages × 5 layers + 2 stages × 4 layers = 78
    pp_layout="Et*4|(t*5|)*14t*4L",
    # EP=8: TP=2, PP=16, 256 GPUs — DP = 256/(2×16) = 8; EP=8 ≤ DP=8. DP%EP = 0 ✓
    expert_model_parallel_size=8,
    moe_a2a_overlap=False,
    recompute_modules=["mla_up_proj"],
)
GLM52_753B_PRETRAIN_CONFIG_B300_BF16 = GLM52_753B_PRETRAIN_CONFIG_B300
GLM52_753B_PRETRAIN_CONFIG_B300_FP8_CS = GLM52_753B_PRETRAIN_CONFIG_B300
GLM52_753B_PRETRAIN_CONFIG_B300_FP8_MX = GLM52_753B_PRETRAIN_CONFIG_B300
GLM52_753B_PRETRAIN_CONFIG_B300_NVFP4 = GLM52_753B_PRETRAIN_CONFIG_B300


GLM52_753B_PRETRAIN_CONFIG_B200 = replace(
    BASE_GLM52_753B_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    micro_batch_size=1,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=16,
    virtual_pipeline_model_parallel_size=None,
    # 78 layers / 16 stages: 14 stages × 5 layers + 2 stages × 4 layers = 78
    pp_layout="Et*4|(t*5|)*14t*4L",
    # EP=8: TP=2, PP=16, 256 GPUs — DP = 256/(2×16) = 8; EP=8 ≤ DP=8. DP%EP = 0 ✓
    expert_model_parallel_size=8,
    moe_a2a_overlap=False,
    recompute_modules=["mla_up_proj"],
)
GLM52_753B_PRETRAIN_CONFIG_B200_BF16 = GLM52_753B_PRETRAIN_CONFIG_B200
GLM52_753B_PRETRAIN_CONFIG_B200_FP8_CS = GLM52_753B_PRETRAIN_CONFIG_B200
GLM52_753B_PRETRAIN_CONFIG_B200_FP8_MX = GLM52_753B_PRETRAIN_CONFIG_B200
GLM52_753B_PRETRAIN_CONFIG_B200_NVFP4 = GLM52_753B_PRETRAIN_CONFIG_B200


GLM52_753B_PRETRAIN_CONFIG_H100 = replace(
    BASE_GLM52_753B_CONFIG,
    num_gpus=1024,
    global_batch_size=8192,
    micro_batch_size=1,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=16,
    virtual_pipeline_model_parallel_size=None,
    # 78 layers / 16 stages: 14 stages × 5 layers + 2 stages × 4 layers = 78
    pp_layout="Et*4|(t*5|)*14t*4L",
    # EP=8: DP = 1024/(8*16) = 8; EP=8 ≤ DP=8, DP%EP=0 ✓.
    # 256 routed experts / EP=8 = 32 experts per EP rank.
    expert_model_parallel_size=8,
    moe_a2a_overlap=False,
    recompute_modules=["mla_up_proj", "mlp"],
)
GLM52_753B_PRETRAIN_CONFIG_H100_BF16 = GLM52_753B_PRETRAIN_CONFIG_H100
GLM52_753B_PRETRAIN_CONFIG_H100_FP8_CS = GLM52_753B_PRETRAIN_CONFIG_H100
GLM52_753B_PRETRAIN_CONFIG_H100_FP8_SC = GLM52_753B_PRETRAIN_CONFIG_H100


__all__ = [
    "GLM52_753B_PRETRAIN_CONFIG_GB300_BF16",
    "GLM52_753B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "GLM52_753B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "GLM52_753B_PRETRAIN_CONFIG_GB300_NVFP4",
    "GLM52_753B_PRETRAIN_CONFIG_GB200_BF16",
    "GLM52_753B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "GLM52_753B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "GLM52_753B_PRETRAIN_CONFIG_GB200_NVFP4",
    "GLM52_753B_PRETRAIN_CONFIG_B300_BF16",
    "GLM52_753B_PRETRAIN_CONFIG_B300_FP8_CS",
    "GLM52_753B_PRETRAIN_CONFIG_B300_FP8_MX",
    "GLM52_753B_PRETRAIN_CONFIG_B300_NVFP4",
    "GLM52_753B_PRETRAIN_CONFIG_B200_BF16",
    "GLM52_753B_PRETRAIN_CONFIG_B200_FP8_CS",
    "GLM52_753B_PRETRAIN_CONFIG_B200_FP8_MX",
    "GLM52_753B_PRETRAIN_CONFIG_B200_NVFP4",
    "GLM52_753B_PRETRAIN_CONFIG_H100_BF16",
    "GLM52_753B_PRETRAIN_CONFIG_H100_FP8_CS",
    "GLM52_753B_PRETRAIN_CONFIG_H100_FP8_SC",
    # GLM-4.5 355B-A32B
    "GLM45_355B_PRETRAIN_CONFIG_GB300_BF16",
    "GLM45_355B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "GLM45_355B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "GLM45_355B_PRETRAIN_CONFIG_GB300_NVFP4",
    "GLM45_355B_PRETRAIN_CONFIG_GB200_BF16",
    "GLM45_355B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "GLM45_355B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "GLM45_355B_PRETRAIN_CONFIG_GB200_NVFP4",
    "GLM45_355B_PRETRAIN_CONFIG_B300_BF16",
    "GLM45_355B_PRETRAIN_CONFIG_B300_FP8_CS",
    "GLM45_355B_PRETRAIN_CONFIG_B300_FP8_MX",
    "GLM45_355B_PRETRAIN_CONFIG_B300_NVFP4",
    "GLM45_355B_PRETRAIN_CONFIG_B200_BF16",
    "GLM45_355B_PRETRAIN_CONFIG_B200_FP8_CS",
    "GLM45_355B_PRETRAIN_CONFIG_B200_FP8_MX",
    "GLM45_355B_PRETRAIN_CONFIG_B200_NVFP4",
    "GLM45_355B_PRETRAIN_CONFIG_H100_BF16",
    "GLM45_355B_PRETRAIN_CONFIG_H100_FP8_CS",
    # GLM-4.5-Air 106B-A12B
    "GLM45_AIR_106B_PRETRAIN_CONFIG_GB300_BF16",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_GB300_NVFP4",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_GB200_BF16",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_GB200_NVFP4",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_B300_BF16",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_B300_FP8_CS",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_B300_FP8_MX",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_B300_NVFP4",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_B200_BF16",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_B200_FP8_CS",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_B200_FP8_MX",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_B200_NVFP4",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_H100_BF16",
    "GLM45_AIR_106B_PRETRAIN_CONFIG_H100_FP8_CS",
]


# ─────────────────────────────────────────────────────────────────────────────
# GLM-4.5 355B-A32B (MoE + GQA + QK-norm, deepep dispatcher)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture: 92 layers (3 dense + 89 MoE), hidden=5120, 160 experts/layer,
# top-8 routing, 1 shared expert, moe_intermediate_size=1536.
#
# PP=4 is the highest clean pipeline depth for 92 layers (92 / 4 = 23 exactly).
# PP=8 (the base recipe default) requires num_layers to be divisible by 8; if
# the actual deployed model has 88 or 96 layers this would also work, but is not
# confirmed here.  VP is omitted because no (PP, VP) pair with PP in {4,8} and
# VP > 1 divides 92 evenly.  Override via `megatron_bridge.pp` / `vp` sweep
# fields once the exact layer count is confirmed.
#
# With 256 GPUs, TP=2, PP=4: DP = 256/(2*4) = 32; EP=16 ≤ DP=32 ✓.
# H100: adds selective MLP recompute to contain 355B activation memory in 80 GB.

BASE_GLM45_355B_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)

GLM45_355B_PRETRAIN_CONFIG_GB300 = replace(
    BASE_GLM45_355B_CONFIG,
    num_gpus=256,
    global_batch_size=4096,
    micro_batch_size=1,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=16,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
)
GLM45_355B_PRETRAIN_CONFIG_GB300_BF16 = GLM45_355B_PRETRAIN_CONFIG_GB300
GLM45_355B_PRETRAIN_CONFIG_GB300_FP8_CS = GLM45_355B_PRETRAIN_CONFIG_GB300
GLM45_355B_PRETRAIN_CONFIG_GB300_FP8_MX = GLM45_355B_PRETRAIN_CONFIG_GB300
GLM45_355B_PRETRAIN_CONFIG_GB300_NVFP4 = GLM45_355B_PRETRAIN_CONFIG_GB300


GLM45_355B_PRETRAIN_CONFIG_GB200 = replace(
    BASE_GLM45_355B_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    micro_batch_size=1,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=16,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
)
GLM45_355B_PRETRAIN_CONFIG_GB200_BF16 = GLM45_355B_PRETRAIN_CONFIG_GB200
GLM45_355B_PRETRAIN_CONFIG_GB200_FP8_CS = GLM45_355B_PRETRAIN_CONFIG_GB200
GLM45_355B_PRETRAIN_CONFIG_GB200_FP8_MX = GLM45_355B_PRETRAIN_CONFIG_GB200
GLM45_355B_PRETRAIN_CONFIG_GB200_NVFP4 = GLM45_355B_PRETRAIN_CONFIG_GB200


GLM45_355B_PRETRAIN_CONFIG_B300 = replace(
    BASE_GLM45_355B_CONFIG,
    num_gpus=256,
    global_batch_size=4096,
    micro_batch_size=1,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=16,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
)
GLM45_355B_PRETRAIN_CONFIG_B300_BF16 = GLM45_355B_PRETRAIN_CONFIG_B300
GLM45_355B_PRETRAIN_CONFIG_B300_FP8_CS = GLM45_355B_PRETRAIN_CONFIG_B300
GLM45_355B_PRETRAIN_CONFIG_B300_FP8_MX = GLM45_355B_PRETRAIN_CONFIG_B300
GLM45_355B_PRETRAIN_CONFIG_B300_NVFP4 = GLM45_355B_PRETRAIN_CONFIG_B300


GLM45_355B_PRETRAIN_CONFIG_B200 = replace(
    BASE_GLM45_355B_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    micro_batch_size=1,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=16,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
)
GLM45_355B_PRETRAIN_CONFIG_B200_BF16 = GLM45_355B_PRETRAIN_CONFIG_B200
GLM45_355B_PRETRAIN_CONFIG_B200_FP8_CS = GLM45_355B_PRETRAIN_CONFIG_B200
GLM45_355B_PRETRAIN_CONFIG_B200_FP8_MX = GLM45_355B_PRETRAIN_CONFIG_B200
GLM45_355B_PRETRAIN_CONFIG_B200_NVFP4 = GLM45_355B_PRETRAIN_CONFIG_B200


GLM45_355B_PRETRAIN_CONFIG_H100 = replace(
    BASE_GLM45_355B_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    micro_batch_size=1,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    expert_model_parallel_size=16,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
    recompute_modules=["mlp"],
)
GLM45_355B_PRETRAIN_CONFIG_H100_BF16 = GLM45_355B_PRETRAIN_CONFIG_H100
GLM45_355B_PRETRAIN_CONFIG_H100_FP8_CS = GLM45_355B_PRETRAIN_CONFIG_H100


# ─────────────────────────────────────────────────────────────────────────────
# GLM-4.5-Air 106B-A12B (same architecture, smaller scale)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture: 46 layers (1 dense + 45 MoE), hidden=4096, 128 experts/layer,
# top-8 routing (or top-9 per some sources), 1 shared expert.
#
# PP=2 divides 46 layers cleanly (46/2 = 23).  The base recipe uses PP=4; if
# the actual num_hidden_layers is 44 or 48 (divisible by 4), override with
# `megatron_bridge.pp: [4]` in the sweep for higher pipeline efficiency.
# With 64 GPUs, TP=1, PP=2: DP = 64/(1*2) = 32; EP=8 ≤ DP=32 ✓.

BASE_GLM45_AIR_106B_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)

GLM45_AIR_106B_PRETRAIN_CONFIG_GB300 = replace(
    BASE_GLM45_AIR_106B_CONFIG,
    num_gpus=64,
    global_batch_size=2048,
    micro_batch_size=2,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
)
GLM45_AIR_106B_PRETRAIN_CONFIG_GB300_BF16 = GLM45_AIR_106B_PRETRAIN_CONFIG_GB300
GLM45_AIR_106B_PRETRAIN_CONFIG_GB300_FP8_CS = GLM45_AIR_106B_PRETRAIN_CONFIG_GB300
GLM45_AIR_106B_PRETRAIN_CONFIG_GB300_FP8_MX = GLM45_AIR_106B_PRETRAIN_CONFIG_GB300
GLM45_AIR_106B_PRETRAIN_CONFIG_GB300_NVFP4 = GLM45_AIR_106B_PRETRAIN_CONFIG_GB300


GLM45_AIR_106B_PRETRAIN_CONFIG_GB200 = replace(
    BASE_GLM45_AIR_106B_CONFIG,
    num_gpus=64,
    global_batch_size=2048,
    micro_batch_size=1,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
)
GLM45_AIR_106B_PRETRAIN_CONFIG_GB200_BF16 = GLM45_AIR_106B_PRETRAIN_CONFIG_GB200
GLM45_AIR_106B_PRETRAIN_CONFIG_GB200_FP8_CS = GLM45_AIR_106B_PRETRAIN_CONFIG_GB200
GLM45_AIR_106B_PRETRAIN_CONFIG_GB200_FP8_MX = GLM45_AIR_106B_PRETRAIN_CONFIG_GB200
GLM45_AIR_106B_PRETRAIN_CONFIG_GB200_NVFP4 = GLM45_AIR_106B_PRETRAIN_CONFIG_GB200


GLM45_AIR_106B_PRETRAIN_CONFIG_B300 = replace(
    BASE_GLM45_AIR_106B_CONFIG,
    num_gpus=64,
    global_batch_size=2048,
    micro_batch_size=2,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
)
GLM45_AIR_106B_PRETRAIN_CONFIG_B300_BF16 = GLM45_AIR_106B_PRETRAIN_CONFIG_B300
GLM45_AIR_106B_PRETRAIN_CONFIG_B300_FP8_CS = GLM45_AIR_106B_PRETRAIN_CONFIG_B300
GLM45_AIR_106B_PRETRAIN_CONFIG_B300_FP8_MX = GLM45_AIR_106B_PRETRAIN_CONFIG_B300
GLM45_AIR_106B_PRETRAIN_CONFIG_B300_NVFP4 = GLM45_AIR_106B_PRETRAIN_CONFIG_B300


GLM45_AIR_106B_PRETRAIN_CONFIG_B200 = replace(
    BASE_GLM45_AIR_106B_CONFIG,
    num_gpus=64,
    global_batch_size=2048,
    micro_batch_size=1,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
)
GLM45_AIR_106B_PRETRAIN_CONFIG_B200_BF16 = GLM45_AIR_106B_PRETRAIN_CONFIG_B200
GLM45_AIR_106B_PRETRAIN_CONFIG_B200_FP8_CS = GLM45_AIR_106B_PRETRAIN_CONFIG_B200
GLM45_AIR_106B_PRETRAIN_CONFIG_B200_FP8_MX = GLM45_AIR_106B_PRETRAIN_CONFIG_B200
GLM45_AIR_106B_PRETRAIN_CONFIG_B200_NVFP4 = GLM45_AIR_106B_PRETRAIN_CONFIG_B200


GLM45_AIR_106B_PRETRAIN_CONFIG_H100 = replace(
    BASE_GLM45_AIR_106B_CONFIG,
    num_gpus=64,
    global_batch_size=2048,
    micro_batch_size=1,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=2,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="deepep",
    moe_a2a_overlap=False,
)
GLM45_AIR_106B_PRETRAIN_CONFIG_H100_BF16 = GLM45_AIR_106B_PRETRAIN_CONFIG_H100
GLM45_AIR_106B_PRETRAIN_CONFIG_H100_FP8_CS = GLM45_AIR_106B_PRETRAIN_CONFIG_H100
