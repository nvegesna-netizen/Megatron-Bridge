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

import logging

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend


logger = logging.getLogger(__name__)

# HF model id registered by MiniMaxM2Bridge via
# @MegatronModelBridge.register_bridge(source="MiniMaxM2ForCausalLM", ...)
# and documented in the bridge docstring example.
_HF_MODEL_ID = "MiniMaxAI/MiniMax-M2"


def minimax_m2_229b_a10b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for MiniMaxAI/MiniMax-M2 (229B total, ~10B active).

    MiniMax-M2 is a sparse MoE model with 256 experts and top-8 sigmoid routing
    with expert bias correction. The bridge (``MiniMaxM2Bridge``) sets:

    - ``moe_router_score_function="sigmoid"`` — MiniMax-M2 uses sigmoid (not softmax)
      to score experts before top-k selection.
    - ``moe_router_enable_expert_bias=True`` — e_score_correction_bias applied per
      expert before top-k, same pattern as DeepSeek-V3.
    - ``moe_router_pre_softmax=False`` — no pre-softmax normalization (sigmoid path).
    - ``moe_router_load_balancing_type="aux_loss"`` — auxiliary loss balancing;
      coefficient read from ``hf_config.router_aux_loss_coef``.
    - ``moe_token_dispatcher_type="alltoall"`` — alltoall dispatcher required for EP.
    - ``moe_permute_fusion=True`` — fused permute kernels for the alltoall path.
    - ``moe_grouped_gemm=True`` — grouped GEMM for expert linear layers.
    - ``qk_layernorm=True`` with a custom ``minimax_m2_layer_spec`` — full-dimension
      RMSNorm applied to Q/K over ``num_heads * head_dim`` before head splitting,
      unlike Megatron's standard per-head norm.

    MTP (Multi-Token Prediction) is NOT configured: ``MiniMaxM2Bridge`` explicitly
    documents that MTP modules are not mapped, so no ``mtp_num_layers`` is set here.

    No shared experts are mapped in ``MiniMaxM2Bridge.mapping_registry()``; therefore
    ``moe_shared_expert_overlap=False``.

    CP is set to 1: MiniMax-M2 uses standard GQA attention (not MLA), but with 256
    experts and EP=16 the primary scaling dimension is expert parallelism. Context
    parallelism over 8 ranks would require additional sequence-dimension all-gather
    bandwidth that is better spent on EP communication at this scale.

    Recommended parallelism: TP=1, PP=2, CP=1, EP=16.

    Returns:
        ConfigContainer with all fields set for MiniMax-M2 pre-training.
    """
    cfg = _pretrain_common()

    # Model — bridge registered as source="MiniMaxM2ForCausalLM", model_type="minimax_m2"
    cfg.model = AutoBridge.from_hf_pretrained(_HF_MODEL_ID).to_megatron_provider(load_weights=False)

    # Tokenizer — MiniMax-M2 ships a HuggingFace-compatible tokenizer on the hub
    cfg.tokenizer.tokenizer_model = _HF_MODEL_ID

    # Dataset — mock data by default; override blend with real data paths when needed
    cfg.dataset.blend = None  # Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Parallelism settings
    # TP=1: full-dimension QK norm (FullDimQNorm/FullDimKNorm) is applied over
    # num_heads * head_dim before TP-axis head splitting; TP>1 computes the norm
    # over only num_heads/TP heads, producing incorrect values.
    # PP=2: 62 transformer layers; Megatron asserts num_layers % PP == 0
    # (transformer_block.py:149). Divisors of 62: 1, 2, 31. PP=2 is the only
    # practical choice.
    # EP=16: NVBug 6104911 (DeepEP timeout at EP=32 for MiniMax-M2.1).
    # 256 experts / EP=16 = 16 experts per rank.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # MoE token dispatcher — alltoall set by bridge; flex dispatcher may override
    # moe_token_dispatcher_type when apply_flex_dispatcher_backend is called below.
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # Options: None, deepep, hybridep
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA graph — disabled; MoE dynamic routing shapes are incompatible with full capture
    cfg.model.cuda_graph_impl = "none"

    # Kernel selections
    cfg.model.attention_backend = None  # auto selection
    cfg.model.moe_router_fusion = False
    # moe_permute_fusion also set by bridge provider_bridge(); echoed here for explicitness
    cfg.model.moe_permute_fusion = True
    # moe_grouped_gemm also set by bridge provider_bridge()
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute)
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 — MiniMax-M2 checkpoint uses FP8 block-wise storage; the bridge dequantizes
    # on load (maybe_modify_loaded_hf_weight). Training in BF16 is the default path.
    cfg.model.moe_router_padding_for_fp8 = False

    # Optimizer — standard float32 moments; no precision-aware optimizer needed for bf16
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # No shared expert in MiniMax-M2 — mapping_registry() maps only block_sparse_moe
    # experts (w1/w3/w2), router weight, and router expert_bias. No shared expert keys.
    cfg.model.moe_shared_expert_overlap = False

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False

    # MoE force load balancing — disabled; aux_loss handles balancing during training
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg
