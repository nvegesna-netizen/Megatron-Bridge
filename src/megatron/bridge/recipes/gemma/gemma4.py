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

"""Gemma 4 Dense (E4B) pre-training recipe."""

import os
from contextlib import contextmanager

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer


_GEMMA4_E4B_HF_PATH = "google/gemma-4-E4B-it"


@contextmanager
def _gemma4_text_conversion_mode():
    previous_mode = os.environ.get("GEMMA4_CONVERSION_MODE")
    os.environ["GEMMA4_CONVERSION_MODE"] = "text"
    try:
        yield
    finally:
        if previous_mode is None:
            os.environ.pop("GEMMA4_CONVERSION_MODE", None)
        else:
            os.environ["GEMMA4_CONVERSION_MODE"] = previous_mode


def gemma4_e4b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Gemma 4 E4B (Dense, ~3.8B parameters).

    Architecture (Gemma 4 E4B):
    - 42 layers, hidden_size=2560, ffn_hidden_size=10240
    - 8 attention heads, 2 KV heads (sliding), 2 KV heads (global, head_dim=512)
    - Sliding-window / global attention interleaved (skip_freq=6)
    - Dual RoPE: sliding θ=10 000, global θ=1 000 000 with 0.25 partial rotation
    - Per-Layer Embeddings (PLE, vocab=262144, dim=256)
    - Shared KV cache across the last 18 layers
    - Local (non-TE) transformer spec via ``get_gemma4_layer_spec``

    Default parallelism: TP=2, PP=1, seq_length=4096.
    Override at launch time with Hydra-style args, e.g.::

        checkpoint.pretrained_checkpoint=/path/to/megatron-ckpt
        checkpoint.save=/path/to/save
        train.train_iters=1000
        model.seq_length=4096
    """
    cfg = _pretrain_common()

    # gemma-4-E4B-it is a ConditionalGeneration HF model; force the text-only
    # Gemma4 bridge path so this pre-training recipe uses Gemma4DenseProvider.
    with _gemma4_text_conversion_mode():
        cfg.model = AutoBridge.from_hf_pretrained(_GEMMA4_E4B_HF_PATH).to_megatron_provider(load_weights=False)

    # Tokenizer — NullTokenizer for mock pre-training; override for real data
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset — mock data by default; override dataset.blend for real data
    cfg.dataset.blend = None
    cfg.dataset.seq_length = 4096

    # Parallelism: TP=2 to match the E4B parity / conversion setup
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096

    # Training
    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.validation.eval_interval = 200
    cfg.validation.eval_iters = 10

    cfg.scheduler.lr_warmup_iters = 100

    # Implementation — Dense E4B uses the local (non-TE) spec
    cfg.model.transformer_impl = "local"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel / fusion settings — disable TE-specific fusions for the local spec
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.masked_softmax_fusion = False
    cfg.model.gradient_accumulation_fusion = False

    # Memory saving (disabled; enable recompute for larger batches)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # DDP
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def gemma4_e4b_fp8_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Gemma 4 E4B with FP8/NVFP4 via the TE spec.

    Builds on :func:`gemma4_e4b_pretrain_config` and switches
    ``transformer_impl`` from ``"local"`` to ``"transformer_engine"``.  This
    activates :func:`get_gemma4_te_layer_spec` in ``Gemma4DenseProvider.build()``,
    which replaces every ``ColumnParallelLinear`` / ``RowParallelLinear`` with
    ``TEColumnParallelLinear`` / ``TERowParallelLinear`` and swaps
    ``DotProductAttention`` for ``Gemma4TEDotProductAttentionDense``.  The TE linear
    layers participate in the FP8 autocast context and are quantized; attention
    GEMMs remain in BF16 (TE's default).

    Precision is controlled at launch time via the WBC config or ``--precision``
    argument.  Supported values::

        bf16          — standard BF16 (TE spec, no FP8 quantisation)
        fp8_cs        — FP8 current-scaling (Hopper and Blackwell)
        fp8_mx        — MXFP8 (Blackwell only, SM100+)
        nvfp4         — NVFP4 (Blackwell only, SM100+)

    Compared to :func:`gemma4_e4b_pretrain_config`:

    * ``transformer_impl = "transformer_engine"``
    * ``gradient_accumulation_fusion = True``   — TE supports weight-grad fusion
    * ``masked_softmax_fusion = False``          — TE handles attention internally
    * ``cross_entropy_fusion_impl`` stays ``"native"`` — the TE CE kernel has
      known stability issues (MCore ``model_parallel_config.py`` UserWarning)

    Returns:
        ConfigContainer with all fields set for Gemma 4 E4B FP8 pre-training.
    """
    cfg = gemma4_e4b_pretrain_config()

    # Switch from the local (non-TE) spec to the TE spec so that
    # TEColumnParallelLinear / TERowParallelLinear replace ColumnParallelLinear /
    # RowParallelLinear in every transformer block.  FP8 quantisation hooks only
    # fire on TE-wrapped linears; the local spec silently falls back to BF16.
    cfg.model.transformer_impl = "transformer_engine"

    # TE supports fused weight-gradient accumulation; not available in local spec.
    cfg.model.gradient_accumulation_fusion = True

    # Keep cross_entropy_fusion_impl="native" from the base recipe.  The TE
    # cross-entropy kernel (impl="te") has known stability issues flagged by
    # MCore (UserWarning in model_parallel_config.py) and is not needed here.

    # TE handles the attention kernel internally; the native masked-softmax
    # fusion is not needed and can conflict with TE's own attention dispatch.
    cfg.model.masked_softmax_fusion = False

    return cfg
