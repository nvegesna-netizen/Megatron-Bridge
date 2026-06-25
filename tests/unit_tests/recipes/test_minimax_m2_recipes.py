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

"""
Unit tests for MiniMax-M2 recipe configuration builders.

Patterned after the Step-3.5-Flash recipe tests: monkeypatch ``AutoBridge`` to a
lightweight fake provider and monkeypatch ``apply_flex_dispatcher_backend`` so the
recipe builds without HF Hub I/O or CUDA device queries.
"""

import importlib

import pytest
import torch


_minimax_m2_module = importlib.import_module("megatron.bridge.recipes.minimax_m2.minimax_m2")


class _FakeModelCfg:
    """Stand-in for the Megatron provider returned by ``AutoBridge``."""

    def __init__(self):
        # Fields the bridge reads before the recipe overwrites them.
        self.num_moe_experts = 256
        self.apply_rope_fusion = False

        # None sentinels for every field the recipe explicitly sets.
        # Using None ensures an assertion failure if the recipe skips a field.
        self.pipeline_model_parallel_layout = None
        self.tensor_model_parallel_size = None
        self.pipeline_model_parallel_size = None
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = None
        self.expert_model_parallel_size = None
        self.expert_tensor_parallel_size = None
        self.sequence_parallel = None
        self.seq_length = None
        self.moe_token_dispatcher_type = None
        self.moe_flex_dispatcher_backend = None
        self.moe_hybridep_num_sms = None
        self.moe_router_fusion = None
        self.moe_permute_fusion = None
        self.moe_grouped_gemm = None
        self.moe_shared_expert_overlap = None
        self.recompute_granularity = None
        self.recompute_method = None
        self.recompute_num_layers = None


class _FakeBridge:
    @staticmethod
    def from_hf_pretrained(hf_path: str):
        return _FakeBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()


def _patch_recipe_env(monkeypatch, mod):
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)
    monkeypatch.setattr(mod, "apply_flex_dispatcher_backend", lambda *a, **kw: None)


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_minimax_m2_229b_recipe_builds_config(monkeypatch):
    _patch_recipe_env(monkeypatch, _minimax_m2_module)

    from megatron.bridge.recipes.minimax_m2.minimax_m2 import minimax_m2_229b_a10b_pretrain_config

    cfg = minimax_m2_229b_a10b_pretrain_config()
    _assert_basic_config(cfg)


# ---------------------------------------------------------------------------
# Default parallelism and core settings
# ---------------------------------------------------------------------------


def test_minimax_m2_229b_pretrain_defaults(monkeypatch):
    """Verify the parallelism layout and core model settings shipped in the recipe."""
    _patch_recipe_env(monkeypatch, _minimax_m2_module)

    from megatron.bridge.recipes.minimax_m2.minimax_m2 import minimax_m2_229b_a10b_pretrain_config

    cfg = minimax_m2_229b_a10b_pretrain_config()
    m = cfg.model

    # Parallelism — TP=1 required by full-dim QK norm; PP=2 (62 layers, divisors 1/2/31)
    assert m.tensor_model_parallel_size == 1
    assert m.pipeline_model_parallel_size == 2
    assert m.context_parallel_size == 1
    assert m.expert_model_parallel_size == 16
    assert m.expert_tensor_parallel_size == 1
    assert m.sequence_parallel is False
    assert m.seq_length == 4096
    assert m.pipeline_dtype is torch.bfloat16

    # MoE kernel selections
    assert m.moe_grouped_gemm is True
    assert m.moe_permute_fusion is True
    assert m.moe_router_fusion is False
    assert m.moe_token_dispatcher_type == "alltoall"
    assert m.moe_flex_dispatcher_backend == "deepep"

    # No shared experts
    assert m.moe_shared_expert_overlap is False

    # Memory / recompute
    assert m.recompute_granularity == "full"
    assert m.recompute_method == "uniform"
    assert m.recompute_num_layers == 1


def test_minimax_m2_229b_recipe_ddp_defaults(monkeypatch):
    _patch_recipe_env(monkeypatch, _minimax_m2_module)

    from megatron.bridge.recipes.minimax_m2.minimax_m2 import minimax_m2_229b_a10b_pretrain_config

    cfg = minimax_m2_229b_a10b_pretrain_config()
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.use_megatron_fsdp is False
    assert cfg.ddp.overlap_grad_reduce is True
    assert cfg.ddp.overlap_param_gather is True


def test_minimax_m2_229b_recipe_uses_minimax_tokenizer(monkeypatch):
    _patch_recipe_env(monkeypatch, _minimax_m2_module)

    from megatron.bridge.recipes.minimax_m2.minimax_m2 import (
        _HF_MODEL_ID,
        minimax_m2_229b_a10b_pretrain_config,
    )

    cfg = minimax_m2_229b_a10b_pretrain_config()
    assert cfg.tokenizer.tokenizer_model == _HF_MODEL_ID


def test_minimax_m2_229b_recipe_optimizer_defaults(monkeypatch):
    """Optimizer should use float32 moments with no precision-aware optimizer."""
    _patch_recipe_env(monkeypatch, _minimax_m2_module)

    from megatron.bridge.recipes.minimax_m2.minimax_m2 import minimax_m2_229b_a10b_pretrain_config

    cfg = minimax_m2_229b_a10b_pretrain_config()
    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.optimizer.main_grads_dtype is torch.float32
    assert cfg.optimizer.main_params_dtype is torch.float32


def test_minimax_m2_229b_recipe_dataset_mock_by_default(monkeypatch):
    _patch_recipe_env(monkeypatch, _minimax_m2_module)

    from megatron.bridge.recipes.minimax_m2.minimax_m2 import minimax_m2_229b_a10b_pretrain_config

    cfg = minimax_m2_229b_a10b_pretrain_config()
    assert cfg.dataset.blend is None


# ---------------------------------------------------------------------------
# __init__.py re-export
# ---------------------------------------------------------------------------


def test_minimax_m2_recipe_exported_from_package():
    """minimax_m2_229b_a10b_pretrain_config must be importable from the package."""
    from megatron.bridge.recipes.minimax_m2 import minimax_m2_229b_a10b_pretrain_config  # noqa: F401
