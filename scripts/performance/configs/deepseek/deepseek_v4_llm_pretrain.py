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

"""DeepSeek-V4-Flash per-GPU golden-config builders (JET-LLM perf patch).

Mirrors deepseek_llm_pretrain.py (V3): each builder loads its WorkloadBaseConfig,
constructs the base recipe, applies precision, PP/VPP/dispatcher from the WBC, the
V4 pipeline layout, common configs, then the workload-base overrides.

The recipe (deepseek_v4_flash_pretrain_config) and the pipeline-layout function are
imported from megatron.bridge.recipes.deepseek.deepseek_v4 — these are shipped by
this patch's src/ overlay (the stock container does not yet have DeepSeek V4), and
require a DSv4-capable megatron-core. This module therefore imports cleanly only on
a V4-capable container; configs/deepseek/__init__.py guards the import so V3 runs on
stock containers are unaffected.
"""

import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.deepseek.deepseek_v4 import (
    deepseek_v4_flash_pretrain_config as pretrain_config,
)
from megatron.bridge.recipes.deepseek.deepseek_v4 import (
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.training.config import ConfigContainer

logger = logging.getLogger(__name__)


def set_deepseek_v4_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all DeepSeek-V4 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # NOTE: unlike V3, V4 intentionally does NOT set moe_router_force_load_balancing.
    # The first `moe_n_hash_layers` MoE layers use deterministic hash routing
    # (a precomputed tid2eid map), not learned routing, so forced load balancing
    # does not apply to them.


def _deepseek_v4_pretrain_config(
    gpu: str, precision: str, config_variant: str
) -> ConfigContainer:
    """Shared builder body for all DeepSeek-V4 per-GPU configs (all Blackwell)."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v4",
        gpu=gpu,
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    # Apply model-specific settings derived from the workload base config.
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute the V4 layout for the (possibly updated) PP/VP sizes. This keeps
        # the hash-routed MoE layers co-located with the embedding on stage 0.
        set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v4_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v4_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    return _deepseek_v4_pretrain_config("gb300", precision, config_variant)


def deepseek_v4_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    return _deepseek_v4_pretrain_config("gb200", precision, config_variant)


def deepseek_v4_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    return _deepseek_v4_pretrain_config("b300", precision, config_variant)


def deepseek_v4_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    return _deepseek_v4_pretrain_config("b200", precision, config_variant)
