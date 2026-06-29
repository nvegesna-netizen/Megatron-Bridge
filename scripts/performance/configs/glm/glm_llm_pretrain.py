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

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.glm.glm45 import glm45_355b_pretrain_config, glm45_air_106b_pretrain_config
from megatron.bridge.recipes.glm.glm5 import glm52_753b_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GLM-5.2 753B (MoE + MLA + DSA)
# ─────────────────────────────────────────────────────────────────────────────


def set_glm52_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all GLM-5.2 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    if hasattr(cfg.dataset, "sequence_length"):
        cfg.dataset.sequence_length = 4096

    cfg.model.context_parallel_size = 1
    cfg.model.apply_rope_fusion = False
    cfg.model.mtp_num_layers = None
    cfg.model.mtp_loss_scaling_factor = None
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.recompute_granularity = "selective"
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    if (
        cfg.model.moe_flex_dispatcher_backend == "hybridep"
        and (cfg.mixed_precision.fp8 is not None or cfg.mixed_precision.fp4 is not None)
    ):
        cfg.model.moe_router_padding_for_quantization = True


def _glm52_pretrain_config_for_gpu(
    gpu: str,
    precision: str,
    config_variant: str,
) -> ConfigContainer:
    base_cfg = get_workload_base_config(
        model_family_name="glm",
        model_recipe_name="glm52_753b",
        gpu=gpu,
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    cfg = glm52_753b_pretrain_config()
    cfg.mixed_precision = get_precision_config(precision)

    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    # Call common configs twice: once before set_workload_base_configs so recipe-
    # specific fields are initialised, and once after to re-apply overrides that
    # _set_common_perf_overrides (inside set_workload_base_configs) would clobber
    # (e.g. cross_entropy_fusion_impl="native", apply_rope_fusion=False).
    set_glm52_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    set_glm52_common_configs(cfg)

    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout

    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def glm52_753b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300 baseline config for GLM-5.2 753B."""
    return _glm52_pretrain_config_for_gpu("gb300", precision, config_variant)


def glm52_753b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200 baseline config for GLM-5.2 753B."""
    return _glm52_pretrain_config_for_gpu("gb200", precision, config_variant)


def glm52_753b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300 baseline config for GLM-5.2 753B."""
    return _glm52_pretrain_config_for_gpu("b300", precision, config_variant)


def glm52_753b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200 baseline config for GLM-5.2 753B."""
    return _glm52_pretrain_config_for_gpu("b200", precision, config_variant)


def glm52_753b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100 baseline config for GLM-5.2 753B."""
    return _glm52_pretrain_config_for_gpu("h100", precision, config_variant)


# ─────────────────────────────────────────────────────────────────────────────
# GLM-4.5 355B-A32B (MoE + GQA + QK-norm, deepep dispatcher)
# ─────────────────────────────────────────────────────────────────────────────


def set_glm45_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance overrides for all GLM-4.5 configs.

    Must be called after set_workload_base_configs() to re-apply values that
    _set_common_perf_overrides() (called inside set_workload_base_configs) would
    otherwise clobber — specifically cross_entropy_fusion_impl.
    """
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    if hasattr(cfg.dataset, "sequence_length"):
        cfg.dataset.sequence_length = 4096

    # GLM-4.5 cross-entropy kernel: native outperforms the TE implementation.
    # _set_common_perf_overrides sets fusion_impl="te"; re-apply "native" after.
    cfg.model.cross_entropy_fusion_impl = "native"

    cfg.model.moe_router_fusion = False
    cfg.model.recompute_granularity = "selective"

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


def _glm45_pretrain_config_for_gpu(
    recipe_name: str,
    recipe_fn,
    gpu: str,
    precision: str,
    config_variant: str,
) -> ConfigContainer:
    base_cfg = get_workload_base_config(
        model_family_name="glm",
        model_recipe_name=recipe_name,
        gpu=gpu,
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    cfg = recipe_fn()
    cfg.mixed_precision = get_precision_config(precision)

    if base_cfg.moe_flex_dispatcher_backend is not None:
        cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    set_workload_base_configs(cfg, base_cfg)
    # Re-apply after set_workload_base_configs to override _set_common_perf_overrides.
    set_glm45_common_configs(cfg)

    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def glm45_355b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300 baseline config for GLM-4.5 355B-A32B."""
    return _glm45_pretrain_config_for_gpu("glm45_355b", glm45_355b_pretrain_config, "gb300", precision, config_variant)


def glm45_355b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200 baseline config for GLM-4.5 355B-A32B."""
    return _glm45_pretrain_config_for_gpu("glm45_355b", glm45_355b_pretrain_config, "gb200", precision, config_variant)


def glm45_355b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300 baseline config for GLM-4.5 355B-A32B."""
    return _glm45_pretrain_config_for_gpu("glm45_355b", glm45_355b_pretrain_config, "b300", precision, config_variant)


def glm45_355b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200 baseline config for GLM-4.5 355B-A32B."""
    return _glm45_pretrain_config_for_gpu("glm45_355b", glm45_355b_pretrain_config, "b200", precision, config_variant)


def glm45_355b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100 baseline config for GLM-4.5 355B-A32B."""
    return _glm45_pretrain_config_for_gpu("glm45_355b", glm45_355b_pretrain_config, "h100", precision, config_variant)


# ─────────────────────────────────────────────────────────────────────────────
# GLM-4.5-Air 106B-A12B
# ─────────────────────────────────────────────────────────────────────────────


def glm45_air_106b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300 baseline config for GLM-4.5-Air 106B-A12B."""
    return _glm45_pretrain_config_for_gpu(
        "glm45_air_106b", glm45_air_106b_pretrain_config, "gb300", precision, config_variant
    )


def glm45_air_106b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200 baseline config for GLM-4.5-Air 106B-A12B."""
    return _glm45_pretrain_config_for_gpu(
        "glm45_air_106b", glm45_air_106b_pretrain_config, "gb200", precision, config_variant
    )


def glm45_air_106b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300 baseline config for GLM-4.5-Air 106B-A12B."""
    return _glm45_pretrain_config_for_gpu(
        "glm45_air_106b", glm45_air_106b_pretrain_config, "b300", precision, config_variant
    )


def glm45_air_106b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200 baseline config for GLM-4.5-Air 106B-A12B."""
    return _glm45_pretrain_config_for_gpu(
        "glm45_air_106b", glm45_air_106b_pretrain_config, "b200", precision, config_variant
    )


def glm45_air_106b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100 baseline config for GLM-4.5-Air 106B-A12B."""
    return _glm45_pretrain_config_for_gpu(
        "glm45_air_106b", glm45_air_106b_pretrain_config, "h100", precision, config_variant
    )
