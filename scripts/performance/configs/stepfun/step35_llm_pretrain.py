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

"""Performance-optimised pretrain config builders for Step-3.5-Flash.

Each public function in this module is named following the convention used by
``utils.utils.get_perf_optimized_recipe``::

    {model_recipe_name}_{train_task}_config_{gpu}

so that the perf-benchmark driver can locate builders by string lookup.
"""

import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.stepfun.step35 import (
    step35_196b_a11b_pretrain_config as pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def set_step35_common_configs(cfg: ConfigContainer) -> None:
    """Apply common performance overrides for all Step-3.5-Flash configs.

    Adjustments made here override the library recipe defaults where the
    performance-benchmark environment requires different behaviour.
    """
    # Sequence length matches the recipe default; set explicitly so the
    # dataset config is kept consistent if the recipe value ever changes.
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    # MoE kernel selections — keep in sync with recipe defaults.
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # Gradient reduce precision: default float32 is already set in the recipe;
    # leave as-is for correctness — override here only if profiling shows a gain.
    cfg.mixed_precision.grad_reduce_in_fp32 = True


def step35_196b_a11b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Step-3.5-Flash on GB300.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"`` or ``"fp8"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.
    """
    base_cfg = get_workload_base_config(
        model_family_name="stepfun",
        model_recipe_name="step35_196b_a11b",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    set_step35_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def step35_196b_a11b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Step-3.5-Flash on GB200.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"`` or ``"fp8"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.
    """
    base_cfg = get_workload_base_config(
        model_family_name="stepfun",
        model_recipe_name="step35_196b_a11b",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    set_step35_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def step35_196b_a11b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Step-3.5-Flash on B200.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.

    Note:
        Parallelism settings are HYPOTHESES derived from GB200 topology.  They
        have not been verified on B200 hardware.
    """
    base_cfg = get_workload_base_config(
        model_family_name="stepfun",
        model_recipe_name="step35_196b_a11b",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    set_step35_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def step35_196b_a11b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Step-3.5-Flash on B300.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.

    Note:
        Parallelism settings are HYPOTHESES.  B300 assumed similar to B200
        until hardware profiling is completed.
    """
    base_cfg = get_workload_base_config(
        model_family_name="stepfun",
        model_recipe_name="step35_196b_a11b",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    set_step35_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def step35_196b_a11b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Step-3.5-Flash on H100.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.

    Note:
        Parallelism settings are HYPOTHESES.  H100 numbers are rough estimates
        pending actual profiling at scale.
    """
    base_cfg = get_workload_base_config(
        model_family_name="stepfun",
        model_recipe_name="step35_196b_a11b",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    set_step35_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    # Disabling to avoid functional errors on H100.
    # TODO: Test with it enabled and keep it enabled if it works.
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg
