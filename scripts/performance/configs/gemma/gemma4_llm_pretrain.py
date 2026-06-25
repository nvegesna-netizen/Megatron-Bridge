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

"""Performance-optimised pretrain config builders for Gemma 4.

Each public function in this module is named following the convention used by
``utils.utils.get_perf_optimized_recipe``::

    {model_recipe_name}_{train_task}_config_{gpu}

so that the perf-benchmark driver can locate builders by string lookup.

NOTE: The Gemma 4 E4B recipe uses transformer_impl="local" (non-TE).
TE-specific fusions (``masked_softmax_fusion``, ``gradient_accumulation_fusion``,
``apply_rope_fusion``) are disabled in the recipe and must not be re-enabled here.

NOTE: Gemma 4 uses SWA hybrid attention (window_attn_skip_freq=6).  Context
parallelism with SWA has not been verified; CP=1 is enforced in all configs.
"""

import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.gemma.gemma4 import gemma4_e4b_pretrain_config as pretrain_config
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)

_MODEL_FAMILY_NAME = "gemma"


def set_gemma4_common_configs(cfg: ConfigContainer) -> None:
    """Apply common performance overrides for all Gemma 4 configs.

    Adjustments made here override the library recipe defaults where the
    performance-benchmark environment requires different behaviour.
    """
    # Sequence length: use 8192 to exercise the full SWA sliding window range.
    # The recipe default is 4096; override here for perf benchmarking.
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    # Cross-entropy fusion: the E4B recipe sets cross_entropy_loss_fusion=True
    # with cross_entropy_fusion_impl="native".  Keep native impl since the local
    # (non-TE) spec does not support the TE cross-entropy kernel.
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Gradient reduce precision: match recipe default.
    cfg.ddp.grad_reduce_in_fp32 = True


def get_gemma4_workload_config(
    model_recipe_name: str,
    gpu: str,
    precision: str = "bf16",
    mock: bool = True,
    config_variant: str = "v1",
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Gemma 4 on a given GPU type.

    Args:
        model_recipe_name: Recipe name used to look up the workload base config,
            e.g. ``"gemma4_e4b"``.
        gpu: Target GPU identifier, e.g. ``"h100"``, ``"gb200"``, ``"gb300"``,
            ``"b200"``, or ``"b300"``.
        precision: Compute dtype string, e.g. ``"bf16"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.
    """
    base_cfg = get_workload_base_config(
        model_family_name=_MODEL_FAMILY_NAME,
        model_recipe_name=model_recipe_name,
        gpu=gpu,
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size

    set_gemma4_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def gemma4_e4b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Gemma 4 E4B on H100.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.

    Note:
        Parallelism settings are HYPOTHESES derived from model size and analogy
        to similar dense ~4B models on H100.  They have not been verified on
        H100 hardware.
    """
    return get_gemma4_workload_config("gemma4_e4b", "h100", precision, mock, config_variant)


def gemma4_e4b_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Gemma 4 E4B on GB200.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.
    """
    return get_gemma4_workload_config("gemma4_e4b", "gb200", precision, mock, config_variant)


def gemma4_e4b_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Gemma 4 E4B on GB300.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.

    Note:
        Parallelism settings are HYPOTHESES.  GB300 assumed similar to GB200
        until hardware profiling is completed.
    """
    return get_gemma4_workload_config("gemma4_e4b", "gb300", precision, mock, config_variant)


def gemma4_e4b_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Gemma 4 E4B on B200.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.

    Note:
        Parallelism settings are HYPOTHESES derived from H100 topology.
        They have not been verified on B200 hardware.
    """
    return get_gemma4_workload_config("gemma4_e4b", "b200", precision, mock, config_variant)


def gemma4_e4b_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """Return a perf-optimised pretrain config for Gemma 4 E4B on B300.

    Args:
        precision: Compute dtype string, e.g. ``"bf16"``.
        mock: Unused — present for API compatibility with the benchmark driver.
        config_variant: Workload base config variant label, e.g. ``"v1"``.

    Returns:
        A fully configured :class:`ConfigContainer` ready for the benchmark
        driver to consume.

    Note:
        Parallelism settings are HYPOTHESES.  B300 assumed similar to GB200
        until hardware profiling is completed.
    """
    return get_gemma4_workload_config("gemma4_e4b", "b300", precision, mock, config_variant)
