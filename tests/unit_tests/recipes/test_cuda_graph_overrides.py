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

"""TE RNG tracker handling in _set_cuda_graph_overrides.

mcore hard-requires the TE RNG tracker only for CUDA graphs. In the
``cuda_graph_impl == "none"`` branch the tracker is therefore a fidelity choice, and it
is disabled only when the recipe wants neither an interleaved pipeline (VPP>1) nor an MTP
head (mtp_num_layers>0). Keeping it on for those cases preserves the tuned recipe's
explicit ``use_te_rng_tracker=True`` (e.g. deepseek_v4 / glm45, which run VPP=1 + MTP with
CUDA graphs off).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS_PERF_PATH = Path(__file__).parents[3] / "scripts" / "performance"
sys.path.insert(0, str(SCRIPTS_PERF_PATH))


pytestmark = pytest.mark.unit


def _recipe(*, cuda_graph_impl="none", vpp=None, mtp=None, te_rng=True):
    """Minimal recipe stub exposing only the fields _set_cuda_graph_overrides reads."""
    return SimpleNamespace(
        model=SimpleNamespace(
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_scope=None,
            virtual_pipeline_model_parallel_size=vpp,
            mtp_num_layers=mtp,
            use_te_rng_tracker=te_rng,
        ),
        rng=SimpleNamespace(te_rng_tracker=te_rng),
        rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=True),
    )


def _apply(recipe, **kwargs):
    from utils.overrides import _set_cuda_graph_overrides

    return _set_cuda_graph_overrides(recipe, **kwargs)


def test_none_plain_ddp_no_mtp_disables_tracker():
    """Plain DDP, no MTP, no CUDA graphs -> tracker is pure overhead -> disabled."""
    r = _apply(_recipe(cuda_graph_impl="none", vpp=None, mtp=None))
    assert r.rng.te_rng_tracker is False
    assert r.model.use_te_rng_tracker is False


def test_none_vpp_gt1_keeps_tracker():
    """Interleaved pipeline (VPP>1) keeps the tracker for per-microbatch RNG coherence."""
    r = _apply(_recipe(cuda_graph_impl="none", vpp=2, mtp=0))
    assert r.rng.te_rng_tracker is True
    assert r.model.use_te_rng_tracker is True


def test_none_mtp_at_vpp1_keeps_tracker():
    """Regression: deepseek_v4 / glm45 run VPP=1 + MTP with CUDA graphs off and explicitly
    enable the tracker; the "none" branch must not silently revert it to False."""
    r = _apply(_recipe(cuda_graph_impl="none", vpp=1, mtp=1))
    assert r.rng.te_rng_tracker is True
    assert r.model.use_te_rng_tracker is True


def test_transformer_engine_graphs_enable_tracker():
    """CUDA graphs branch enables the tracker (mcore hard requirement)."""
    r = _apply(
        _recipe(cuda_graph_impl="none", vpp=1, mtp=0, te_rng=False),
        cuda_graph_impl="transformer_engine",
        cuda_graph_scope="attn",
    )
    assert r.rng.te_rng_tracker is True
    assert r.model.use_te_rng_tracker is True
