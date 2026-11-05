import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE = _load("upstream_probe", ROOT / "src/megatron/bridge/training/dsv4_upstream.py")
COMPARE = _load("upstream_compare", ROOT / "scripts/debug/compare_dsv4_upstream.py")


@pytest.mark.unit
def test_sha_detects_duplicate_cancellation_and_permutation():
    a = torch.tensor([1, 1, 2, 2], dtype=torch.int32)
    b = torch.tensor([3, 3, 4, 4], dtype=torch.int32)
    c = a.flip(0)
    sig = PROBE.tensor_signature
    assert sig(a)["sha256"] != sig(b)["sha256"]
    assert sig(a)["sha256"] != sig(c)["sha256"]
    matrix = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    assert sig(matrix.T)["sha256"] == sig(matrix.T.contiguous())["sha256"]
    assert sig(matrix)["sha256"] == sig(matrix.clone())["sha256"]


@pytest.mark.unit
def test_trace_is_bounded_and_does_not_change_outputs_or_gradients(tmp_path):
    torch.manual_seed(5)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh())
    x = torch.randn(2, 8, requires_grad=True)
    original = model(x)
    original.sum().backward()
    grad = x.grad.clone()
    x.grad = None
    attention = ModuleType("fixture_attention")
    attention.__file__ = __file__
    sys.modules[attention.__name__] = attention
    trace = PROBE.UpstreamTrace(tmp_path, 2, [model], attention)
    for i in range(3):
        trace.projection_input(i, x, model[0].weight)
        result = model(x)
        trace.projection_output(i, result)
    trace.finish_iteration()
    assert torch.equal(result, original)
    result.sum().backward()
    assert torch.equal(x.grad, grad)
    rows = [json.loads(line) for line in (tmp_path / "upstream_rank00002.jsonl").read_text().splitlines()]
    assert rows[-1]["kind"] == "upstream_complete"
    assert trace.hooks == [] and not trace.active
    assert any(r.get("phase") == "module_parameters" for r in rows)


@pytest.mark.unit
def test_opaque_input_does_not_vacuously_match():
    assert PROBE.signature(object())["complete"] is False


@pytest.mark.unit
def test_inactive_upstream_does_not_compute_extra_canonical_reshape():
    boundary = _load("boundary_probe", ROOT / "src/megatron/bridge/training/dsv4_numerics.py")
    trace = boundary._Trace.__new__(boundary._Trace)
    trace.rank, trace.step, trace.projection_calls = 2, 2, 0
    trace.record = lambda *args: None

    def unexpected(*args):
        raise AssertionError("Inactive upstream callback must not run")

    trace.upstream = SimpleNamespace(active=False, projection_input=unexpected, projection_output=unexpected)
    x = torch.randn(2, 1, 4)
    weight = torch.randn(4, 4)
    expected = x + 1
    assert trace._projection(lambda: expected, x, weight) is expected


@pytest.mark.unit
def test_incomplete_window_fails_closed(tmp_path):
    for arm in ("a", "b"):
        folder = tmp_path / arm
        folder.mkdir()
        (folder / "upstream_rank00000.jsonl").write_text("")
    with pytest.raises(ValueError, match="completed"):
        COMPARE.compare(tmp_path / "a", tmp_path / "b", ranks=1)


@pytest.mark.unit
def test_duplicate_identity_fails_closed(tmp_path):
    row = {
        "schema": "dsv4-upstream-sha256-v1",
        "rank": 0,
        "kind": "upstream_boundary",
        "iteration": 1,
        "execution": "eager",
        "phase": "module_input",
        "name": "fixture",
        "occurrence": 0,
        "seq": 0,
        "value": {"sha256": "x", "complete": True},
    }
    complete = {
        "schema": row["schema"],
        "rank": 0,
        "kind": "upstream_complete",
        "projections": [0, 1, 2],
        "boundary_records": 2,
    }
    for arm in ("a", "b"):
        folder = tmp_path / arm
        folder.mkdir()
        (folder / "upstream_rank00000.jsonl").write_text(
            "\n".join(map(json.dumps, [row, {**row, "seq": 1}, complete]))
        )
    with pytest.raises(ValueError, match="Duplicate"):
        COMPARE.compare(tmp_path / "a", tmp_path / "b", ranks=1)


@pytest.mark.unit
def test_comparator_detects_value_difference_not_tensor_type(tmp_path):
    row = {
        "schema": "dsv4-upstream-sha256-v1",
        "rank": 0,
        "kind": "upstream_boundary",
        "iteration": 1,
        "execution": "eager",
        "phase": "projection_input",
        "name": "projection0002",
        "occurrence": 0,
        "seq": 0,
        "value": {"shape": [1], "sha256": "same", "complete": True, "tensor_type": "A"},
    }
    complete = {
        "schema": row["schema"],
        "rank": 0,
        "kind": "upstream_complete",
        "projections": [0, 1, 2],
        "boundary_records": 1,
    }
    for arm in ("a", "b"):
        folder = tmp_path / arm
        folder.mkdir()
        other = {**row, "value": {**row["value"], "tensor_type": arm}}
        (folder / "upstream_rank00000.jsonl").write_text("\n".join(map(json.dumps, [other, complete])))
    assert COMPARE.compare(tmp_path / "a", tmp_path / "b", ranks=1)["ranks"][0]["first_observed_difference"] is None
    changed = {**row, "value": {**row["value"], "sha256": "different"}}
    (tmp_path / "b/upstream_rank00000.jsonl").write_text("\n".join(map(json.dumps, [changed, complete])))
    assert (
        COMPARE.compare(tmp_path / "a", tmp_path / "b", ranks=1)["ranks"][0]["first_observed_difference"] is not None
    )
