import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "compare_dsv4", Path(__file__).resolve().parents[3] / "scripts/debug/compare_dsv4_numerics.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(directory, *, changed=False, complete=True):
    directory.mkdir()
    records = [
        {
            "kind": "boundary",
            "schema": 1,
            "iteration": 1,
            "execution": "eager",
            "phase": "grad_pre_reduce",
            "name": "parameter",
            "occurrence": 0,
            "shape": [1],
            "dtype": "torch.bfloat16",
            "numel": 1,
            "digest": ["different" if changed else "same"],
        }
    ]
    records += [{"kind": "metrics", "iteration": i, "loss": {}, "grad_norm": 1.0} for i in range(1, 6)]
    if complete:
        records.append({"kind": "diagnostic_complete"})
    (directory / "rank00000.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")


@pytest.mark.unit
def test_detects_known_difference(tmp_path):
    _write(tmp_path / "a")
    _write(tmp_path / "b", changed=True)
    result = MODULE.compare(tmp_path / "a", tmp_path / "b", expected_ranks=1)
    assert result["ranks"][0]["mismatch_count"] == 1


@pytest.mark.unit
def test_equal_screens_not_reported_as_cause(tmp_path):
    _write(tmp_path / "a")
    _write(tmp_path / "b")
    result = MODULE.compare(tmp_path / "a", tmp_path / "b", expected_ranks=1)
    assert result["ranks"][0]["first_observed_boundary_difference"] is None


@pytest.mark.unit
def test_incomplete_trace_fails_closed(tmp_path):
    _write(tmp_path / "a")
    _write(tmp_path / "b", complete=False)
    with pytest.raises(ValueError, match="completed"):
        MODULE.compare(tmp_path / "a", tmp_path / "b", expected_ranks=1)
