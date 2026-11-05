"""CPU subprocess regression checks for opt-in Slurm startup readiness."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "scripts" / "performance" / "startup_guard.py"
SPEC = importlib.util.spec_from_file_location("startup_guard_test", SOURCE)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


@pytest.fixture
def contract(tmp_path):
    repo = tmp_path / "repo"
    for name in guard.SOURCE_FILES:
        source = repo / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {name}\n")
    manifest = guard.prepare_manifest(
        repo, tmp_path / "startup", world_size=4, tasks_per_node=2, excluded_hosts=["excluded"], timeout_s=3
    )
    return repo, manifest


def _launch(manifest, rank, *, host=None, world=4):
    code = (
        "import importlib.util,os,socket; "
        f"s=importlib.util.spec_from_file_location('guard',{str(SOURCE)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "socket.gethostname=lambda:os.environ['TEST_HOST']; "
        "m.check_startup(); m.record_stage('exec_training')"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "MB_STARTUP_MANIFEST": str(manifest),
            "SLURM_PROCID": str(rank),
            "SLURM_LOCALID": str(rank % 2),
            "SLURM_NODEID": str(rank // 2),
            "SLURM_JOB_ID": "123",
            "SLURM_STEP_ID": "1",
            "SLURM_NTASKS": str(world),
            "TEST_HOST": host or f"node{rank // 2}",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(processes):
    results = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            results.append((process.returncode, stdout, stderr))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    return results


def test_guard_is_inactive_without_opt_in(monkeypatch):
    monkeypatch.delenv(guard.MANIFEST_ENV, raising=False)
    guard.check_startup()
    guard.record_stage("ignored")


def test_all_cpu_tasks_agree_before_training(contract):
    _, manifest = contract
    results = _finish([_launch(manifest, rank) for rank in range(4)])
    assert all(code == 0 for code, _, _ in results), results
    ready = json.loads((manifest.parent / "ready.json").read_text())
    assert ready["world_size"] == 4
    receipts = [json.loads(path.read_text()) for path in manifest.parent.glob("checked_rank*.json")]
    assert len(receipts) == 4
    assert {item["host"] for item in receipts} == {"node0", "node1"}
    assert all(
        json.loads(path.read_text())["stage"] == "exec_training" for path in manifest.parent.glob("stage_*.json")
    )


@pytest.mark.parametrize("fault", ["missing", "changed", "excluded", "wrong_world", "missing_task", "duplicate_host"])
def test_failures_prevent_readiness_and_preserve_evidence(contract, fault):
    repo, manifest = contract
    if fault == "missing":
        (repo / guard.SOURCE_FILES[2]).unlink()
    elif fault == "changed":
        (repo / guard.SOURCE_FILES[2]).write_text("changed\n")
    ranks = range(3) if fault == "missing_task" else range(4)
    results = _finish(
        [
            _launch(
                manifest,
                rank,
                host="excluded"
                if fault == "excluded" and rank == 3
                else "same"
                if fault == "duplicate_host"
                else None,
                world=8 if fault == "wrong_world" else 4,
            )
            for rank in ranks
        ]
    )
    assert all(code != 0 for code, _, _ in results), results
    assert not (manifest.parent / "ready.json").exists()
    failures = [json.loads(path.read_text()) for path in manifest.parent.glob("failure_rank*.json")]
    assert failures
    assert all(item["host"] and item["error"] for item in failures)


def test_manifest_cannot_reuse_previous_attempt(contract):
    repo, manifest = contract
    with pytest.raises(FileExistsError):
        guard.prepare_manifest(repo, manifest.parent, world_size=4, tasks_per_node=2, excluded_hosts=[])


def test_mixed_job_receipts_are_rejected(contract):
    _, manifest = contract
    results = _finish([_launch(manifest, rank) for rank in range(4)])
    assert all(code == 0 for code, _, _ in results)
    receipts = [json.loads(path.read_text()) for path in manifest.parent.glob("checked_rank*.json")]
    receipts[0]["job_id"] = "other"
    with pytest.raises(ValueError, match="different Slurm jobs"):
        guard._validate_receipts(receipts, json.loads(manifest.read_text()))


def test_no_gpu_or_framework_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import runpy,sys; runpy.run_path({str(SOURCE)!r}); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_full_128_rank_topology_rejects_the_observed_missing_node(contract):
    _, path = contract
    manifest = json.loads(path.read_text())
    manifest.update(world_size=128, tasks_per_node=4)
    receipts = [
        {
            "rank": rank,
            "node_id": rank // 4,
            "local_rank": rank % 4,
            "host": f"node{rank // 4}",
            "world_size": 128,
            "run_id": manifest["run_id"],
            "files": manifest["files"],
            "status": "checked",
            "job_id": "123",
            "step_id": "1",
        }
        for rank in range(128)
    ]
    guard._validate_receipts(receipts, manifest)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        guard._validate_receipts([item for item in receipts if item["rank"] not in range(120, 124)], manifest)


@pytest.fixture
def bootstrap(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE.parent))
    parser = SimpleNamespace(parse_known_args=lambda: (SimpleNamespace(), ["++sentinel=1"]))
    monkeypatch.setitem(sys.modules, "argument_parser", SimpleNamespace(parse_cli_args=lambda: parser))
    specification = importlib.util.spec_from_file_location("bootstrap_startup_test", SOURCE.parent / "bootstrap.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_bootstrap_guard_precedes_recipe_import_and_exec(bootstrap, monkeypatch):
    events = []
    monkeypatch.setattr(bootstrap, "check_startup", lambda: events.append("guard"))

    def prepare(_args, _overrides):
        events.append("recipe")
        return SimpleNamespace(env_vars={}, model=None), "run_script.py"

    monkeypatch.setattr(bootstrap, "_prepare_recipe_and_target", prepare)
    monkeypatch.setattr(bootstrap, "record_stage", lambda stage: events.append(stage))
    monkeypatch.setattr(bootstrap.os, "execvpe", lambda *_: events.append("exec"))
    bootstrap.main()
    assert events == ["guard", "recipe", "exec_training", "exec"]


def test_failed_guard_never_prepares_recipe(bootstrap, monkeypatch):
    def failed():
        raise RuntimeError("missing task")

    monkeypatch.setattr(bootstrap, "check_startup", failed)
    monkeypatch.setattr(
        bootstrap, "_prepare_recipe_and_target", lambda *_: pytest.fail("training import after failure")
    )
    with pytest.raises(RuntimeError, match="missing task"):
        bootstrap.main()


def test_unconfigured_bootstrap_preserves_exec_and_recipe_environment(bootstrap, monkeypatch):
    monkeypatch.delenv("MB_STARTUP_MANIFEST", raising=False)
    monkeypatch.setattr(bootstrap.sys, "argv", ["bootstrap.py", "--model_recipe_name", "fixture"])
    recipe = SimpleNamespace(env_vars={"STARTUP_TEST_VALUE": "unchanged"}, model=None)
    monkeypatch.setattr(bootstrap, "_prepare_recipe_and_target", lambda *_: (recipe, "run_script.py"))
    executions = []
    monkeypatch.setattr(
        bootstrap.os, "execvpe", lambda executable, argv, env: executions.append((executable, argv, env))
    )
    bootstrap.main()
    assert len(executions) == 1
    executable, argv, environment = executions[0]
    assert executable == sys.executable
    assert argv == [sys.executable, str(SOURCE.parent / "run_script.py"), "--model_recipe_name", "fixture"]
    assert environment["STARTUP_TEST_VALUE"] == "unchanged"
    monkeypatch.delenv("STARTUP_TEST_VALUE")
