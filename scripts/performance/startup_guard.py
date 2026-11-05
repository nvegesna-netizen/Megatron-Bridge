# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in CPU-only, durable Slurm startup checks before importing training code.

This checks launch readiness, not GPU, network, or training correctness. It adds
no CUDA/distributed imports or collectives and is inactive without a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
MANIFEST_ENV = "MB_STARTUP_MANIFEST"
SOURCE_FILES = (
    "scripts/performance/bootstrap.py",
    "scripts/performance/startup_guard.py",
    "scripts/performance/run_script.py",
    "scripts/performance/argument_parser.py",
    "src/megatron/bridge/training/dsv4_numerics.py",
    "src/megatron/bridge/training/dsv4_upstream.py",
    "3rdparty/Megatron-LM/megatron/core/transformer/attention.py",
    "3rdparty/Megatron-LM/megatron/core/transformer/moe/experts.py",
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_manifest(
    repo_root: Path,
    directory: Path,
    *,
    world_size: int,
    tasks_per_node: int,
    excluded_hosts: list[str],
    timeout_s: float = 180,
) -> Path:
    """Create a fresh launch contract; never reuse another attempt's receipts."""
    if world_size <= 0 or tasks_per_node <= 0 or world_size % tasks_per_node or timeout_s <= 0:
        raise ValueError("Invalid startup topology or timeout")
    repo_root = repo_root.resolve(strict=True)
    files = {str(repo_root / name): _digest(repo_root / name) for name in SOURCE_FILES}
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory / "manifest.json"
    _write(
        manifest,
        {
            "schema": 1,
            "run_id": uuid.uuid4().hex,
            "world_size": world_size,
            "tasks_per_node": tasks_per_node,
            "timeout_s": timeout_s,
            "excluded_hosts": excluded_hosts,
            "files": files,
        },
    )
    return manifest


def record_stage(stage: str, error: str | None = None) -> None:
    """Persist an allowlisted rank-local bootstrap/exec stage when enabled."""
    if not (manifest_name := os.environ.get(MANIFEST_ENV)):
        return
    directory = Path(manifest_name).parent
    rank = int(os.environ["SLURM_PROCID"])
    _write(
        directory / f"stage_rank{rank:05d}.json",
        {
            "rank": rank,
            "host": socket.gethostname(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "stage": stage,
            "error": error,
            "time_ns": time.time_ns(),
        },
    )


def _validate_receipts(receipts: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    world = manifest["world_size"]
    tasks = manifest["tasks_per_node"]
    if sorted(item["rank"] for item in receipts) != list(range(world)):
        raise ValueError("Incomplete or duplicate startup ranks")
    hosts: set[str] = set()
    for node in range(world // tasks):
        group = [item for item in receipts if item["node_id"] == node]
        if len(group) != tasks or {item["local_rank"] for item in group} != set(range(tasks)):
            raise ValueError(f"Invalid startup task coverage for node {node}")
        names = {item["host"] for item in group}
        if len(names) != 1 or hosts.intersection(names):
            raise ValueError("Startup host/node mapping is not one-to-one")
        hosts.update(names)
    for item in receipts:
        if (
            item["run_id"] != manifest["run_id"]
            or item["world_size"] != world
            or item["files"] != manifest["files"]
            or item["status"] != "checked"
            or item["host"].split(".")[0] in manifest["excluded_hosts"]
        ):
            raise ValueError(f"Invalid startup receipt rank {item['rank']}")
    if len({item["job_id"] for item in receipts}) != 1 or len({item["step_id"] for item in receipts}) != 1:
        raise ValueError("Startup receipts span different Slurm jobs or steps")


def check_startup() -> None:
    """Check each task's sources, then require all tasks before training imports."""
    if not (manifest_name := os.environ.get(MANIFEST_ENV)):
        return
    path = Path(manifest_name)
    rank = int(os.environ["SLURM_PROCID"])
    record_stage("checking_sources")
    try:
        manifest = json.loads(path.read_text())
        world = int(os.environ["SLURM_NTASKS"])
        if world != manifest["world_size"] or not 0 <= rank < world:
            raise ValueError("Slurm task count/rank disagrees with startup contract")
        host = socket.gethostname()
        if host.split(".")[0] in manifest["excluded_hosts"]:
            raise ValueError(f"Excluded host allocated: {host}")
        files = {name: _digest(Path(name)) for name in manifest["files"]}
        if files != manifest["files"]:
            raise ValueError("Startup source SHA256 differs from launcher contract")
        receipt = {
            "run_id": manifest["run_id"],
            "rank": rank,
            "local_rank": int(os.environ["SLURM_LOCALID"]),
            "node_id": int(os.environ["SLURM_NODEID"]),
            "job_id": os.environ["SLURM_JOB_ID"],
            "step_id": os.environ["SLURM_STEP_ID"],
            "host": host,
            "world_size": world,
            "status": "checked",
            "files": files,
        }
        _write(path.parent / f"checked_rank{rank:05d}.json", receipt)
        deadline = time.monotonic() + manifest["timeout_s"]
        ready_path = path.parent / "ready.json"
        while time.monotonic() < deadline:
            failures = sorted(path.parent.glob("failure_rank*.json"))
            if failures:
                raise RuntimeError(f"Startup task failed; see {failures[0].name}")
            if rank == 0:
                receipts = [json.loads(item.read_text()) for item in sorted(path.parent.glob("checked_rank*.json"))]
                if len(receipts) >= world:
                    _validate_receipts(receipts, manifest)
                    _write(
                        ready_path, {"run_id": manifest["run_id"], "world_size": world, "job_id": receipt["job_id"]}
                    )
            if ready_path.is_file():
                ready = json.loads(ready_path.read_text())
                if ready != {"run_id": manifest["run_id"], "world_size": world, "job_id": receipt["job_id"]}:
                    raise ValueError("Stale or inconsistent startup ready marker")
                record_stage("all_tasks_checked")
                return
            time.sleep(0.2)
        raise TimeoutError(f"Startup readiness timed out before all {world} tasks checked in")
    except Exception as error:
        _write(
            path.parent / f"failure_rank{rank:05d}.json",
            {"rank": rank, "host": socket.gethostname(), "error": f"{type(error).__name__}: {error}"},
        )
        record_stage("startup_failed", f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--tasks-per-node", type=int, required=True)
    parser.add_argument("--exclude-hosts", default="")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Prepared startup manifest: %s",
        prepare_manifest(
            arguments.repo_root,
            arguments.directory,
            world_size=arguments.world_size,
            tasks_per_node=arguments.tasks_per_node,
            excluded_hosts=[host for host in arguments.exclude_hosts.split(",") if host],
        ),
    )
