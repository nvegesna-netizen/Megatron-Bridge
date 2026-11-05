# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compare order-sensitive upstream boundaries without asserting operator causes."""

import argparse
import gzip
import json
from pathlib import Path


def _logical(value: object) -> object:
    if isinstance(value, dict):
        # Tensor layout/type is diagnostic metadata, not a logical value mismatch.
        return {key: _logical(item) for key, item in value.items() if key != "tensor_type"}
    if isinstance(value, list):
        return [_logical(item) for item in value]
    return value


def _incomplete(value: object) -> bool:
    if isinstance(value, dict):
        return value.get("complete") is False or any(_incomplete(item) for item in value.values())
    return isinstance(value, list) and any(_incomplete(item) for item in value)


def compare(left: Path, right: Path, *, ranks: int) -> dict[str, object]:
    """Align same-rank identities, require completed windows, retain coverage gaps."""
    reports = []
    for root in (left, right):
        found = {p.name.removesuffix(".gz") for p in root.glob("upstream_rank*.jsonl*")}
        expected = {f"upstream_rank{rank:05d}.jsonl" for rank in range(ranks)}
        if found != expected:
            raise ValueError(f"Incomplete upstream rank coverage: {root}")
    for rank in range(ranks):
        streams = []
        for root in (left, right):
            path = root / f"upstream_rank{rank:05d}.jsonl"
            path = path if path.exists() else path.with_suffix(".jsonl.gz")
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt") as file:
                rows = [json.loads(line) for line in file]
            if any(r["schema"] != "dsv4-upstream-sha256-v1" or r["rank"] != rank for r in rows):
                raise ValueError("Schema or rank mismatch")
            completed = [r for r in rows if r["kind"] == "upstream_complete"]
            if len(completed) != 1 or completed[0]["projections"] != [0, 1, 2]:
                raise ValueError("Missing completed first-three-projection window")
            events = [r for r in rows if r["kind"] == "upstream_boundary"]
            if not events or len(events) != completed[0]["boundary_records"]:
                raise ValueError("Missing boundaries")
            if [r["seq"] for r in events] != list(range(len(events))):
                raise ValueError("Broken rank-local sequence")
            streams.append(events)

        def identity(row: dict[str, object]) -> tuple[object, ...]:
            return row["iteration"], row["execution"], row["phase"], row["name"], row["occurrence"]

        a, b = streams
        ai, bi = {identity(row): row for row in a}, {identity(row): row for row in b}
        if len(ai) != len(a) or len(bi) != len(b):
            raise ValueError("Duplicate boundary identity")
        differences = []
        for row in a:
            other = bi.pop(identity(row), None)
            if other is None or _logical(row["value"]) != _logical(other["value"]):
                differences.append({"identity": identity(row), "baseline": row, "patched": other})
        differences.extend({"identity": key, "baseline": None, "patched": row} for key, row in bi.items())
        gaps = {
            "baseline": [identity(r) for r in a if _incomplete(r["value"])],
            "patched": [identity(r) for r in b if _incomplete(r["value"])],
        }
        reports.append(
            {
                "rank": rank,
                "first_observed_difference": differences[0] if differences else None,
                "differences": differences,
                "incomplete_signatures": gaps,
                "boundary_counts": [len(a), len(b)],
            }
        )
    return {
        "scope": "First observed same-rank SHA256 boundaries; no exact causal-op or cross-rank ordering claim.",
        "ranks": reports,
    }


def main() -> None:
    """Compare complete diagnostic trees without importing training packages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.left, args.right, ranks=args.ranks), indent=2) + "\n")


if __name__ == "__main__":
    main()
