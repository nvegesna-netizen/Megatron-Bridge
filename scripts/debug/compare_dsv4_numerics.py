# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compare rank-local boundary screens, never assert causality from hashes alone."""

import argparse
import gzip
import json
from pathlib import Path


def _load(path):
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    records = [json.loads(line) for line in text.splitlines()]
    if any(row.get("schema", 1) != 1 for row in records):
        raise ValueError(f"Incompatible schema: {path}")
    metrics = [row for row in records if row["kind"] == "metrics"]
    if [row["iteration"] for row in metrics] != [1, 2, 3, 4, 5]:
        raise ValueError(f"Missing/duplicated iterations in {path}")
    if not any(row["kind"] == "diagnostic_complete" for row in records):
        raise ValueError(f"No completed diagnostic marker: {path}")
    return records


def compare(left: Path, right: Path, *, expected_ranks: int) -> dict:
    """Report boundary mismatches by rank; a missing record is not a value match."""
    expected = {f"rank{rank:05d}.jsonl" for rank in range(expected_ranks)}
    for directory in (left, right):
        actual = {path.name.removesuffix(".gz") for path in directory.glob("rank*.jsonl*")}
        if actual != expected:
            raise ValueError(
                f"Rank coverage differs: {directory}: missing={expected - actual}, extra={actual - expected}"
            )
    reports = []
    metrics = []
    for filename in sorted(expected):

        def source(directory):
            path = directory / filename
            return path if path.exists() else path.with_suffix(path.suffix + ".gz")

        a, b = _load(source(left)), _load(source(right))
        events_a = [row for row in a if row["kind"] == "boundary"]
        events_b = [row for row in b if row["kind"] == "boundary"]

        def key(row):
            return row["iteration"], row["execution"], row["phase"], row["name"], row["occurrence"]

        index_b = {key(row): row for row in events_b}
        if len(index_b) != len(events_b) or len({key(row) for row in events_a}) != len(events_a):
            raise ValueError(f"Ambiguous duplicated event identities in {filename}")
        mismatches = []
        for row in events_a:
            other = index_b.pop(key(row), None)
            if other is None:
                mismatches.append({"kind": "missing_right", "identity": key(row)})
            elif any(row[field] != other[field] for field in ("shape", "dtype", "numel", "digest")):
                mismatches.append({"kind": "boundary_screen_difference", "identity": key(row), "a": row, "b": other})
        mismatches.extend({"kind": "missing_left", "identity": identity} for identity in index_b)
        reports.append(
            {
                "rank": int(filename[4:9]),
                "mismatch_count": len(mismatches),
                "first_observed_boundary_difference": mismatches[0] if mismatches else None,
            }
        )
        ma = {row["iteration"]: row for row in a if row["kind"] == "metrics"}
        mb = {row["iteration"]: row for row in b if row["kind"] == "metrics"}
        for iteration in range(1, 6):
            metrics.append(
                {"rank": int(filename[4:9]), "iteration": iteration, "a": ma[iteration], "b": mb[iteration]}
            )
    return {
        "scope": "Rank-local first observed boundary screens, not exact op-level causes. No cross-rank seq ordering.",
        "ranks": reports,
        "metrics": metrics,
    }


def main() -> None:
    """Require complete evidence before producing a comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--ranks", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.left, args.right, expected_ranks=args.ranks), indent=2) + "\n")


if __name__ == "__main__":
    main()
