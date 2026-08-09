#!/usr/bin/env python3
"""Validate benchmark CSVs against the release result schema v1."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


REQUIRED = {
    "workload", "phase", "backend", "device", "status", "metric",
    "value", "max_abs_error", "oracle", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
}
STATUSES = {"pass", "failed", "skipped", "unavailable", "refused", "conditional"}
DEVICES = {"cpu", "cuda", "openacc", "rocm", "metal", "tpu", "unknown"}


def finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate(path: Path, allow_dirty: bool) -> list[str]:
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = set(reader.fieldnames or ())
        missing = sorted(REQUIRED - header)
        if missing:
            return [f"{path}: missing required columns: {', '.join(missing)}"]
        for line_number, row in enumerate(reader, start=2):
            prefix = f"{path}:{line_number}"
            status = row["status"].strip().lower()
            device = row["device"].strip().lower()
            if status not in STATUSES:
                errors.append(f"{prefix}: invalid status {status!r}")
            if device not in DEVICES:
                errors.append(f"{prefix}: invalid device {device!r}")
            for field in ("workload", "phase", "backend", "metric", "oracle",
                          "fortml_revision", "benchmark_revision"):
                if not row[field].strip():
                    errors.append(f"{prefix}: empty {field}")
            if not allow_dirty and ("+dirty" in row["fortml_revision"] or
                                    "+dirty" in row["benchmark_revision"]):
                errors.append(f"{prefix}: dirty revision in release row")
            if status == "pass" and not finite_number(row["max_abs_error"]):
                errors.append(f"{prefix}: passing row has non-finite max_abs_error")
            if status in {"unavailable", "refused"} and not row["notes"].strip():
                errors.append(f"{prefix}: capability boundary needs notes")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true",
                        help="validate every results/*.csv file")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "results").glob("*.csv")) if args.all else args.csv
    if not paths:
        parser.error("provide CSV paths or --all")
    errors = [error for path in paths for error in validate(path, args.allow_dirty)]
    if errors:
        print("RESULT_SCHEMA_V1_FAIL")
        print("\n".join(errors))
        return 1
    print(f"RESULT_SCHEMA_V1_OK {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
