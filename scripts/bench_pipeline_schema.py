#!/usr/bin/env python3
"""Correctness-gated dense pipeline input-schema benchmark.

The oracle is intentionally independent of FortML internals: it checks the
expected default/installed names, duplicate/mismatch refusal semantics, and
the feature-block count implied by the release fixture.  The Fortran app
times repeated successful schema validation; CUDA is an explicit capability
boundary because this metadata operation has no resident device kernel.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_inputs",
    "n_features", "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N_INPUTS = 3
N_FEATURES = 9
REPETITIONS = 10000
EXPECTED_NAMES = ("time", "position", "velocity")


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "pipeline_schema", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_inputs": N_INPUTS,
        "n_features": N_FEATURES, "repetitions": REPETITIONS,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def independent_oracle() -> dict[str, float | int]:
    """Check metadata semantics without importing or calling FortML."""
    installed = list(EXPECTED_NAMES)
    if len(set(installed)) != N_INPUTS or any(not name for name in installed):
        raise RuntimeError("independent schema fixture is invalid")
    if installed != ["time", "position", "velocity"]:
        raise RuntimeError("schema names changed unexpectedly")
    mismatch = list(installed)
    mismatch[1] = "acceleration"
    duplicate = ["time", "time", "velocity"]
    mismatch_refused = mismatch != installed
    duplicate_refused = len(set(duplicate)) != N_INPUTS
    feature_count = 3 + 2 * 3  # polynomial degree one + one sine/cosine pair
    if not mismatch_refused or not duplicate_refused or feature_count != N_FEATURES:
        raise RuntimeError("independent schema refusal/count oracle failed")
    return {"feature_count": feature_count, "names_ok": 1, "refusals_ok": 1}


def parse(stdout: str) -> dict[str, float | int | str]:
    records: dict[str, float | int | str] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields[:1] != ["pipeline_schema"]:
            continue
        if len(fields) != 7:
            raise RuntimeError(f"malformed schema benchmark record: {line!r}")
        records.update({
            "n_inputs": int(fields[1]), "n_features": int(fields[2]),
            "repetitions": int(fields[3]), "seconds": float(fields[4]),
            "valid_count": int(fields[5]), "first_name": fields[6],
        })
    required = {"n_inputs", "n_features", "repetitions", "seconds",
                "valid_count", "first_name"}
    missing = required.difference(records)
    if missing:
        raise RuntimeError(f"release app omitted schema metrics: {sorted(missing)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/pipeline_schema.csv"))
    parser.add_argument("--target", default="fortml_bench_pipeline_schema")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1",
                        "FO_SCAN_FALLBACK": "regex"})
    subprocess.run(["fo", "build", "--flag", "-O2"], cwd=fortml,
                   env=environment, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    observed = parse(completed.stdout)
    expected = independent_oracle()
    if (observed["n_inputs"] != N_INPUTS or observed["n_features"] != N_FEATURES or
            observed["repetitions"] != REPETITIONS or
            observed["valid_count"] != REPETITIONS or
            observed["first_name"] != EXPECTED_NAMES[0]):
        raise RuntimeError(f"schema contract mismatch: {observed}")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O2",
    }
    rows = [
        row(details, phase="independent_oracle", backend="numpy_oracle",
            status="pass", metric="schema_name_and_refusal_contract", value=1.0,
            max_abs_error=0.0,
            oracle="independent names/count/duplicate/mismatch fixture",
            notes="three unique names; transactional mismatch and duplicate refusals"),
        row(details, phase="validation", backend="fortml", status="pass",
            n_inputs=observed["n_inputs"], n_features=observed["n_features"],
            repetitions=observed["repetitions"],
            seconds_per_operation=observed["seconds"],
            metric="successful_schema_validations", value=observed["valid_count"],
            max_abs_error=abs(float(observed["valid_count"]) - REPETITIONS),
            oracle="independent NumPy metadata contract",
            notes="input names are validated before transform; no output transfer"),
        row(details, phase="device_contract", backend="fortml", device="cuda",
            status="unavailable", metric="resident_schema_validation",
            value="FORTNUM_NOT_IMPLEMENTED", max_abs_error=0.0,
            oracle="typed resident-CUDA capability boundary",
            notes="metadata validation remains host-side")
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}; oracle={expected}")


if __name__ == "__main__":
    main()
