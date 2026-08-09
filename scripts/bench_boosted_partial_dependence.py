#!/usr/bin/env python3
"""Correctness-gated boosted-tree partial-dependence benchmark."""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "dimensions",
    "evaluations", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)
FORTNUM_NOT_IMPLEMENTED = 3
TOLERANCE = 2.0e-12


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    )
    dirty = any(line[3:].strip() not in ignored_names for line in status.splitlines())
    return value + ("+dirty" if dirty else "")


def run_app(fortml: Path) -> dict[str, str]:
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_boosted_partial_dependence"],
        cwd=fortml, check=True, capture_output=True, text=True,
    )
    records: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0] in {
            "xgb_pdp_seconds", "xgb_pdp_values", "xgb_pdp_link_error",
            "lgbm_pdp_seconds", "lgbm_pdp_values", "lgbm_pdp_link_error",
            "xgb_pdp_cuda", "lgbm_pdp_cuda",
        }:
            records[fields[0]] = fields[1]
    required = {
        "xgb_pdp_seconds", "xgb_pdp_values", "xgb_pdp_link_error",
        "lgbm_pdp_seconds", "lgbm_pdp_values", "lgbm_pdp_link_error",
        "xgb_pdp_cuda", "lgbm_pdp_cuda",
    }
    if set(records) != required:
        raise RuntimeError(f"release app fields missing: {required - set(records)}")
    return records


def independent_oracle() -> np.ndarray:
    """Return weighted PDP followed by column-major ICE for the stump fixture."""
    left = 5.0 - 10.0 / 3.0
    right = 5.0 + 10.0 / 3.0
    ice_column = np.array((left, left, right, right), dtype=np.float64)
    weights = np.array((1.0, 1.0, 1.0, 3.0), dtype=np.float64)
    weighted_average = float(np.dot(weights, ice_column) / np.sum(weights))
    return np.concatenate((
        np.full(2, weighted_average),
        np.column_stack((ice_column, ice_column)).ravel(order="F"),
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/boosted_partial_dependence.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/BOOSTED_PARTIAL_DEPENDENCE.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    records = run_app(fortml)
    expected = independent_oracle()
    fortml_rev = revision(
        fortml, (fortml / "verification" / "fortml-gfortran.txt",),
    )
    bench_rev = revision(root, (args.output, args.report))
    rows: list[dict[str, str]] = []

    def row(**kwargs: object) -> dict[str, str]:
        output = {field: "" for field in FIELDS}
        output.update({
            "workload": "boosted_partial_dependence",
            "backend": "fortml",
            "device": "cpu",
            "status": "pass",
            "dimensions": "4x2; grid=2",
            "compiler": "gfortran",
            "flags": "-O3",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "fortml_revision": fortml_rev,
            "benchmark_revision": bench_rev,
            "oracle": "independent_numpy_weighted_newton_stump",
        })
        output.update({key: str(value) for key, value in kwargs.items()})
        return output

    for prefix, family in (("xgb", "xgboost"), ("lgbm", "lightgbm")):
        observed = np.fromstring(records[f"{prefix}_pdp_values"], sep=" ")
        if observed.shape != expected.shape:
            raise RuntimeError(f"{family} PDP/ICE output shape changed: {observed.shape}")
        error = float(np.max(np.abs(observed - expected)))
        if error > TOLERANCE:
            raise RuntimeError(f"{family} weighted PDP/ICE oracle error {error:.3e}")
        link_error = float(records[f"{prefix}_pdp_link_error"])
        if link_error > TOLERANCE:
            raise RuntimeError(f"{family} response-link error {link_error:.3e}")
        cuda_status = int(records[f"{prefix}_pdp_cuda"])
        if cuda_status != FORTNUM_NOT_IMPLEMENTED:
            raise RuntimeError(f"{family} CUDA refusal status changed: {cuda_status}")
        rows.append(row(phase=f"{family}_pdp_ice", evaluations=2,
                        seconds_per_operation=float(records[f"{prefix}_pdp_seconds"]),
                        metric="max_abs_error", value=error, max_abs_error=error,
                        notes="weighted brute-force PDP plus complete ICE matrix"))
        rows.append(row(phase=f"{family}_response_link", evaluations=2,
                        seconds_per_operation=0.0, metric="max_abs_error",
                        value=link_error, max_abs_error=link_error,
                        oracle="numpy_logistic_of_raw_margin",
                        notes="transformed prediction and raw-margin response selectors"))
        rows.append(row(phase=f"{family}_pdp_ice", device="cuda",
                        status="unavailable", evaluations=0,
                        seconds_per_operation=0.0, metric="capability", value=0,
                        max_abs_error="", oracle="typed_device_contract",
                        notes="FORTNUM_NOT_IMPLEMENTED; no resident CUDA tree PDP kernel"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = """# Boosted-tree partial dependence

This lane checks weighted one-feature partial dependence and individual
conditional expectation values for the XGBoost-style and LightGBM-style tree
estimators. The four-row fixture fits one Newton stump. Its leaf predictions
are `5/3` and `25/3`. Weights `[1, 1, 1, 3]` give a partial-dependence value of
`55/9` when the intervention changes an unused feature.

NumPy constructs the full expected PDP and ICE arrays from those analytic leaf
values and weights. A separate check compares transformed binary predictions
with the logistic transform of raw margins. CUDA rows record the typed refusal
because no resident boosted-tree PDP kernel is linked.

Reproduce:

```bash
.venv/bin/python -B scripts/bench_boosted_partial_dependence.py \
  --fortml ../fortml --output results/boosted_partial_dependence.csv \
  --report results/BOOSTED_PARTIAL_DEPENDENCE.md
```

Raw data: [`boosted_partial_dependence.csv`](boosted_partial_dependence.csv).
"""
    args.report.write_text(report)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
