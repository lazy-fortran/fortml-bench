#!/usr/bin/env python3
"""Correctness-gated LightGBM multiclass log-probability benchmark.

The NumPy side independently checks sorted-label OVR log-sum-exp and the
finite probability tail. The FortML release workload checks input and packed
leaf-coordinate products and records the explicit CUDA refusal.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "seconds", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[float, float]:
    labels = np.array([11, -8, 2], dtype=np.int64)
    if not np.array_equal(np.sort(labels), np.array([-8, 2, 11])):
        raise RuntimeError("sorted arbitrary-label oracle failed")
    margins = np.array([[-1000.0, 0.0, 1000.0], [2.0, -1.0, 0.5]], dtype=np.float64)
    log_positive = np.empty_like(margins)
    nonnegative = margins >= 0.0
    log_positive[nonnegative] = -np.log1p(np.exp(-margins[nonnegative]))
    log_positive[~nonnegative] = margins[~nonnegative] - np.log1p(
        np.exp(margins[~nonnegative]))
    log_normalization = np.logaddexp.reduce(log_positive, axis=1, keepdims=True)
    log_probability = log_positive - log_normalization
    probability = np.exp(log_probability)
    roundtrip = float(np.max(np.abs(probability.sum(axis=1) - 1.0)))
    expected_tail = np.array([-1000.0 - np.log(1.5), -np.log(3.0), -np.log(1.5)])
    tail_error = float(np.max(np.abs(log_probability[0] - expected_tail)))
    if roundtrip > 3.0e-15 or tail_error > 3.0e-13:
        raise RuntimeError(f"stable log-probability oracle failed: {roundtrip}, {tail_error}")
    return roundtrip, tail_error


def parse_release(stdout: str) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[0].startswith("lgbm_mc_log_"):
            continue
        key, value = fields
        parsed[key] = int(value) if key.endswith("status") else float(value)
    required = {
        "lgbm_mc_log_fit_seconds", "lgbm_mc_log_roundtrip_error",
        "lgbm_mc_log_input_jvp_error", "lgbm_mc_log_input_vjp_error",
        "lgbm_mc_log_parameter_vjp_error", "lgbm_mc_log_cuda_status",
    }
    missing = required.difference(parsed)
    if missing:
        raise RuntimeError(f"release app omitted fields: {sorted(missing)}")
    return parsed


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/lightgbm_multiclass_log_proba.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/LIGHTGBM_MULTICLASS_LOG_PROBA.md"))
    parser.add_argument("--target", default="fortml_bench_lightgbm_multiclass_log_proba")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output, report = args.output.resolve(), args.report.resolve()
    simplex_error, tail_error = oracle()
    env = os.environ.copy()
    env.update({"FO_SCAN_FALLBACK": "regex", "FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O2"], cwd=fortml, env=env, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml, env=env,
        check=True, capture_output=True, text=True,
    )
    observed = parse_release(completed.stdout)
    if float(observed["lgbm_mc_log_roundtrip_error"]) > 3.0e-13:
        raise RuntimeError("FortML log/simplex round trip exceeded tolerance")
    for key in ("lgbm_mc_log_input_jvp_error", "lgbm_mc_log_input_vjp_error",
                "lgbm_mc_log_parameter_vjp_error"):
        if float(observed[key]) > 4.0e-11:
            raise RuntimeError(f"FortML derivative oracle failed: {key}")
    if int(observed["lgbm_mc_log_cuda_status"]) != 3:
        raise RuntimeError("multiclass LightGBM CUDA refusal changed")

    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": env.get("FO_FC", "gfortran"), "flags": "-O2",
        "oracle": "independent NumPy stable OVR log-sum-exp and simplex",
    }
    rows = [
        row(details, workload="lightgbm_multiclass_log_proba", phase="contract_oracle",
            backend="numpy_oracle", device="cpu", status="pass",
            metric="simplex_max_abs_error", value=simplex_error,
            max_abs_error=max(simplex_error, tail_error), seconds=0.0,
            notes=f"stable tail log error={tail_error:.3e}; labels=[-8,2,11]"),
        row(details, workload="lightgbm_multiclass_log_proba", phase="fit",
            backend="fortml", device="cpu", status="pass",
            metric="fit_seconds", value=observed["lgbm_mc_log_fit_seconds"],
            seconds=observed["lgbm_mc_log_fit_seconds"],
            notes="four depth-one leaf-wise logistic trees per sorted OVR child"),
        row(details, workload="lightgbm_multiclass_log_proba", phase="predict",
            backend="fortml", device="cpu", status="pass",
            metric="log_simplex_roundtrip_error", value=observed["lgbm_mc_log_roundtrip_error"],
            max_abs_error=observed["lgbm_mc_log_roundtrip_error"],
            notes="exp(predict_log_proba) agrees with predict_proba"),
        row(details, workload="lightgbm_multiclass_log_proba", phase="derivatives",
            backend="fortml", device="cpu", status="pass",
            metric="max_input_parameter_product_error",
            value=max(float(observed["lgbm_mc_log_input_jvp_error"]),
                      float(observed["lgbm_mc_log_input_vjp_error"]),
                      float(observed["lgbm_mc_log_parameter_vjp_error"])),
            max_abs_error=max(float(observed["lgbm_mc_log_input_jvp_error"]),
                              float(observed["lgbm_mc_log_input_vjp_error"]),
                              float(observed["lgbm_mc_log_parameter_vjp_error"])),
            notes="input central difference/adjoint and packed leaf adjoint"),
        row(details, workload="lightgbm_multiclass_log_proba", phase="device_capability",
            backend="fortml", device="cuda", status="unavailable",
            metric="resident_multiclass_tree_log_probability", value="nan",
            max_abs_error="nan", oracle="typed FORTNUM_NOT_IMPLEMENTED refusal",
            notes="no host fallback is counted as GPU support"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Multiclass LightGBM log-probability products\n\n"
        "This lane checks stable sorted-label one-vs-rest `predict_log_proba`, "
        "input and packed leaf-coordinate JVP/VJP products, and explicit CPU/CUDA "
        "dispatch. The independent NumPy oracle exercises a probability tail "
        "that would underflow under `log(predict_proba)` and checks the simplex.\n\n"
        "Reproduce:\n\n"
        "```bash\n"
        "python -B scripts/bench_lightgbm_multiclass_log_proba.py "
        "--fortml ../fortml --output results/lightgbm_multiclass_log_proba.csv "
        "--report results/LIGHTGBM_MULTICLASS_LOG_PROBA.md\n"
        "```\n\n"
        "The CUDA row is `unavailable` with typed `FORTNUM_NOT_IMPLEMENTED`. No "
        "host fallback timing is reported.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
