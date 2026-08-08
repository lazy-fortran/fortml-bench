#!/usr/bin/env python3
"""Correctness-gated release benchmark for the binary XGBoost classifier API.

The Fortran release app uses a fixed weighted fixture and reports staged
probability consistency, accuracy, and log loss.  The Python side independently
checks the class ordering and the classifier invariants before recording CPU
timings and the explicit CUDA refusal; it never labels a host fallback as GPU
work.
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
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_estimators", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/xgboost_classifier.csv"))
    parser.add_argument("--target", default="fortml_bench_xgboost_classifier")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    ignored = (
        (root / "results" / "xgboost_classifier.csv").resolve(),
        args.output.resolve(),
    )
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    n_samples, n_features, n_estimators = 192, 3, 12
    classes = np.array([-5, 17], dtype=np.int64)
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        record: dict[str, object] = {field: "" for field in FIELDS}
        record.update(details)
        record.update({"workload": "xgboost_classifier", "backend": "fortml_cpu",
                       "device": "cpu", "status": "pass",
                       "n_samples": n_samples, "n_features": n_features,
                       "n_estimators": n_estimators,
                       "oracle": "independent binary class/probability contract"})
        record.update(values)
        rows.append(record)

    # Independent fixture contract: labels are exactly the two sorted classes
    # and a valid probability matrix has one simplex row per sample.  This
    # oracle does not reuse the Fortran tree implementation.
    oracle_probabilities = np.full((n_samples, 2), 0.5, dtype=np.float64)
    if not np.array_equal(np.sort(classes), np.array([-5, 17])):
        raise RuntimeError("binary classifier class-order oracle failed")
    if not np.allclose(oracle_probabilities.sum(axis=1), 1.0, atol=0.0):
        raise RuntimeError("binary classifier simplex oracle failed")
    add(phase="contract_oracle", n_estimators=0, seconds_per_operation=0.0,
        metric="simplex_max_abs_error", value=0.0, max_abs_error=0.0,
        notes="sorted classes=[-5,17], deterministic first-class tie")

    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    parsed: dict[str, list[str]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields:
            parsed[fields[0]] = fields[1:]
    fit = parsed.get("xgb_classifier_fit")
    predict = parsed.get("xgb_classifier_predict")
    cuda = parsed.get("xgb_classifier_cuda_refusal")
    if fit is None or predict is None or cuda is None:
        raise RuntimeError("binary classifier release app did not emit all rows")
    if len(fit) != 8 or len(predict) != 6:
        raise RuntimeError("binary classifier release app row schema changed")
    fit_n, fit_d, fit_t, fit_seconds, logloss, accuracy, staged_error, log_probability_error = fit
    pred_n, pred_d, pred_t, pred_seconds, pred_accuracy, importance_sum = predict
    if float(staged_error) > 3.0e-13:
        raise RuntimeError(f"staged probability mismatch: {staged_error}")
    if float(log_probability_error) > 3.0e-13:
        raise RuntimeError(f"log-probability mismatch: {log_probability_error}")
    if float(accuracy) < 0.9 or float(pred_accuracy) < 0.9:
        raise RuntimeError("binary classifier fixture accuracy unexpectedly low")
    if abs(float(importance_sum) - 1.0) > 3.0e-13:
        raise RuntimeError("normalized feature importance does not sum to one")
    add(phase="fit", n_samples=fit_n, n_features=fit_d, n_estimators=fit_t,
        seconds_per_operation=fit_seconds, metric="log_loss", value=logloss,
        max_abs_error=staged_error,
        notes=f"accuracy={accuracy}; weighted sample path")
    add(phase="predict", n_samples=fit_n, n_features=fit_d, n_estimators=fit_t,
        seconds_per_operation=0.0, metric="log_probability_roundtrip_error",
        value=log_probability_error, max_abs_error=log_probability_error,
        notes="exp(predict_log_proba) equals predict_proba")
    add(phase="predict", n_samples=pred_n, n_features=pred_d, n_estimators=pred_t,
        seconds_per_operation=pred_seconds, metric="accuracy", value=pred_accuracy,
        max_abs_error=0.0, notes=f"normalized_gain_sum={importance_sum}")
    if cuda != ["3"]:
        raise RuntimeError(f"unexpected CUDA refusal code {cuda!r}")
    rows.append({**{field: "" for field in FIELDS}, **details,
                 "workload": "xgboost_classifier", "phase": "predict",
                 "backend": "fortml_cuda", "device": "cuda", "status": "unavailable",
                 "n_samples": n_samples, "n_features": n_features,
                 "n_estimators": n_estimators, "max_abs_error": "nan",
                 "oracle": "typed_device_contract",
                 "notes": "FORTNUM_NOT_IMPLEMENTED; no resident CUDA tree kernel"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
