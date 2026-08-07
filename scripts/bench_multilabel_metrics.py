#!/usr/bin/env python3
"""Correctness-gated multilabel-metric and ROC-AUC benchmark."""

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
    "n_labels", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int64)
    predictions = np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
    probabilities = np.array([[0.5, 0.2, 0.8], [0.1, 0.5, 0.2], [0.2, 0.1, 0.3]], dtype=np.float64)
    return labels, predictions, probabilities


def scores(labels: np.ndarray, predictions: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.sum((labels == 1) & (predictions == 1), axis=0).astype(np.float64)
    fp = np.sum((labels == 0) & (predictions == 1), axis=0).astype(np.float64)
    fn = np.sum((labels == 1) & (predictions == 0), axis=0).astype(np.float64)

    def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)

    macro = np.array([ratio(tp, tp + fp).mean(), ratio(tp, tp + fn).mean(),
                      ratio(2.0 * tp, 2.0 * tp + fp + fn).mean()])
    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro = np.array([micro_tp / (micro_tp + micro_fp), micro_tp / (micro_tp + micro_fn),
                      2.0 * micro_tp / (2.0 * micro_tp + micro_fp + micro_fn)])
    beta_squared = 4.0
    micro_fbeta = np.array([
        micro_tp / (micro_tp + micro_fp),
        micro_tp / (micro_tp + micro_fn),
        (1.0 + beta_squared) * micro_tp /
        ((1.0 + beta_squared) * micro_tp + beta_squared * micro_fn + micro_fp),
    ])
    rows = []
    for y_true, y_pred in zip(labels, predictions):
        row_tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        row_fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        row_fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        rows.append([0.0 if row_tp + row_fp == 0 else row_tp / (row_tp + row_fp),
                     0.0 if row_tp + row_fn == 0 else row_tp / (row_tp + row_fn),
                     0.0 if 2.0 * row_tp + row_fp + row_fn == 0 else 2.0 * row_tp / (2.0 * row_tp + row_fp + row_fn)])
    fbeta_rows = []
    for y_true, y_pred in zip(labels, predictions):
        row_tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        row_fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        row_fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        fbeta_rows.append([
            0.0 if row_tp + row_fp == 0 else row_tp / (row_tp + row_fp),
            0.0 if row_tp + row_fn == 0 else row_tp / (row_tp + row_fn),
            0.0 if (1.0 + beta_squared) * row_tp + beta_squared * row_fn + row_fp == 0
            else (1.0 + beta_squared) * row_tp /
            ((1.0 + beta_squared) * row_tp + beta_squared * row_fn + row_fp),
        ])
    samples = np.mean(np.asarray(rows), axis=0)
    fbeta_samples = np.mean(np.asarray(fbeta_rows), axis=0)
    jaccard = []
    hamming = []
    for average in ("micro", "macro", "samples"):
        if average == "micro":
            intersection = float(np.sum((labels == 1) & (predictions == 1)))
            union = float(np.sum((labels == 1) | (predictions == 1)))
            jaccard.append(0.0 if union == 0.0 else intersection / union)
            hamming.append(float(np.mean(labels != predictions)))
        elif average == "macro":
            intersections = np.sum((labels == 1) & (predictions == 1), axis=0)
            unions = np.sum((labels == 1) | (predictions == 1), axis=0)
            jaccard.append(float(np.mean(np.divide(intersections, unions,
                out=np.zeros_like(intersections, dtype=np.float64), where=unions > 0))))
            hamming.append(float(np.mean(np.mean(labels != predictions, axis=0))))
        else:
            row_intersection = np.sum((labels == 1) & (predictions == 1), axis=1)
            row_union = np.sum((labels == 1) | (predictions == 1), axis=1)
            jaccard.append(float(np.mean(np.divide(row_intersection, row_union,
                out=np.zeros_like(row_intersection, dtype=np.float64), where=row_union > 0))))
            hamming.append(float(np.mean(np.mean(labels != predictions, axis=1))))
    return {"micro": micro, "macro": macro, "samples": samples,
            "fbeta_micro": micro_fbeta, "fbeta_samples": fbeta_samples,
            "jaccard_micro": np.asarray([jaccard[0]]),
            "jaccard_macro": np.asarray([jaccard[1]]),
            "jaccard_samples": np.asarray([jaccard[2]]),
            "hamming_micro": np.asarray([hamming[0]]),
            "hamming_macro": np.asarray([hamming[1]]),
            "hamming_samples": np.asarray([hamming[2]])}


def parse(output: str) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0].startswith("multilabel_"):
            values[fields[0]] = np.asarray([float(field) for field in fields[1:]], dtype=np.float64)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/multilabel_metrics.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    labels, predictions, probabilities = fixture()
    expected = scores(labels, predictions)
    threshold_predictions = (probabilities >= 0.5).astype(np.int64)
    expected["threshold"] = scores(labels, threshold_predictions)["micro"]
    completed = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_multilabel_metrics"], cwd=fortml, check=True, capture_output=True, text=True)
    observed = parse(completed.stdout)
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, tuple((root / "results" / name).resolve()
                                                    for name in ("multilabel_metrics.csv",
                                                                 "roc_auc.csv",
                                                                 "device_contracts.csv"))),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows = []
    for phase in ("micro", "macro", "samples", "threshold"):
        actual = observed[f"multilabel_{phase}"]
        error = float(np.max(np.abs(actual - expected[phase])))
        if error > 2.0e-14:
            raise RuntimeError(f"multilabel {phase} mismatch: {error:.3e}")
        for metric, value, metric_error in zip(("precision", "recall", "f1"), actual, np.atleast_1d(actual - expected[phase])):
            rows.append({**metadata, "workload": "multilabel_metrics", "phase": phase,
                "backend": "fortml", "device": "cpu", "status": "pass", "n_samples": labels.shape[0],
                "n_labels": labels.shape[1], "seconds_per_operation": "", "metric": metric,
                "value": value, "max_abs_error": abs(float(metric_error)),
                "oracle": "independent NumPy multilabel TP/FP/FN oracle", "notes": "explicit zero-division and >= threshold"})
    for phase in ("fbeta_micro", "fbeta_samples"):
        actual = observed[f"multilabel_{phase}"]
        error = float(np.max(np.abs(actual - expected[phase])))
        if error > 2.0e-14:
            raise RuntimeError(f"multilabel {phase} mismatch: {error:.3e}")
        for metric, value, metric_error in zip(("precision", "recall", "fbeta_beta2"), actual,
                                                actual - expected[phase]):
            rows.append({**metadata, "workload": "multilabel_metrics", "phase": phase,
                "backend": "fortml", "device": "cpu", "status": "pass", "n_samples": labels.shape[0],
                "n_labels": labels.shape[1], "seconds_per_operation": "", "metric": metric,
                "value": value, "max_abs_error": abs(float(metric_error)),
                "oracle": "independent NumPy weighted F-beta (beta=2) oracle",
                "notes": "micro and samples reductions; explicit zero-division policy"})
    for metric in ("jaccard_micro", "jaccard_macro", "jaccard_samples",
                   "hamming_micro", "hamming_macro", "hamming_samples"):
        actual = observed[f"multilabel_{metric}"]
        error = float(np.max(np.abs(actual - expected[metric])))
        if error > 2.0e-14:
            raise RuntimeError(f"multilabel {metric} mismatch: {error:.3e}")
        rows.append({**metadata, "workload": "multilabel_metrics", "phase": metric,
            "backend": "fortml", "device": "cpu", "status": "pass", "n_samples": labels.shape[0],
            "n_labels": labels.shape[1], "seconds_per_operation": "", "metric": metric.split("_", 1)[0],
            "value": float(actual[0]), "max_abs_error": error,
            "oracle": "independent NumPy multilabel intersection/union/error oracle",
            "notes": "explicit empty-union and row-weight policy"})
    rows.append({**metadata, "workload": "multilabel_metrics", "phase": "all",
        "backend": "fortml", "device": "cuda", "status": "unavailable",
        "n_samples": labels.shape[0], "n_labels": labels.shape[1],
        "seconds_per_operation": "", "metric": "precision/recall/F1/F-beta/Jaccard/Hamming",
        "value": "", "max_abs_error": "", "oracle": "typed_device_contract",
        "notes": "classification_multilabel_*_device is not implemented; no host fallback"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
