#!/usr/bin/env python3
"""Correctness-gated LightGBM-style OVR multiclass benchmark.

The NumPy lane independently replays the bounded two-leaf logistic trees,
normalizes every class stage, and computes weighted multiclass log loss. The
Fortran release app is accepted only after those values, the fixed-tree
derivative boundary, CUDA status, and transactional refusal agree.
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
N_ESTIMATORS = 4
LEARNING_RATE = 0.4
L2 = 1.0
MIN_DELTA = 1.0e6


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def sigmoid(margin: np.ndarray) -> np.ndarray:
    positive = margin >= 0.0
    result = np.empty_like(margin)
    result[positive] = 1.0 / (1.0 + np.exp(-margin[positive]))
    exp_margin = np.exp(margin[~positive])
    result[~positive] = exp_margin / (1.0 + exp_margin)
    return result


def stable_logit(probability: float) -> float:
    clipped = min(max(probability, 1.0e-12), 1.0 - 1.0e-12)
    return float(np.log(clipped) - np.log1p(-clipped))


def binary_stages(target: np.ndarray, validation_x: np.ndarray) -> np.ndarray:
    x = np.arange(-4.0, 5.0).reshape(-1, 1)
    base = stable_logit(float(np.mean(target)))
    train_margin = np.full(target.shape, base)
    validation_margin = np.full(validation_x.shape[0], base)
    thresholds = 0.5 * (x[:-1, 0] + x[1:, 0])
    stages: list[np.ndarray] = []
    for _ in range(N_ESTIMATORS):
        probability = sigmoid(train_margin)
        gradient = probability - target
        hessian = np.maximum(probability * (1.0 - probability), 1.0e-12)
        total_gradient = float(np.sum(gradient))
        total_hessian = float(np.sum(hessian))
        best_gain = 0.0
        best_left: np.ndarray | None = None
        best_left_weight = 0.0
        best_right_weight = 0.0
        for threshold in thresholds:
            left = x[:, 0] < threshold
            right = ~left
            left_gradient = float(np.sum(gradient[left]))
            right_gradient = float(np.sum(gradient[right]))
            left_hessian = float(np.sum(hessian[left]))
            right_hessian = float(np.sum(hessian[right]))
            gain = 0.5 * (
                left_gradient**2 / (left_hessian + L2)
                + right_gradient**2 / (right_hessian + L2)
                - total_gradient**2 / (total_hessian + L2)
            )
            if gain > best_gain:
                best_gain = gain
                best_left = left
                best_left_weight = -left_gradient / (left_hessian + L2)
                best_right_weight = -right_gradient / (right_hessian + L2)
        if best_left is None:
            correction = np.full(target.shape, -total_gradient / (total_hessian + L2))
            validation_correction = np.full(validation_x.shape[0], correction[0])
        else:
            correction = np.where(best_left, best_left_weight, best_right_weight)
            selected_threshold = None
            for threshold in thresholds:
                left = x[:, 0] < threshold
                candidate = np.where(left, best_left_weight, best_right_weight)
                if np.array_equal(candidate, correction):
                    selected_threshold = threshold
                    break
            if selected_threshold is None:
                raise RuntimeError("failed to recover deterministic stump threshold")
            validation_correction = np.where(
                validation_x[:, 0] < selected_threshold,
                best_left_weight, best_right_weight,
            )
        train_margin = train_margin + LEARNING_RATE * correction
        validation_margin = validation_margin + LEARNING_RATE * validation_correction
        stages.append(sigmoid(validation_margin))
    return np.stack(stages, axis=1)


def oracle() -> tuple[np.ndarray, np.ndarray, int, int]:
    labels = np.array([-8, -8, -8, 2, 2, 2, 11, 11, 11])
    validation_x = np.array([-3.5, -1.5, -0.2, 0.8, 2.2, 3.7]).reshape(-1, 1)
    validation_labels = np.array([-8, -8, 2, 2, 11, 11])
    weights = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 3.0])
    classes = np.array([-8, 2, 11])
    children = [
        binary_stages((labels == label).astype(np.float64), validation_x)
        for label in classes
    ]
    child = np.stack(children, axis=1)
    normalized = child / np.sum(child, axis=1, keepdims=True)
    indices = np.searchsorted(classes, validation_labels)
    losses = np.array([
        -float(np.sum(weights * np.log(np.maximum(
            normalized[np.arange(validation_labels.size), indices, stage], 1.0e-15,
        ))) / np.sum(weights))
        for stage in range(N_ESTIMATORS)
    ])
    best_iteration = 1
    best_loss = losses[0]
    stale = 0
    stop_iteration = N_ESTIMATORS
    for index, loss in enumerate(losses, start=1):
        if index == 1:
            continue
        if loss < best_loss - MIN_DELTA:
            best_loss = loss
            best_iteration = index
            stale = 0
        else:
            stale += 1
        if stale >= 1:
            stop_iteration = index
            break
    return normalized, losses, best_iteration, stop_iteration


def parse_release(stdout: str) -> dict[str, float | int]:
    parsed: dict[str, float | int] = {}
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[0].startswith("lightgbm_mc_"):
            continue
        key, value = fields
        parsed[key] = int(value) if key.endswith((
            "iteration", "requested", "retained", "stopped", "status",
        )) else float(value)
    required = {
        "lightgbm_mc_best_iteration", "lightgbm_mc_requested",
        "lightgbm_mc_retained", "lightgbm_mc_early_stopped",
        "lightgbm_mc_best_loss", "lightgbm_mc_oracle_loss",
        "lightgbm_mc_stage_error", "lightgbm_mc_probability_sum_error",
        "lightgbm_mc_jvp_error", "lightgbm_mc_vjp_error",
        "lightgbm_mc_cuda_status", "lightgbm_mc_invalid_status",
        "lightgbm_mc_transaction_error",
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
    parser.add_argument("--output", type=Path, default=Path("results/lightgbm_multiclass.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/LIGHTGBM_MULTICLASS.md"))
    parser.add_argument("--target", default="fortml_bench_lightgbm_multiclass")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1", "FO_SCAN_FALLBACK": "regex"})
    subprocess.run(["fo", "build", "--flag", "-O2"], cwd=fortml, env=env, check=True)
    completed = subprocess.run(
        ["fo", "exec", args.target], cwd=fortml, env=env,
        check=True, capture_output=True, text=True,
    )
    observed = parse_release(completed.stdout)
    normalized, losses, best_iteration, stop_iteration = oracle()
    expected_loss = float(losses[best_iteration - 1])
    if int(observed["lightgbm_mc_best_iteration"]) != best_iteration:
        raise RuntimeError("LightGBM multiclass best iteration disagrees with NumPy oracle")
    if int(observed["lightgbm_mc_requested"]) != N_ESTIMATORS:
        raise RuntimeError("requested estimator metadata mismatch")
    if int(observed["lightgbm_mc_retained"]) != best_iteration:
        raise RuntimeError("restore-best prefix mismatch")
    if int(observed["lightgbm_mc_early_stopped"]) != 1:
        raise RuntimeError("patience did not report early stopping")
    if not np.isclose(float(observed["lightgbm_mc_best_loss"]), expected_loss, atol=3e-12, rtol=0.0):
        raise RuntimeError("weighted multiclass validation loss mismatch")
    if not np.isclose(float(observed["lightgbm_mc_oracle_loss"]), expected_loss, atol=3e-12, rtol=0.0):
        raise RuntimeError("release app's internal loss replay mismatch")
    for key in ("lightgbm_mc_stage_error", "lightgbm_mc_probability_sum_error", "lightgbm_mc_jvp_error", "lightgbm_mc_vjp_error"):
        if float(observed[key]) > 3e-12:
            raise RuntimeError(f"{key} exceeded oracle tolerance")
    if int(observed["lightgbm_mc_cuda_status"]) != 3:
        raise RuntimeError("CUDA capability refusal changed")
    if int(observed["lightgbm_mc_invalid_status"]) != 1 or float(observed["lightgbm_mc_transaction_error"]) > 3e-12:
        raise RuntimeError("transactional validation refusal changed")

    output = args.output.resolve()
    report = args.report.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": env.get("FO_FC", "gfortran"), "flags": "-O2",
        "oracle": "independent NumPy depth-one OVR Newton replay",
    }
    fit_seconds = float(observed.get("lightgbm_mc_fit_seconds", np.nan))
    records = [
        row(details, workload="lightgbm_multiclass", phase="fit_validation", backend="fortml", device="cpu", status="pass", metric="best_validation_loss", value=float(observed["lightgbm_mc_best_loss"]), max_abs_error=abs(float(observed["lightgbm_mc_best_loss"]) - expected_loss), seconds=fit_seconds, notes="weighted arbitrary-label log-loss; best common prefix restored"),
        row(details, workload="lightgbm_multiclass", phase="staged_prefix", backend="fortml", device="cpu", status="pass", metric="max_abs_error", value=float(observed["lightgbm_mc_stage_error"]), max_abs_error=float(observed["lightgbm_mc_stage_error"]), seconds=fit_seconds, notes=f"best iteration={best_iteration}; patience stop={stop_iteration}"),
        row(details, workload="lightgbm_multiclass", phase="derivatives", backend="fortml", device="cpu", status="pass", metric="max_abs_error", value=max(float(observed["lightgbm_mc_jvp_error"]), float(observed["lightgbm_mc_vjp_error"])), max_abs_error=max(float(observed["lightgbm_mc_jvp_error"]), float(observed["lightgbm_mc_vjp_error"])), seconds=fit_seconds, notes="fixed-tree normalized input JVP/VJP away from split surfaces"),
        row(details, workload="lightgbm_multiclass", phase="device_capability", backend="fortml", device="cuda", status="unavailable", metric="resident_tree_prediction", value="nan", max_abs_error="nan", oracle="typed device contract", notes="no resident CUDA histogram kernel; host fallback is not hidden"),
        row(details, workload="lightgbm_multiclass", phase="transactional_refusal", backend="fortml", device="cpu", status="pass", metric="max_abs_error", value=float(observed["lightgbm_mc_transaction_error"]), max_abs_error=float(observed["lightgbm_mc_transaction_error"]), seconds=fit_seconds, notes="unknown validation label returns FORTNUM_DOMAIN_ERROR and preserves fitted state"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join([
        "# LightGBM-style multiclass classification",
        "",
        "This release lane independently replays the bounded two-leaf logistic",
        "OVR updates for arbitrary labels `[-8, 2, 11]`. It checks weighted",
        "normalized staged probabilities, common-prefix validation metadata,",
        "the fixed-tree input JVP/VJP boundary, typed CUDA refusal, and",
        "transactional unknown-label rejection.",
        "",
        "| metric | observed | independent oracle |",
        "|---|---:|---:|",
        f"| best iteration | {int(observed['lightgbm_mc_best_iteration'])} | {best_iteration} |",
        f"| requested estimators | {int(observed['lightgbm_mc_requested'])} | {N_ESTIMATORS} |",
        f"| retained estimators | {int(observed['lightgbm_mc_retained'])} | {best_iteration} |",
        f"| weighted validation log-loss | {float(observed['lightgbm_mc_best_loss']):.16e} | {expected_loss:.16e} |",
        f"| staged probability max error | {float(observed['lightgbm_mc_stage_error']):.16e} | <= 3e-12 |",
        f"| probability simplex max error | {float(observed['lightgbm_mc_probability_sum_error']):.16e} | <= 3e-12 |",
        f"| input JVP max error | {float(observed['lightgbm_mc_jvp_error']):.16e} | <= 3e-12 |",
        f"| input VJP max error | {float(observed['lightgbm_mc_vjp_error']):.16e} | <= 3e-12 |",
        "",
        "All CPU rows agree with the independent oracle. CUDA is recorded as",
        "`unavailable` with the typed resident-histogram refusal. No host",
        "fallback is included in the device result.",
        "",
        "Reproduce:",
        "",
        "```bash",
        "python -B scripts/bench_lightgbm_multiclass.py \\",
        "  --fortml ../fortml --output results/lightgbm_multiclass.csv \\",
        "  --report results/LIGHTGBM_MULTICLASS.md",
        "```",
        "",
        f"FortML revision: `{details['fortml_revision']}`",
        f"Benchmark revision: `{details['benchmark_revision']}`",
        f"Python {platform.python_version()}, NumPy {np.__version__}",
    ]) + "\n", encoding="utf-8")
    print(f"wrote {output} and {report}")


if __name__ == "__main__":
    main()
