#!/usr/bin/env python3
"""Correctness-gated benchmark for FortML's differentiable neural losses.

The NumPy formulas are independent value/derivative oracles.  The release app
reports checksums for the same fixed fixture before its CPU timings are kept;
CUDA is recorded as an explicit capability refusal until resident loss kernels
exist.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "dimensions",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N = 64
REPETITIONS = 2048
EPS = 3.0e-12


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, ...]:
    indices = np.arange(1, N + 1, dtype=np.float64)
    logits = np.column_stack((np.sin(0.13 * indices), np.cos(0.07 * indices),
                               0.2 * np.sin(0.19 * indices)))
    cotangent = np.column_stack((0.7 * np.sin(0.08 * indices),
                                 -0.5 * np.cos(0.06 * indices),
                                 0.2 + 0.3 * np.sin(0.04 * indices)))
    targets = np.column_stack((np.where((indices.astype(int) % 3) == 0, 1.0, 0.0),
                               np.where((indices.astype(int) % 3) == 1, 1.0, 0.0),
                               np.where((indices.astype(int) % 3) == 2, 1.0, 0.0)))
    direction = np.column_stack((0.01 * np.sin(0.11 * indices),
                                 -0.02 * np.cos(0.17 * indices),
                                 np.full(N, 0.03)))
    prediction = logits[:, :1]
    target = 0.4 * np.sin(0.05 * indices)[:, None]
    weights = 0.5 + (indices.astype(int) % 7) / 7.0
    log_variance = np.column_stack((0.2 * np.sin(0.03 * indices),
                                    -0.1 * np.cos(0.05 * indices),
                                    0.15 * np.sin(0.07 * indices)))
    count_targets = np.column_stack(((indices.astype(int) % 5).astype(float),
                                     ((indices.astype(int) + 1) % 5).astype(float),
                                     0.5 + ((indices.astype(int) + 2) % 4)))
    variance_direction = np.column_stack((0.02 * np.cos(0.09 * indices),
                                          -0.01 * np.sin(0.08 * indices),
                                          np.full(N, 0.015)))
    return (logits, targets, direction, cotangent, prediction, target, weights,
            log_variance, count_targets, variance_direction)


def oracle() -> dict[str, float]:
    (logits, targets, direction, cotangent, prediction, target, weights, log_variance,
     count_targets, variance_direction) = fixture()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    bce_hvp = probabilities * (1.0 - probabilities) * direction / logits.size
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax = exponent / exponent.sum(axis=1, keepdims=True)
    softmax_hvp = softmax * (direction -
                             (softmax * direction).sum(axis=1, keepdims=True)) / N
    softmax_direction = softmax * (direction -
                                   (softmax * direction).sum(axis=1, keepdims=True))
    softmax_hvp_vector = (softmax_direction *
                          (cotangent - (softmax * cotangent).sum(axis=1,
                                                                   keepdims=True)) -
                          softmax * (softmax_direction * cotangent).sum(axis=1,
                                                                         keepdims=True))
    log_softmax_hvp = -softmax * (direction -
                                  (softmax * direction).sum(axis=1, keepdims=True)) * \
        cotangent.sum(axis=1, keepdims=True)
    weighted_softmax_hvp = weights[:, None] * softmax * (direction -
        (softmax * direction).sum(axis=1, keepdims=True)) / weights.sum()
    weighted = weights[:, None] * direction[:, :1] / weights.sum()
    residual = prediction - target
    huber = np.where(np.abs(residual) < 0.75, direction[:, :1], 0.0) / N
    mae = (weights[:, None] * np.sign(residual) * direction[:, :1]).sum() / weights.sum()
    p = 1.0 / (1.0 + np.exp(-logits))
    bce = np.logaddexp(0.0, logits) - targets * logits
    pt = targets * p + (1.0 - targets) * (1.0 - p)
    alpha_t = 0.25 * targets + 0.75 * (1.0 - targets)
    focal_factor = (1.0 - pt) ** 2.0
    focal_factor_dot = -2.0 * (1.0 - pt) * (2.0 * targets - 1.0) * p * (1.0 - p)
    focal_derivative = alpha_t * (focal_factor_dot * bce + focal_factor * (p - targets))
    focal_bce = (weights[:, None] * focal_derivative * direction).sum() / weights.sum()
    one_minus_pt = targets * (1.0 - p) + (1.0 - targets) * p
    bce_second = p * (1.0 - p)
    pt_prime = (2.0 * targets - 1.0) * bce_second
    pt_second = (2.0 * targets - 1.0) * bce_second * (1.0 - 2.0 * p)
    focal = one_minus_pt ** 2.0
    focal_first = -2.0 * one_minus_pt * pt_prime
    focal_second = 2.0 * pt_prime ** 2.0 - 2.0 * one_minus_pt * pt_second
    focal_hessian = alpha_t * (focal_second * bce +
                               2.0 * focal_first * (p - targets) +
                               focal * bce_second)
    focal_hvp = (weights[:, None] * focal_hessian * direction).sum() / weights.sum()
    residual = logits - targets
    inverse_variance = np.exp(-log_variance)
    gaussian_hvp = (weights[:, None] * inverse_variance *
                    (direction - residual * variance_direction) / weights.sum())
    gaussian_variance_hvp = (weights[:, None] * inverse_variance *
                             (-residual * direction +
                              0.5 * residual * residual * variance_direction) /
                             weights.sum())
    poisson_hvp = (weights[:, None] * np.exp(logits) * direction / weights.sum())
    multilabel_bce_hvp = (weights[:, None] * probabilities * (1.0 - probabilities) *
                          direction / weights.sum())
    ordinal_logits = np.column_stack((0.4 * np.sin(0.05 * np.arange(1, N + 1)),
                                       0.9 + 0.3 * np.cos(0.04 * np.arange(1, N + 1))))
    ordinal_direction = np.column_stack((0.02 * np.cos(0.09 * np.arange(1, N + 1)),
                                         -0.01 * np.sin(0.08 * np.arange(1, N + 1))))
    ordinal_labels = (np.arange(N) % 3) + 1
    ordinal_q = 1.0 / (1.0 + np.exp(-ordinal_logits))
    ordinal_qp = ordinal_q * (1.0 - ordinal_q)
    ordinal_qpp = ordinal_qp * (1.0 - 2.0 * ordinal_q)
    ordinal_d = np.zeros_like(ordinal_logits)
    ordinal_d2 = np.zeros_like(ordinal_logits)
    ordinal_p = np.empty(N)
    first, middle, last = ordinal_labels == 1, ordinal_labels == 2, ordinal_labels == 3
    ordinal_p[first] = ordinal_q[first, 0]
    ordinal_d[first, 0] = ordinal_qp[first, 0]
    ordinal_d2[first, 0] = ordinal_qpp[first, 0]
    ordinal_p[middle] = ordinal_q[middle, 1] - ordinal_q[middle, 0]
    ordinal_d[middle, 0] = -ordinal_qp[middle, 0]
    ordinal_d[middle, 1] = ordinal_qp[middle, 1]
    ordinal_d2[middle, 0] = -ordinal_qpp[middle, 0]
    ordinal_d2[middle, 1] = ordinal_qpp[middle, 1]
    ordinal_p[last] = 1.0 - ordinal_q[last, 1]
    ordinal_d[last, 1] = -ordinal_qp[last, 1]
    ordinal_d2[last, 1] = -ordinal_qpp[last, 1]
    ordinal_p_dot = (ordinal_d * ordinal_direction).sum(axis=1)
    ordinal_hvp = -weights[:, None] / weights.sum() * (
        ordinal_d2 * ordinal_direction / ordinal_p[:, None] -
        ordinal_d * ordinal_p_dot[:, None] / ordinal_p[:, None] ** 2)
    return {
        "bce_hvp": float(bce_hvp.sum()),
        "multilabel_bce_hvp": float(multilabel_bce_hvp.sum()),
        "ordinal_cumulative_logit_hvp": float(ordinal_hvp.sum()),
        "softmax_cross_entropy_hvp": float(softmax_hvp.sum()),
        "softmax_hvp": float(softmax_hvp_vector.sum()),
        "log_softmax_hvp": float(log_softmax_hvp.sum()),
        "weighted_softmax_cross_entropy_hvp": float(weighted_softmax_hvp.sum()),
        "weighted_mse_hvp": float(weighted.sum()),
        "huber_hvp": float(huber.sum()),
        "mae_jvp": float(mae),
        "focal_bce_jvp": float(focal_bce),
        "focal_bce_hvp": float(focal_hvp),
        "gaussian_nll_hvp": float(gaussian_hvp.sum() + gaussian_variance_hvp.sum()),
        "poisson_nll_hvp": float(poisson_hvp.sum()),
    }


def row(details: dict[str, str], **updates: object) -> dict[str, str]:
    output = {field: "" for field in FIELDS}
    output.update(details)
    output.update({key: str(value) for key, value in updates.items()})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/neural_losses.csv"))
    parser.add_argument("--target", default="fortml_bench_neural_losses")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected = oracle()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    actual: dict[str, tuple[float, float]] = {}
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 3 and fields[0] in expected:
            actual[fields[0]] = (float(fields[1]), float(fields[2]))
        elif len(fields) == 3 and fields[0] == "mlp_weighted_objective":
            actual[fields[0]] = (float(fields[1]), float(fields[2]))
    required = set(expected) | {"mlp_weighted_objective"}
    if set(actual) != required:
        raise RuntimeError(f"missing neural-loss release rows: {sorted(actual)}")
    errors = {name: abs(actual[name][1] - expected[name]) for name in expected}
    if max(errors.values()) > EPS:
        raise RuntimeError(f"neural-loss checksum mismatch: {max(errors.values()):.3e}")
    if not math.isfinite(actual["mlp_weighted_objective"][1]):
        raise RuntimeError("MLP weighted objective checksum is not finite")

    details = {
        "workload": "neural_losses", "backend": "fortml", "device": "cpu",
        "status": "pass", "dimensions": "64x3; ordinal 64x2; weighted MLP 64x1",
        "repetitions": str(REPETITIONS),
        "oracle": "independent NumPy loss value/derivative formulas",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
        "notes": "weighted multilabel/ordinal plus shared value/JVP/VJP/HVP facade; MLP uses weighted-MSE products",
    }
    rows = []
    for phase, (seconds, checksum) in actual.items():
        rows.append(row(details, phase=phase,
                        seconds_per_operation=f"{seconds:.17e}",
                        metric="seconds_per_operation", value=f"{checksum:.17e}",
                        max_abs_error=(f"{errors[phase]:.17e}" if phase in errors else "")))
    rows.append(row(details, phase="device_capability", device="cuda",
                    status="unavailable", repetitions="", seconds_per_operation="",
                    metric="", value="", max_abs_error="",
                    oracle="typed_device_contract",
                    notes="loss and MLP objective CUDA kernels are not resident; no host fallback"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
