#!/usr/bin/env python3
"""Correctness-gated XGBoost validation/early-stopping benchmark.

The release app reports the selected validation round and the number of
retained trees for both ``restore_best`` policies.  This script independently
replays the depth-one Newton updates in NumPy, scores every validation stage,
and derives the patience stop before accepting any release-app record.
No existing CSV is changed; the lane writes a small Markdown protocol report.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

import numpy as np


OBJECTIVES = ("squared", "logistic", "squared_log")
N_ESTIMATORS = 8
PATIENCE = 2
L2 = 1.0
LEARNING_RATE = 1.0


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return a reproducible commit id, marking unrelated edits as dirty."""
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = False
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        if name not in ignored_names:
            dirty = True
            break
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(8, dtype=np.float64).reshape(-1, 1)
    squared = np.array((0, 0, 0, 0, 10, 10, 10, 10), dtype=np.float64)
    logistic = np.array((0, 0, 0, 0, 1, 1, 1, 1), dtype=np.float64)
    return x, squared, logistic


def stable_logit(probability: float) -> float:
    clipped = min(max(probability, 1.0e-12), 1.0 - 1.0e-12)
    return float(np.log(clipped) - np.log1p(-clipped))


def derivatives(objective: str, margin: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if objective == "squared":
        return margin - target, np.ones_like(margin)
    if objective == "logistic":
        probability = np.where(
            margin >= 0.0,
            1.0 / (1.0 + np.exp(-margin)),
            np.exp(margin) / (1.0 + np.exp(margin)),
        )
        return probability - target, np.maximum(probability * (1.0 - probability), 1.0e-12)
    if objective == "squared_log":
        residual = margin - np.log1p(target)
        scale = np.exp(np.clip(margin, np.log(np.finfo(float).tiny), np.log(np.finfo(float).max) - 1.0))
        return residual / scale, np.maximum((1.0 - residual) / scale, 1.0e-12)
    raise ValueError(f"unsupported objective: {objective}")


def loss(objective: str, margin: np.ndarray, target: np.ndarray) -> float:
    if objective == "squared":
        values = 0.5 * (margin - target) ** 2
    elif objective == "logistic":
        values = np.where(
            margin >= 0.0,
            (1.0 - target) * margin + np.log1p(np.exp(-margin)),
            -target * margin + np.log1p(np.exp(margin)),
        )
    elif objective == "squared_log":
        values = 0.5 * (margin - np.log1p(target)) ** 2
    else:
        raise ValueError(f"unsupported objective: {objective}")
    value = float(np.mean(values))
    if not np.isfinite(value):
        raise RuntimeError(f"nonfinite {objective} validation loss")
    return value


def independent_stages(objective: str) -> tuple[np.ndarray, int, int]:
    """Replay exhaustive depth-one Newton trees and patience independently."""
    x, squared, logistic = fixture()
    target = squared if objective != "logistic" else logistic
    validation = (10.0 - squared) if objective != "logistic" else (1.0 - logistic)
    if objective == "squared_log":
        target = squared
        validation = 10.0 - squared
        base = float(np.mean(np.log1p(target)))
    elif objective == "logistic":
        base = stable_logit(float(np.mean(target)))
    else:
        base = float(np.mean(target))
    train_margin = np.full(target.shape, base)
    validation_margin = np.full(validation.shape, base)
    stages: list[float] = []
    for _ in range(N_ESTIMATORS):
        gradient, hessian = derivatives(objective, train_margin, target)
        total_gradient = float(np.sum(gradient))
        total_hessian = float(np.sum(hessian))
        best_gain = 0.0
        best_left: np.ndarray | None = None
        best_left_weight = 0.0
        best_right_weight = 0.0
        # The app's exact path enumerates finite feature thresholds in order.
        for threshold in 0.5 * (x[:-1, 0] + x[1:, 0]):
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
            correction = np.full(
                target.shape, -total_gradient / (total_hessian + L2),
            )
        else:
            correction = np.where(best_left, best_left_weight, best_right_weight)
        train_margin = train_margin + LEARNING_RATE * correction
        validation_margin = validation_margin + LEARNING_RATE * correction
        if objective == "squared_log":
            stage_loss = loss(objective, validation_margin, validation)
        else:
            stage_loss = loss(objective, validation_margin, validation)
        stages.append(stage_loss)

    values = np.asarray(stages, dtype=np.float64)
    best_iteration = int(np.argmin(values)) + 1
    best_loss = float(values[best_iteration - 1])
    stop_iteration = N_ESTIMATORS
    stale = 0
    running_best = np.inf
    for index, value in enumerate(values, start=1):
        if value < running_best:
            running_best = value
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            stop_iteration = index
            break
    return np.array((best_loss,), dtype=np.float64), best_iteration, stop_iteration


def run_release_app(fortml: Path, target: str) -> list[str]:
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def parse(lines: list[str]) -> tuple[dict[tuple[str, str], tuple[int, int, int, float]], int]:
    rows: dict[tuple[str, str], tuple[int, int, int, float]] = {}
    invalid: int | None = None
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if fields[0] == "xgb_early_invalid_validation":
            if len(fields) != 2:
                raise RuntimeError(f"malformed refusal row: {line!r}")
            invalid = int(fields[1])
            continue
        if not fields[0].startswith("xgb_early_") or len(fields) != 6:
            raise RuntimeError(f"malformed early-stopping row: {line!r}")
        objective = fields[0].removeprefix("xgb_early_")
        rows[(objective, fields[1])] = (
            int(fields[2]), int(fields[3]), int(fields[4]), float(fields[5]),
        )
    if invalid is None:
        raise RuntimeError("release app omitted malformed-validation refusal")
    return rows, invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/XGBOOST_EARLY_STOPPING.md"),
    )
    parser.add_argument("--target", default="fortml_bench_xgboost_early_stopping")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    observed, invalid = parse(run_release_app(fortml, args.target))
    if invalid != 1:
        raise RuntimeError(f"expected FORTNUM_DOMAIN_ERROR=1, got {invalid}")

    records: list[tuple[str, str, tuple[int, int, int, float], int, int, float]] = []
    for objective in OBJECTIVES:
        oracle_loss, best_iteration, stop_iteration = independent_stages(objective)
        expected_loss = float(oracle_loss[0])
        for policy, expected_count in (
            ("restore_best", best_iteration), ("retain_all", stop_iteration),
        ):
            key = (objective, policy)
            if key not in observed:
                raise RuntimeError(f"release app omitted {key}")
            row = observed[key]
            expected = (best_iteration, expected_count, 1, expected_loss)
            if row[:3] != expected[:3] or not np.isclose(
                row[3], expected_loss, rtol=0.0, atol=3.0e-12,
            ):
                raise RuntimeError(
                    f"{key} mismatch: app={row}, independent={expected}"
                )
            records.append((objective, policy, row, best_iteration, expected_count, expected_loss))

    source_revision = revision(
        fortml, (fortml / "verification" / "fortml-gfortran.txt",)
    )
    benchmark_revision = revision(root, (args.output.resolve(),))
    lines = [
        "# XGBoost validation and early stopping",
        "",
        "This release-app lane independently replays exact depth-one Newton",
        "updates for squared, binary logistic, and squared-log objectives.",
        "Validation loss is evaluated after every stage; patience is two",
        "consecutive non-improving rounds. Both `restore_best` policies are",
        "checked, and malformed validation data must return",
        "`FORTNUM_DOMAIN_ERROR` (code 1).",
        "",
        "| objective | policy | best iteration | retained estimators | early stopped | best validation loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for objective, policy, row, *_ in records:
        lines.append(
            f"| {objective} | {policy} | {row[0]} | {row[1]} | {row[2]} | {row[3]:.16e} |"
        )
    lines.extend([
        "",
        "The independent oracle and release app agree for every row. The",
        "release app also emits `xgb_early_invalid_validation,1`; the benchmark",
        "fails if that typed refusal changes. This is CPU correctness evidence,",
        "not resident CUDA performance evidence: XGBoost tree prediction still",
        "uses FortML's explicit CUDA refusal contract.",
        "",
        "Reproduce:",
        "",
        "```bash",
        "python -B scripts/bench_xgboost_early_stopping.py \\",
        "  --fortml ../fortml --output results/XGBOOST_EARLY_STOPPING.md",
        "```",
        "",
        f"FortML revision: `{source_revision}`",
        f"Benchmark revision: `{benchmark_revision}`",
        f"Python {platform.python_version()}, NumPy {np.__version__}",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
