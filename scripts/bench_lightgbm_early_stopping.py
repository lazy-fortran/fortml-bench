#!/usr/bin/env python3
"""Correctness-gated LightGBM validation/early-stopping benchmark.

The release app reports best-round metadata for regression and binary logistic
leaf-wise boosters.  This script independently replays the one-feature
Newton updates, computes weighted validation loss and patience, and rejects
the release output if either policy or the typed CUDA/refusal contract drifts.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

import numpy as np


N_ESTIMATORS = 8
PATIENCE = 2
LEARNING_RATE = 1.0
L2 = 1.0


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = False
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty = True
            break
    return head + ("+dirty" if dirty else "")


def stable_logit(probability: float) -> float:
    clipped = min(max(probability, 1.0e-12), 1.0 - 1.0e-12)
    return float(np.log(clipped) - np.log1p(-clipped))


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    out = np.empty_like(values)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def objective_stages(binary: bool) -> tuple[np.ndarray, int, int]:
    x = np.arange(8, dtype=np.float64)
    target = np.array((0, 0, 0, 0, 1, 1, 1, 1), dtype=np.float64)
    if not binary:
        target *= 10.0
    validation = 1.0 - target if binary else 10.0 - target
    margin = np.full_like(target, stable_logit(float(np.mean(target))) if binary else np.mean(target))
    validation_margin = margin.copy()
    stages: list[float] = []
    for _ in range(N_ESTIMATORS):
        if binary:
            probability = stable_sigmoid(margin)
            gradient = probability - target
            hessian = np.maximum(probability*(1.0-probability), 1.0e-12)
        else:
            gradient = margin - target
            hessian = np.ones_like(target)
        total_g = float(np.sum(gradient))
        total_h = float(np.sum(hessian))
        best_gain = 0.0
        best_correction = np.full_like(target, -total_g/(total_h+L2))
        for split in range(1, x.size):
            left = x < float(split)-0.5
            right = ~left
            left_g, right_g = float(np.sum(gradient[left])), float(np.sum(gradient[right]))
            left_h, right_h = float(np.sum(hessian[left])), float(np.sum(hessian[right]))
            gain = 0.5*(left_g*left_g/(left_h+L2) + right_g*right_g/(right_h+L2) -
                        total_g*total_g/(total_h+L2))
            if gain > best_gain:
                best_gain = gain
                best_correction = np.where(
                    left, -left_g/(left_h+L2), -right_g/(right_h+L2),
                )
        margin = margin + LEARNING_RATE*best_correction
        validation_margin = validation_margin + LEARNING_RATE*best_correction
        if binary:
            probability = np.clip(stable_sigmoid(validation_margin), 1.0e-15, 1.0-1.0e-15)
            stage_loss = float(np.mean(-(validation*np.log(probability) +
                                         (1.0-validation)*np.log1p(-probability))))
        else:
            stage_loss = float(0.5*np.mean((validation_margin-validation)**2))
        stages.append(stage_loss)
    losses = np.asarray(stages)
    best_iteration = int(np.argmin(losses))+1
    best_loss = float(losses[best_iteration-1])
    stale = 0
    stop_iteration = N_ESTIMATORS
    running_best = np.inf
    for index, value in enumerate(losses, start=1):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/LIGHTGBM_EARLY_STOPPING.md"))
    parser.add_argument("--target", default="fortml_bench_lightgbm_early_stopping")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    rows: dict[tuple[str, str], tuple[int, int, bool, float]] = {}
    invalid = None
    cuda = None
    for line in run_release_app(fortml, args.target):
        fields = [field.strip() for field in line.split(",")]
        if fields[0] == "lgbm_early_invalid_validation":
            invalid = int(fields[1])
        elif fields[0] == "lgbm_early_cuda":
            cuda = int(fields[1])
        elif fields[0].startswith("lgbm_early_") and len(fields) == 6:
            rows[(fields[0].removeprefix("lgbm_early_"), fields[1])] = (
                int(fields[2]), int(fields[3]), fields[4].lower() in {"t", "true"}, float(fields[5]),
            )
        else:
            raise RuntimeError(f"malformed LightGBM release row: {line!r}")
    if invalid != 1:
        raise RuntimeError(f"expected FORTNUM_DOMAIN_ERROR=1, got {invalid}")
    if cuda != 3:
        raise RuntimeError(f"expected FORTNUM_NOT_IMPLEMENTED=3, got {cuda}")

    records: list[tuple[str, str, int, int, bool, float, int, int, float]] = []
    for objective, binary in (("regression", False), ("binary", True)):
        oracle_loss, best_iteration, stop_iteration = objective_stages(binary)
        expected_loss = float(oracle_loss[0])
        for policy, expected_count in (("restore_best", best_iteration), ("retain_all", stop_iteration)):
            observed = rows[("binary" if binary else "squared", policy)]
            if observed[0] != best_iteration or observed[1] != expected_count or not observed[2] or \
                    abs(observed[3]-expected_loss) > 2.0e-12:
                raise RuntimeError(
                    f"{objective}/{policy} mismatch: observed={observed}, "
                    f"oracle={(best_iteration, expected_count, expected_loss)}",
                )
            records.append((objective, policy, observed[0], observed[1], observed[2],
                            observed[3], best_iteration, expected_count, expected_loss))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LightGBM validation and early stopping",
        "",
        "This correctness-gated release lane independently replays the one-feature "
        "leaf-wise Newton recurrence for regression and binary logistic objectives. "
        "Patience is two rounds; both ensemble-retention policies and malformed "
        "validation/CUDA refusals are checked.",
        "",
        "| objective | policy | best iteration | retained estimators | early stopped | best validation loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for objective, policy, best, retained, stopped, observed_loss, _, _, _ in records:
        lines.append(f"| {objective} | {policy} | {best} | {retained} | {int(stopped)} | {observed_loss:.16e} |")
    lines += [
        "", "Independent NumPy losses and the release app agree for every row. "
        "The app also emits `lgbm_early_invalid_validation,1` and "
        "`lgbm_early_cuda,3`; these typed contracts are required.", "",
        "This is CPU correctness evidence, not resident CUDA performance evidence: "
        "LightGBM histogram prediction retains its explicit CUDA refusal.", "",
        "Reproduce with:", "", "```bash",
        "python -B scripts/bench_lightgbm_early_stopping.py \\",
        "  --fortml ../fortml --output results/LIGHTGBM_EARLY_STOPPING.md", "```", "",
        f"FortML revision: `{revision(fortml)}`",
        f"Benchmark revision: `{revision(root, (args.output.resolve(),))}`",
        f"Python {platform.python_version()}, NumPy {np.__version__}", "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
