#!/usr/bin/env python3
"""Correctness-gated grouped coupled-L2 Adam trajectory benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

STEPS = 4
LR = 0.08
L2 = 0.03
BETA1 = 0.85
BETA2 = 0.97
EPSILON = 1.0e-7
SCALES = np.array([0.7, 1.3], dtype=np.float64)
FD_STEP = 2.0e-6
REPETITIONS = 32
TOLERANCE = 3.0e-8
PARAMETERS = np.array([np.log(LR), np.log(L2),
                       np.log(BETA1/(1.0-BETA1)),
                       np.log(BETA2/(1.0-BETA2)),
                       np.log(SCALES[0]), np.log(SCALES[1])], dtype=np.float64)
DIRECTION = np.array([0.17, -0.13, 0.07, -0.05, 0.11, -0.09], dtype=np.float64)
FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status",
    "n_train", "n_validation", "n_parameters", "steps", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                   text=True).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def details(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([-1.0, -0.2, 0.7, 1.5], dtype=np.float64)
    y = 0.6*x - 0.15
    vx = np.array([-0.8, 0.4, 1.2], dtype=np.float64)
    vy = 0.6*vx - 0.15
    return x, y, vx, vy


def trajectory(parameters: np.ndarray) -> float:
    lr, l2 = np.exp(parameters[:2])
    beta1, beta2 = 1.0/(1.0+np.exp(-parameters[2:4]))
    weight_scale, bias_scale = np.exp(parameters[4:6])
    x, y, vx, vy = fixture()
    theta = np.array([0.25, 0.1], dtype=np.float64)
    first = np.zeros(2, dtype=np.float64)
    second = np.zeros(2, dtype=np.float64)
    scale = np.array([weight_scale, bias_scale], dtype=np.float64)
    for step in range(1, STEPS + 1):
        residual = theta[0]*x + theta[1] - y
        gradient = np.array([np.mean(residual*x), np.mean(residual)]) + l2*theta
        first = beta1*first + (1.0-beta1)*gradient
        second = beta2*second + (1.0-beta2)*gradient*gradient
        direction = (first/(1.0-beta1**step)) / (
            np.sqrt(second/(1.0-beta2**step)) + EPSILON)
        theta -= lr*scale*direction
    residual = theta[0]*vx + theta[1] - vy
    return 0.5*float(np.mean(residual*residual))


def oracle() -> dict[str, Any]:
    value = trajectory(PARAMETERS)
    gradient = np.empty(6, dtype=np.float64)
    started = time.perf_counter()
    for i in range(6):
        plus, minus = PARAMETERS.copy(), PARAMETERS.copy()
        plus[i] += FD_STEP
        minus[i] -= FD_STEP
        gradient[i] = (trajectory(plus)-trajectory(minus))/(2.0*FD_STEP)
    tangent = (trajectory(PARAMETERS+FD_STEP*DIRECTION)-
               trajectory(PARAMETERS-FD_STEP*DIRECTION))/(2.0*FD_STEP)
    seconds = (time.perf_counter()-started)/REPETITIONS
    if not np.all(np.isfinite(gradient)):
        raise RuntimeError("grouped Adam NumPy oracle is nonfinite")
    return {"value": value, "gradient": gradient, "tangent": tangent,
            "seconds": seconds}


def base(meta: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(meta)
    row.update({"variant": "fixed_full_batch_grouped_coupled_l2_adam", "device": "cpu",
                "n_train": 4, "n_validation": 3, "n_parameters": 6,
                "steps": STEPS, "repetitions": REPETITIONS,
                "oracle": "independent NumPy grouped coupled-L2 Adam recurrence"})
    row.update(values)
    return row


def oracle_rows(meta: dict[str, str], expected: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [base(meta, workload="mlp_adam_optimizer_group_hypergradient",
                 phase="value_gradient", backend="numpy_oracle", status="pass",
                 seconds_per_operation=expected["seconds"], metric="validation_mse",
                 value=expected["value"], max_abs_error=0.0,
                 notes="packed=[log_lr,log_l2,logit_beta1,logit_beta2,log_multiplier_weight,log_multiplier_bias]"),
            base(meta, workload="mlp_adam_optimizer_group_hypergradient", phase="jvp",
                 backend="numpy_oracle", status="pass", seconds_per_operation=expected["seconds"],
                 metric="directional_validation_mse_derivative", value=expected["tangent"],
                 max_abs_error=0.0, notes=f"direction={DIRECTION.tolist()}; h={FD_STEP:g}")]
    names = ("log_learning_rate", "log_l2", "logit_beta1", "logit_beta2",
             "log_multiplier_weight", "log_multiplier_bias")
    rows.extend(base(meta, workload="mlp_adam_optimizer_group_hypergradient",
                     phase="gradient_component", backend="numpy_oracle", status="pass",
                     repetitions=1, metric=f"gradient_{name}", value=float(value),
                     max_abs_error=0.0, notes="independent central finite difference")
               for name, value in zip(names, expected["gradient"]))
    return rows


def unavailable(meta: dict[str, str], device: str, note: str) -> list[dict[str, Any]]:
    rows = [base(meta, workload="mlp_adam_optimizer_group_hypergradient",
                 phase="value_gradient", backend="fortml", device=device,
                 status="unavailable", metric="validation_mse",
                 oracle="FortML release-app protocol", notes=note),
            base(meta, workload="mlp_adam_optimizer_group_hypergradient", phase="jvp",
                 backend="fortml", device=device, status="unavailable",
                 metric="directional_validation_mse_derivative",
                 oracle="FortML release-app protocol", notes=note)]
    rows.extend(base(meta, workload="mlp_adam_optimizer_group_hypergradient",
                     phase="gradient_component", backend="fortml", device=device,
                     status="unavailable", metric=f"gradient_{name}",
                     oracle="FortML release-app protocol", notes=note)
               for name in ("log_learning_rate", "log_l2", "logit_beta1", "logit_beta2",
                            "log_multiplier_weight", "log_multiplier_bias"))
    return rows


def read_oracle(path: Path) -> dict[tuple[str, int], float]:
    with path.open(newline="") as stream:
        return {(row["quantity"], int(row["index"])): float(row["value"])
                for row in csv.DictReader(stream)}


def fortml_rows(fortml: Path, meta: dict[str, str], expected: dict[str, Any],
                target: str, no_build: bool) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    if not no_build:
        built = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                               env=environment, capture_output=True, text=True)
        if built.returncode:
            return unavailable(meta, "cpu", "fo build failed")
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-adam-group-") as work:
        oracle_path = Path(work)/"oracle.csv"
        env = dict(environment)
        env.update({"FORTML_BENCH_ADAM_GROUP_HYPERGRADIENT_ORACLE": str(oracle_path),
                    "FORTML_BENCH_ORACLE_ONLY": "1"})
        checked = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                                 env=env, capture_output=True, text=True)
        if checked.returncode or not oracle_path.is_file():
            return unavailable(meta, "cpu", "release app emitted no complete oracle")
        actual = read_oracle(oracle_path)
        required = {("value", 1), ("jvp", 1)} | {("gradient", i) for i in range(1, 7)}
        if set(actual) != required:
            raise RuntimeError("grouped Adam app omitted a complete product array")
        errors = [abs(actual[("value", 1)]-expected["value"]),
                  abs(actual[("jvp", 1)]-expected["tangent"])]
        errors.extend(abs(actual[("gradient", i)]-expected["gradient"][i-1])
                      for i in range(1, 7))
        error = float(max(errors))
        if error > TOLERANCE:
            raise RuntimeError(f"grouped Adam oracle mismatch: {error:.3e}")
        timed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=environment, capture_output=True, text=True)
    if timed.returncode:
        return unavailable(meta, "cpu", "release timing failed")
    marker = "mlp_adam_optimizer_group_hypergradient_value_gradient,"
    timing = next((float(line.split(",", 1)[1]) for line in timed.stdout.splitlines()
                   if line.startswith(marker)), None)
    if timing is None:
        raise RuntimeError("grouped Adam app emitted no timing marker")
    rows = [base(meta, workload="mlp_adam_optimizer_group_hypergradient",
                 phase="value_gradient", backend="fortml", status="pass",
                 seconds_per_operation=timing, metric="validation_mse",
                 value=actual[("value", 1)], max_abs_error=error,
                 oracle="complete NumPy grouped Adam value/gradient/JVP array", notes=target),
            base(meta, workload="mlp_adam_optimizer_group_hypergradient", phase="jvp",
                 backend="fortml", status="pass", metric="directional_validation_mse_derivative",
                 value=actual[("jvp", 1)], max_abs_error=error,
                 oracle="complete NumPy grouped Adam value/gradient/JVP array", notes=target)]
    names = ("log_learning_rate", "log_l2", "logit_beta1", "logit_beta2",
             "log_multiplier_weight", "log_multiplier_bias")
    rows.extend(base(meta, workload="mlp_adam_optimizer_group_hypergradient",
                     phase="gradient_component", backend="fortml", status="pass",
                     repetitions=1, metric=f"gradient_{name}", value=actual[("gradient", i)],
                     max_abs_error=error,
                     oracle="complete NumPy grouped Adam value/gradient/JVP array", notes=target)
                for i, name in enumerate(names, start=1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_adam_optimizer_group_hypergradient.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/MLP_ADAM_OPTIMIZER_GROUP_HYPERGRADIENT.md"))
    parser.add_argument("--target", default="fortml_bench_mlp_adam_optimizer_group_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    meta = details(root, fortml, output)
    expected = oracle()
    rows = oracle_rows(meta, expected)
    if args.skip_fortml:
        rows.extend(unavailable(meta, "cpu", "--skip-fortml requested"))
    else:
        rows.extend(fortml_rows(fortml, meta, expected, args.target, args.no_build))
    rows.extend(unavailable(meta, "cuda", "typed CUDA refusal: resident grouped Adam trajectory unavailable"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    max_error = max(float(row["max_abs_error"]) for row in rows if row["max_abs_error"] != "")
    report.write_text(f"""# Grouped coupled-L2 Adam trajectory benchmark

This lane compares FortML's fixed full-batch grouped Adam trajectory with an
independent NumPy two-parameter recurrence over six packed hyperparameters.
Adam moment state is updated
before each group's post-update multiplier, matching `mlp_train`.

Run it with:

```bash
python -B scripts/bench_mlp_adam_optimizer_group_hypergradient.py \\
  --fortml ../fortml --output results/mlp_adam_optimizer_group_hypergradient.csv \\
  --report results/MLP_ADAM_OPTIMIZER_GROUP_HYPERGRADIENT.md
```

The release fixture records {len(rows)} rows.  The maximum FortML-versus-NumPy
discrepancy was `{max_error:.3e}` (gate `{TOLERANCE:.1e}`).  CUDA is recorded as
an explicit typed-unavailable boundary because the complete resident Adam
trajectory and derivative state are not implemented.

Source revision: `{meta['fortml_revision']}`  
Benchmark revision: `{meta['benchmark_revision']}`
""")
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
