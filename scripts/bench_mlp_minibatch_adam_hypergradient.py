#!/usr/bin/env python3
"""Correctness-gated fixed seeded mini-batch Adam hypergradient benchmark."""

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

N_TRAIN, N_VALIDATION, EPOCHS, BATCH_SIZE = 7, 3, 3, 3
N_PARAMETERS, LEARNING_RATE, L2 = 2, 0.08, 0.035
BETA1, BETA2, EPSILON = 0.84, 0.93, 0.025
SHUFFLE_SEED, FD_STEP, REPETITIONS = 43, 2.0e-6, 16
PARAMETERS = np.log([LEARNING_RATE, L2])
DIRECTION = np.array([0.27, -0.19])
ORACLE_TOLERANCE = 3.0e-10
FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status",
    "n_train", "n_validation", "n_parameters", "epochs", "batch_size",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.array([[-2.2], [-1.1], [-0.35], [0.2], [0.9], [1.6], [2.3]], dtype=np.float64)
    train_target = 0.65 * train_x - 0.11
    validation_x = np.array([[-1.8], [0.35], [1.9]], dtype=np.float64)
    validation_target = 0.65 * validation_x - 0.11
    return train_x, train_target, validation_x, validation_target


def shuffle_order(order: np.ndarray, state: int) -> tuple[np.ndarray, int]:
    order = order.copy()
    for i in range(order.size, 1, -1):
        state = (48271 * state) % 2147483647
        j = state % i
        order[i - 1], order[j] = order[j], order[i - 1]
    return order, state


def loss_gradient(theta: np.ndarray, x: np.ndarray, target: np.ndarray, l2: float) -> np.ndarray:
    residual = x[:, 0] * theta[0] + theta[1] - target[:, 0]
    return np.array([np.mean(residual * x[:, 0]) + l2 * theta[0],
                     np.mean(residual) + l2 * theta[1]], dtype=np.float64)


def trajectory(parameters: np.ndarray) -> float:
    train_x, train_target, validation_x, validation_target = fixture()
    learning_rate, l2 = np.exp(parameters)
    theta = np.array([0.21, -0.06], dtype=np.float64)
    first = np.zeros(2, dtype=np.float64)
    second = np.zeros(2, dtype=np.float64)
    state, step = SHUFFLE_SEED, 0
    for _ in range(EPOCHS):
        order, state = shuffle_order(np.arange(N_TRAIN, dtype=np.int64), state)
        for start in range(0, N_TRAIN, BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            gradient = loss_gradient(theta, train_x[indices], train_target[indices], l2)
            step += 1
            first = BETA1 * first + (1.0 - BETA1) * gradient
            second = BETA2 * second + (1.0 - BETA2) * gradient * gradient
            first_hat = first / (1.0 - BETA1**step)
            second_hat = second / (1.0 - BETA2**step)
            theta -= learning_rate * first_hat / (np.sqrt(np.maximum(second_hat, 0.0)) + EPSILON)
    residual = validation_x[:, 0] * theta[0] + theta[1] - validation_target[:, 0]
    return 0.5 * float(np.mean(residual * residual))


def oracle() -> dict[str, Any]:
    value = trajectory(PARAMETERS)
    gradient = np.empty(N_PARAMETERS)
    started = time.perf_counter()
    for index in range(N_PARAMETERS):
        plus, minus = PARAMETERS.copy(), PARAMETERS.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        gradient[index] = (trajectory(plus) - trajectory(minus)) / (2.0 * FD_STEP)
    tangent = (trajectory(PARAMETERS + FD_STEP * DIRECTION) -
               trajectory(PARAMETERS - FD_STEP * DIRECTION)) / (2.0 * FD_STEP)
    return {"value": value, "gradient": gradient, "tangent": tangent,
            "seconds": (time.perf_counter() - started) / REPETITIONS}


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update({"device": "cpu", "n_train": N_TRAIN, "n_validation": N_VALIDATION,
                   "n_parameters": N_PARAMETERS, "epochs": EPOCHS,
                   "batch_size": BATCH_SIZE, "repetitions": REPETITIONS})
    result.update(values)
    return result


def unavailable(details: dict[str, str], device: str, status: str, notes: str) -> list[dict[str, Any]]:
    return [base(details, workload="mlp_minibatch_adam_hypergradient", phase=phase,
                 variant="fixed_seeded_shuffle", backend="fortml", device=device,
                 status=status, metric=metric, oracle="FortML release-app protocol",
                 notes=notes)
            for phase, metric in (("value_gradient", "validation_mse"),
                                  ("gradient_component", "gradient_1"),
                                  ("gradient_component", "gradient_2"),
                                  ("jvp", "directional_validation_mse_derivative"))]


def read_oracle(path: Path) -> dict[tuple[str, int], float]:
    with path.open(newline="") as stream:
        return {(row["quantity"], int(row["index"])): float(row["value"])
                for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_minibatch_adam_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_minibatch_adam_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root, fortml, output = (Path(__file__).resolve().parents[1], args.fortml.resolve(),
                             args.output.resolve())
    details = {"python_version": platform.python_version(), "numpy_version": np.__version__,
               "fortml_revision": revision(fortml),
               "benchmark_revision": revision(root, (output, root / "scripts" / "__pycache__")),
               "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3"}
    expected = oracle()
    rows = [base(details, workload="mlp_minibatch_adam_hypergradient", phase="value_gradient",
                 variant="fixed_seeded_shuffle", backend="numpy_oracle", status="pass",
                 seconds_per_operation=expected["seconds"], metric="validation_mse",
                 value=expected["value"], max_abs_error=0.0,
                 oracle="independent NumPy coupled-L2 Adam recurrence with central-FD products",
                 notes="packed=[log_lr,log_l2]; epochs=3; batch=3; LCG shuffle seed=43")]
    for index, value in enumerate(expected["gradient"], 1):
        rows.append(base(details, workload="mlp_minibatch_adam_hypergradient", phase="gradient_component",
                         variant="fixed_seeded_shuffle", backend="numpy_oracle", status="pass",
                         metric=f"gradient_{index}", value=float(value), max_abs_error=0.0,
                         oracle="independent central finite-difference outer objective",
                         notes="all packed components checked"))
    rows.append(base(details, workload="mlp_minibatch_adam_hypergradient", phase="jvp",
                     variant="fixed_seeded_shuffle", backend="numpy_oracle", status="pass",
                     metric="directional_validation_mse_derivative", value=expected["tangent"],
                     max_abs_error=0.0, oracle="independent central finite-difference directional product",
                     notes=f"direction={DIRECTION.tolist()}; h={FD_STEP:g}"))
    if args.skip_fortml:
        rows.extend(unavailable(details, "cpu", "skipped", "--skip-fortml"))
    else:
        source = fortml / "app" / f"{args.target}.f90"
        if not source.is_file():
            rows.extend(unavailable(details, "cpu", "unavailable", f"release source absent: {source.name}"))
        else:
            env = os.environ.copy(); env.update({"FO_FC": env.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
            build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=env,
                                   capture_output=True, text=True)
            if build.returncode:
                rows.extend(unavailable(details, "cpu", "unavailable", "fo build failed"))
            else:
                with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-minibatch-adam-") as temporary:
                    oracle_path = Path(temporary) / "oracle.csv"
                    check_env = dict(env); check_env.update({
                        "FORTML_BENCH_MINIBATCH_ADAM_ORACLE": str(oracle_path),
                        "FORTML_BENCH_ORACLE_ONLY": "1"})
                    check = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                                           env=check_env, capture_output=True, text=True)
                    if check.returncode or not oracle_path.is_file():
                        rows.extend(unavailable(details, "cpu", "unavailable", "release target emitted no complete oracle"))
                    else:
                        actual = read_oracle(oracle_path)
                        required = {("value", 1), ("jvp", 1)} | {("gradient", i) for i in range(1, N_PARAMETERS + 1)}
                        if set(actual) != required:
                            raise RuntimeError("mini-batch Adam release app omitted a complete oracle")
                        errors = [abs(actual[("value", 1)] - expected["value"]),
                                  abs(actual[("jvp", 1)] - expected["tangent"])]
                        errors.extend(abs(actual[("gradient", i)] - expected["gradient"][i - 1])
                                      for i in range(1, N_PARAMETERS + 1))
                        error = max(errors)
                        if error > ORACLE_TOLERANCE:
                            raise RuntimeError(f"mini-batch Adam oracle mismatch: {error:.3e}")
                        timed = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                                               env=env, capture_output=True, text=True)
                        if timed.returncode:
                            rows.extend(unavailable(details, "cpu", "unavailable", "timing execution failed"))
                        else:
                            prefix = "mlp_minibatch_adam_hypergradient_value_gradient,"
                            timing = next(float(line.split(",", 1)[1]) for line in timed.stdout.splitlines()
                                          if line.startswith(prefix))
                            rows.append(base(details, workload="mlp_minibatch_adam_hypergradient",
                                             phase="value_gradient", variant="fixed_seeded_shuffle",
                                             backend="fortml", status="pass", seconds_per_operation=timing,
                                             metric="validation_mse", value=actual[("value", 1)],
                                             max_abs_error=error,
                                             oracle="independent NumPy Adam trajectory and central-FD products",
                                             notes="three products checked"))
                            rows.extend(base(details, workload="mlp_minibatch_adam_hypergradient",
                                             phase="gradient_component", variant="fixed_seeded_shuffle",
                                             backend="fortml", status="pass", metric=f"gradient_{i}",
                                             value=actual[("gradient", i)],
                                             max_abs_error=abs(actual[("gradient", i)] - expected["gradient"][i - 1]),
                                             oracle="independent NumPy Adam trajectory and central-FD products",
                                             notes=args.target) for i in range(1, N_PARAMETERS + 1))
                            rows.append(base(details, workload="mlp_minibatch_adam_hypergradient", phase="jvp",
                                             variant="fixed_seeded_shuffle", backend="fortml", status="pass",
                                             metric="directional_validation_mse_derivative",
                                             value=actual[("jvp", 1)],
                                             max_abs_error=abs(actual[("jvp", 1)] - expected["tangent"]),
                                             oracle="independent NumPy Adam trajectory and central-FD products",
                                             notes=args.target))
    rows.extend(unavailable(details, "cuda", "unavailable",
                            "complete mini-batch Adam hypergradient is CPU-only until resident state derivatives exist"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
