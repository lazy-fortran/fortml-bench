#!/usr/bin/env python3
"""Correctness-gated scheduled MLP hypergradient benchmark.

The NumPy implementation below is an independent tanh MLP and trajectory
oracle.  FortML rows are retained only after the complete value, reverse
gradient, and directional JVP arrays agree.  CUDA is recorded as a capability
refusal until a resident trajectory kernel exists.
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


STEPS = 8
BASE_RATE = 1.0e-2
L2 = 1.0e-4
MIN_FRACTION = 0.1
DECAY_FACTOR = 0.9
LAYERS = (3, 8, 1)
TRAIN_COUNT = 72
N_SAMPLES = 96
REPETITIONS = 16
FD_STEP = 2.0e-5
TOLERANCE = 5.0e-8
DIRECTION = np.array([0.7, -0.3, 0.2, -0.1], dtype=np.float64)
FIELDS = (
    "workload", "phase", "backend", "device", "status", "steps",
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
    x = np.empty((N_SAMPLES, 3), dtype=np.float64)
    y = np.empty((N_SAMPLES, 1), dtype=np.float64)
    for j in range(3):
        for i in range(N_SAMPLES):
            x[i, j] = math.sin(0.017 * (i + 1) + 0.13 * (j + 1)) + \
                0.15 * math.cos(0.009 * (i + 1) * (j + 1))
    y[:, 0] = (0.4 * np.sin(x[:, 0]) + 0.2 * x[:, 1] - 0.1 * x[:, 2] +
               0.03 * np.cos(2.0 * x[:, 0]))
    return x[:TRAIN_COUNT], y[:TRAIN_COUNT], x[TRAIN_COUNT:], y[TRAIN_COUNT:]


def initial_parameters(seed: int = 23) -> np.ndarray:
    parts: list[np.ndarray] = []
    for layer, (n_in, n_out) in enumerate(zip(LAYERS[:-1], LAYERS[1:]), 1):
        scale = math.sqrt(6.0 / (n_in + n_out))
        values = []
        for j in range(1, n_out + 1):
            for i in range(1, n_in + 1):
                index = i + n_in * (j - 1)
                values.append(scale * math.sin(seed + 1009 * layer + 9176 * index))
        parts.append(np.asarray(values, dtype=np.float64))
        parts.append(np.asarray([
            0.01 * scale * math.sin(seed + 1009 * layer + 7919 * j)
            for j in range(1, n_out + 1)
        ], dtype=np.float64))
    return np.concatenate(parts)


def unpack(theta: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    position = 0
    for n_in, n_out in zip(LAYERS[:-1], LAYERS[1:]):
        n_weight = n_in * n_out
        weights.append(theta[position:position + n_weight].reshape((n_in, n_out), order="F"))
        position += n_weight
        biases.append(theta[position:position + n_out])
        position += n_out
    return weights, biases


def loss_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    weights, biases = unpack(theta)
    activations = [x]
    preactivations: list[np.ndarray] = []
    current = x
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        z = current @ weight + bias
        preactivations.append(z)
        current = np.tanh(z) if layer < len(weights) - 1 else z
        activations.append(current)
    residual = current - y
    value = 0.5 * np.mean(residual * residual) + 0.5 * l2 * np.dot(theta, theta)
    dz = residual / x.shape[0]
    gradients: list[np.ndarray] = [np.empty_like(w) for w in weights]
    bias_gradients: list[np.ndarray] = [np.empty_like(b) for b in biases]
    for layer in range(len(weights) - 1, -1, -1):
        gradients[layer] = activations[layer].T @ dz
        bias_gradients[layer] = dz.sum(axis=0)
        if layer:
            da = dz @ weights[layer].T
            dz = da * (1.0 - np.tanh(preactivations[layer - 1]) ** 2)
    packed: list[np.ndarray] = []
    for weight, bias in zip(gradients, bias_gradients):
        packed.extend((weight.reshape(-1, order="F"), bias))
    return float(value), np.concatenate(packed) + l2 * theta


def schedule_rate(update: int, base_rate: float, minimum: float) -> float:
    if update <= 2:
        return base_rate * update / 2.0
    progress = min(1.0, (update - 2) / 10.0)
    factor = minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_rate * factor


def objective(parameters: np.ndarray) -> float:
    base_rate = math.exp(parameters[0])
    l2 = math.exp(parameters[1])
    minimum = 1.0 / (1.0 + math.exp(-parameters[2]))
    train_x, train_y, validation_x, validation_y = fixture()
    theta = initial_parameters()
    for update in range(1, STEPS + 1):
        _, gradient = loss_gradient(theta, train_x, train_y, l2)
        theta = theta - schedule_rate(update, base_rate, minimum) * gradient
    value, _ = loss_gradient(theta, validation_x, validation_y, 0.0)
    return value


def oracle() -> tuple[float, np.ndarray, float]:
    parameters = np.array([math.log(BASE_RATE), math.log(L2),
                           math.log(MIN_FRACTION / (1.0 - MIN_FRACTION)),
                           math.log(DECAY_FACTOR / (1.0 - DECAY_FACTOR))])
    value = objective(parameters)
    gradient = np.empty(4, dtype=np.float64)
    for i in range(4):
        plus = parameters.copy(); plus[i] += FD_STEP
        minus = parameters.copy(); minus[i] -= FD_STEP
        gradient[i] = (objective(plus) - objective(minus)) / (2.0 * FD_STEP)
    tangent = float(np.dot(gradient, DIRECTION))
    return value, gradient, tangent


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def make_row(details: dict[str, str], phase: str, backend: str, device: str,
             status: str, metric: str, value: object, error: object,
             seconds: object, notes: str) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "mlp_schedule_hypergradient", "phase": phase,
        "backend": backend, "device": device, "status": status,
        "steps": STEPS, "repetitions": REPETITIONS,
        "seconds_per_operation": seconds, "metric": metric, "value": value,
        "max_abs_error": error,
        "oracle": "independent NumPy tanh trajectory with central-FD outer oracle",
        "notes": notes,
    })
    return result


def run_fortml(fortml: Path, target: str, oracle_path: Path) -> tuple[bool, str, dict[str, float]]:
    env = os.environ.copy()
    env["FORTML_BENCH_MLP_SCHEDULE_HYPERGRADIENT_ORACLE"] = str(oracle_path)
    env["FORTML_BENCH_ORACLE_ONLY"] = "1"
    try:
        completed = subprocess.run(["fo", "exec", target], cwd=fortml, env=env,
                                   text=True, capture_output=True, check=False)
    except OSError as exc:
        return False, str(exc), {}
    if completed.returncode != 0:
        return False, completed.stdout + completed.stderr, {}
    values: dict[str, float] = {}
    if not oracle_path.exists():
        return False, "release app did not write its complete-array oracle", values
    with oracle_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values[f"{row['quantity']}:{row['index']}"] = float(row["value"])
    return True, "complete-array oracle accepted", values


def timing_seconds(fortml: Path, target: str) -> float:
    completed = subprocess.run(["fo", "exec", target], cwd=fortml,
                               env=os.environ.copy(), text=True,
                               capture_output=True, check=True)
    for line in completed.stdout.splitlines():
        if line.startswith("mlp_schedule_hypergradient_value_gradient,"):
            return float(line.split(",", 1)[1].strip())
    raise RuntimeError("release app did not emit scheduled hypergradient timing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_schedule_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_schedule_hypergradient")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    details = metadata(root, fortml, output)
    expected_value, expected_gradient, expected_tangent = oracle()
    bench_tmp = root / ".bench_tmp"
    bench_tmp.mkdir(exist_ok=True)
    oracle_path = bench_tmp / "mlp_schedule_hypergradient.csv"
    success, note, values = run_fortml(fortml, args.target, oracle_path)
    rows: list[dict[str, object]] = []
    if success:
        observed = [values.get("gradient:%d" % i, float("nan")) for i in range(1, 5)]
        observed_value = values.get("value:1", float("nan"))
        observed_tangent = values.get("jvp:1", float("nan"))
        errors = [abs(observed_value - expected_value),
                  *[abs(a - b) for a, b in zip(observed, expected_gradient)],
                  abs(observed_tangent - expected_tangent)]
        passed = all(np.isfinite(errors)) and max(errors) <= TOLERANCE
        status = "pass" if passed else "fail"
        timing = timing_seconds(fortml, args.target) if passed else float("nan")
        rows.append(make_row(details, "value_gradient", "fortml", "cpu", status,
                             "validation_mse", observed_value, max(errors), timing, note))
        rows.append(make_row(details, "jvp", "fortml", "cpu", status,
                             "directional_validation_mse_derivative", observed_tangent,
                             max(errors), timing, note))
        for index, observed_component in enumerate(observed, 1):
            rows.append(make_row(details, "gradient_component", "fortml", "cpu", status,
                                 f"gradient_parameter_{index}", observed_component,
                                 max(errors), timing, note))
    else:
        rows.append(make_row(details, "value_gradient", "fortml", "cpu", "unavailable",
                             "validation_mse", "", "", "", note))
    for phase in ("value_gradient", "jvp"):
        rows.append(make_row(details, phase, "fortml", "cuda", "unavailable", "", "", "", "",
                             "resident CUDA trajectory kernel is not linked; no host fallback"))
    rows.append(make_row(details, "hvp", "fortml", "cpu", "unavailable", "", "", "", "",
                         "outer hyper-HVP requires third network derivatives; typed refusal"))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
