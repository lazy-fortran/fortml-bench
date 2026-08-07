#!/usr/bin/env python3
"""Benchmark the AdamW trainer and fixed-trajectory MLP hypergradients.

The NumPy implementations are independent behavioral oracles.  AdamW is
implemented from the decoupled-moment recurrence, while the hypergradient
oracle evaluates the complete inner SGD trajectory and differentiates the
outer validation loss with a central finite difference in log learning-rate
and log L2.  A FortML executable is accepted only after it writes every
prediction, scalar, and hypergradient entry.  Missing release targets are
retained as explicit ``unavailable`` rows.

The protocol is intentionally separate from ``bench_training.py`` so a future
release app can be added without changing the established SGD/Nesterov raw
record.  The app names are ``fortml_bench_adamw_training`` and
``fortml_bench_mlp_hypergradient``; both receive a path through an environment
variable and emit comma-separated timing records.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 96
N_FEATURES = 3
N_HIDDEN = 8
N_OUTPUTS = 1
ADAMW_EPOCHS = 24
ADAMW_LR = 1.0e-2
ADAMW_BETA1 = 0.9
ADAMW_BETA2 = 0.999
ADAMW_EPSILON = 1.0e-8
ADAMW_WEIGHT_DECAY = 1.0e-2
ADAMW_L2 = 1.0e-4
ADAMW_REPETITIONS = 4
HYPER_TRAIN = 72
HYPER_VALIDATION = N_SAMPLES - HYPER_TRAIN
HYPER_STEPS = 8
HYPER_LR = 1.0e-2
HYPER_L2 = 1.0e-4
HYPER_FD_STEP = 2.0e-5
HYPER_REPETITIONS = 16

FIELDS = (
    "workload",
    "phase",
    "variant",
    "backend",
    "device",
    "status",
    "n_samples",
    "n_features",
    "n_hidden",
    "epochs",
    "steps",
    "repetitions",
    "seconds_per_operation",
    "metric",
    "value",
    "max_abs_error",
    "oracle",
    "python_version",
    "numpy_version",
    "torch_version",
    "fortml_revision",
    "benchmark_revision",
    "compiler",
    "flags",
    "notes",
)


def revision(repository: Path, ignored_paths: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status_lines = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines()
    ignored = {path.resolve() for path in ignored_paths}
    dirty = []
    for line in status_lines:
        path_text = line[3:].split(" -> ")[-1].strip()
        if (repository / path_text).resolve() not in ignored:
            dirty.append(line)
    return value + ("+dirty" if dirty else "")


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "installed"))


def metadata(
    root: Path, fortml: Path, ignored_paths: tuple[Path, ...] = ()
) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": package_version("torch"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored_paths),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def base_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict(details)
    row.update(
        {
            "workload": "",
            "phase": "",
            "variant": "",
            "backend": "",
            "device": "cpu",
            "status": "",
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_hidden": N_HIDDEN,
            "epochs": "",
            "steps": "",
            "repetitions": "",
            "seconds_per_operation": "",
            "metric": "",
            "value": "",
            "max_abs_error": "",
            "oracle": "",
            "notes": "",
        }
    )
    row.update(values)
    return row


def mlp_inputs() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * rows + 0.13 * columns)
    x += 0.15 * np.cos(0.009 * rows * columns)
    target = 0.4 * np.sin(x[:, :1]) + 0.2 * x[:, 1:2] - 0.1 * x[:, 2:3]
    target += 0.03 * np.cos(2.0 * x[:, :1])
    return x, target


def initial_theta(seed: int = 23) -> np.ndarray:
    layers = ((N_FEATURES, N_HIDDEN), (N_HIDDEN, N_OUTPUTS))
    pieces: list[np.ndarray] = []
    for layer_index, (n_in, n_out) in enumerate(layers, start=1):
        scale = np.sqrt(6.0 / float(n_in + n_out))
        index = np.arange(1, n_in * n_out + 1, dtype=np.float64)
        phase = seed + 1009 * layer_index + 9176 * index
        weight = (scale * np.sin(phase)).reshape((n_in, n_out), order="F")
        bias_index = np.arange(1, n_out + 1, dtype=np.float64)
        bias = 0.01 * scale * np.sin(seed + 1009 * layer_index + 7919 * bias_index)
        pieces.extend((weight.reshape(-1, order="F"), bias))
    return np.concatenate(pieces)


def unpack_theta(theta: np.ndarray) -> tuple[np.ndarray, ...]:
    position = 0
    count = N_FEATURES * N_HIDDEN
    weight_1 = theta[position : position + count].reshape(
        (N_FEATURES, N_HIDDEN), order="F"
    )
    position += count
    bias_1 = theta[position : position + N_HIDDEN]
    position += N_HIDDEN
    count = N_HIDDEN * N_OUTPUTS
    weight_2 = theta[position : position + count].reshape(
        (N_HIDDEN, N_OUTPUTS), order="F"
    )
    position += count
    bias_2 = theta[position : position + N_OUTPUTS]
    return weight_1, bias_1, weight_2, bias_2


def mlp_value_gradient(
    theta: np.ndarray, x: np.ndarray, target: np.ndarray, l2: float
) -> tuple[float, np.ndarray, np.ndarray]:
    weight_1, bias_1, weight_2, bias_2 = unpack_theta(theta)
    hidden = np.tanh(x @ weight_1 + bias_1)
    prediction = hidden @ weight_2 + bias_2
    residual = prediction - target
    n = float(x.shape[0])
    preactivation_bar = (residual / n) @ weight_2.T
    preactivation_bar *= 1.0 - hidden * hidden
    gradient = np.concatenate(
        (
            (x.T @ preactivation_bar).reshape(-1, order="F"),
            np.sum(preactivation_bar, axis=0),
            (hidden.T @ (residual / n)).reshape(-1, order="F"),
            np.sum(residual / n, axis=0),
        )
    )
    value = 0.5 * np.sum(residual * residual) / n + 0.5 * l2 * np.sum(theta**2)
    gradient += l2 * theta
    return float(value), gradient, prediction


def adamw_oracle() -> dict[str, Any]:
    x, target = mlp_inputs()
    theta = initial_theta()
    initial_loss, _, _ = mlp_value_gradient(theta, x, target, ADAMW_L2)
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    for step in range(1, ADAMW_EPOCHS + 1):
        _, gradient, _ = mlp_value_gradient(theta, x, target, ADAMW_L2)
        first = ADAMW_BETA1 * first + (1.0 - ADAMW_BETA1) * gradient
        second = ADAMW_BETA2 * second + (1.0 - ADAMW_BETA2) * gradient**2
        first_hat = first / (1.0 - ADAMW_BETA1**step)
        second_hat = second / (1.0 - ADAMW_BETA2**step)
        theta = (1.0 - ADAMW_LR * ADAMW_WEIGHT_DECAY) * theta
        theta -= ADAMW_LR * first_hat / (np.sqrt(second_hat) + ADAMW_EPSILON)
    final_loss, _, prediction = mlp_value_gradient(theta, x, target, ADAMW_L2)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "prediction": prediction[:, 0],
        "theta": theta,
    }


def hyper_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, target = mlp_inputs()
    return x[:HYPER_TRAIN], target[:HYPER_TRAIN], x[HYPER_TRAIN:], target[HYPER_TRAIN:]


def hyper_value(log_parameters: np.ndarray) -> float:
    learning_rate, l2 = np.exp(log_parameters)
    train_x, train_target, validation_x, validation_target = hyper_fixture()
    theta = initial_theta()
    for _ in range(HYPER_STEPS):
        _, gradient, _ = mlp_value_gradient(theta, train_x, train_target, l2)
        theta -= learning_rate * gradient
    value, _, _ = mlp_value_gradient(theta, validation_x, validation_target, 0.0)
    return value


def hyper_oracle() -> dict[str, Any]:
    parameters = np.log(np.array([HYPER_LR, HYPER_L2], dtype=np.float64))
    gradient = np.empty(2, dtype=np.float64)
    for index in range(2):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += HYPER_FD_STEP
        minus[index] -= HYPER_FD_STEP
        gradient[index] = (hyper_value(plus) - hyper_value(minus)) / (
            2.0 * HYPER_FD_STEP
        )
    direction = np.array([0.7, -0.3], dtype=np.float64)
    tangent = (
        hyper_value(parameters + HYPER_FD_STEP * direction)
        - hyper_value(parameters - HYPER_FD_STEP * direction)
    ) / (2.0 * HYPER_FD_STEP)
    return {
        "parameters": parameters,
        "value": hyper_value(parameters),
        "gradient": gradient,
        "direction": direction,
        "jvp": float(tangent),
    }


def numpy_rows(details: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_adamw = adamw_oracle()
    started = time.perf_counter()
    for _ in range(ADAMW_REPETITIONS):
        adamw_oracle()
    seconds = (time.perf_counter() - started) / ADAMW_REPETITIONS
    rows.append(
        base_row(
            details,
            workload="mlp_adamw_training",
            phase="fit",
            variant="adamw",
            backend="numpy_oracle",
            status="pass",
            epochs=ADAMW_EPOCHS,
            repetitions=ADAMW_REPETITIONS,
            seconds_per_operation=seconds,
            metric="final_loss",
            value=expected_adamw["final_loss"],
            max_abs_error=0.0,
            oracle="independent NumPy AdamW moments and decoupled weight decay",
            notes=(
                f"lr={ADAMW_LR:g}; beta1={ADAMW_BETA1:g}; beta2={ADAMW_BETA2:g}; "
                f"epsilon={ADAMW_EPSILON:g}; weight_decay={ADAMW_WEIGHT_DECAY:g}; "
                f"l2={ADAMW_L2:g}; initial_loss={expected_adamw['initial_loss']:.16g}"
            ),
        )
    )
    for metric, value in (
        ("initial_loss", expected_adamw["initial_loss"]),
        ("final_loss", expected_adamw["final_loss"]),
    ):
        rows.append(
            base_row(
                details,
                workload="mlp_adamw_training",
                phase="check",
                variant="adamw",
                backend="numpy_oracle",
                status="pass",
                epochs=ADAMW_EPOCHS,
                repetitions=1,
                metric=metric,
                value=value,
                max_abs_error=0.0,
                oracle="independent NumPy AdamW moments and decoupled weight decay",
            )
        )

    expected_hyper = hyper_oracle()
    started = time.perf_counter()
    for _ in range(HYPER_REPETITIONS):
        hyper_oracle()
    seconds = (time.perf_counter() - started) / HYPER_REPETITIONS
    rows.append(
        base_row(
            details,
            workload="mlp_hypergradient",
            phase="value_gradient",
            variant="fixed_sgd_log_hyperparameters",
            backend="numpy_oracle",
            status="pass",
            n_samples=HYPER_TRAIN + HYPER_VALIDATION,
            steps=HYPER_STEPS,
            repetitions=HYPER_REPETITIONS,
            seconds_per_operation=seconds,
            metric="validation_mse",
            value=expected_hyper["value"],
            max_abs_error=0.0,
            oracle="independent NumPy trajectory with central finite-difference log hypergradient",
            notes=(
                f"train={HYPER_TRAIN}; validation={HYPER_VALIDATION}; "
                f"log_parameters={expected_hyper['parameters'].tolist()}; "
                f"fd_step={HYPER_FD_STEP:g}"
            ),
        )
    )
    rows.append(
        base_row(
            details,
            workload="mlp_hypergradient",
            phase="jvp",
            variant="fixed_sgd_log_hyperparameters",
            backend="numpy_oracle",
            status="pass",
            n_samples=HYPER_TRAIN + HYPER_VALIDATION,
            steps=HYPER_STEPS,
            repetitions=HYPER_REPETITIONS,
            seconds_per_operation=seconds,
            metric="directional_validation_mse_derivative",
            value=expected_hyper["jvp"],
            max_abs_error=0.0,
            oracle="independent NumPy trajectory with central finite-difference JVP",
            notes=f"direction={expected_hyper['direction'].tolist()}; fd_step={HYPER_FD_STEP:g}",
        )
    )
    rows.extend(
        base_row(
            details,
            workload="mlp_hypergradient",
            phase="gradient_component",
            variant="fixed_sgd_log_hyperparameters",
            backend="numpy_oracle",
            status="pass",
            n_samples=HYPER_TRAIN + HYPER_VALIDATION,
            steps=HYPER_STEPS,
            repetitions=1,
            metric=f"gradient_log_{name}",
            value=value,
            max_abs_error=0.0,
            oracle="independent central finite-difference outer objective",
        )
        for name, value in zip(("learning_rate", "l2"), expected_hyper["gradient"])
    )
    return rows, expected_adamw, expected_hyper


def unavailable_rows(details: dict[str, str], workload: str, phases: tuple[str, ...], note: str) -> list[dict[str, Any]]:
    return [
        base_row(
            details,
            workload=workload,
            phase=phase,
            variant="adamw" if workload == "mlp_adamw_training" else "fixed_sgd_log_hyperparameters",
            backend="fortml",
            status="unavailable",
            oracle="FortML release-app protocol",
            notes=note,
        )
        for phase in phases
    ]


def device_refusal_rows(
    details: dict[str, str], workload: str, phases: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Keep the host-only boundary machine-readable and untimed."""

    return [
        base_row(
            details,
            workload=workload,
            phase=phase,
            variant="adamw" if workload == "mlp_adamw_training" else "fixed_sgd_log_hyperparameters",
            backend="fortml",
            device="cuda",
            status="unavailable",
            oracle="FortML device capability boundary",
            notes="current trainer/hypergradient release apps are host-only; no CPU timing is relabeled as CUDA",
        )
        for phase in phases
    ]


def parse_timing(stdout: str, name: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(name)},(.*)$")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            fields = [part.strip() for part in match.group(1).split(",")]
            try:
                return float(fields[-1])
            except (IndexError, ValueError):
                return None
    return None


def read_oracle(path: Path) -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            result[(record["quantity"], int(record.get("index", "1")))] = float(
                record["value"]
            )
    return result


def run_target(
    fortml: Path,
    target: str,
    environment_name: str,
    workload: str,
    phases: tuple[str, ...],
    details: dict[str, str],
    expected_adamw: dict[str, Any],
    expected_hyper: dict[str, Any],
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    with tempfile.TemporaryDirectory(dir="/mnt/storage") as directory:
        oracle_path = Path(directory) / f"{workload}_oracle.csv"
        environment[environment_name] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", target],
            cwd=fortml,
            env=environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip().splitlines()
            note = stderr[-1] if stderr else f"{target}: execution unavailable"
            return unavailable_rows(details, workload, phases, f"{target}: {note}")
        if not oracle_path.is_file():
            return unavailable_rows(details, workload, phases, f"{target}: no oracle was written")
        actual = read_oracle(oracle_path)

    if workload == "mlp_adamw_training":
        prediction = np.array(
            [actual.get(("prediction", index), np.nan) for index in range(1, N_SAMPLES + 1)]
        )
        initial = actual.get(("initial_loss", 1), np.nan)
        final = actual.get(("final_loss", 1), np.nan)
        error = max(
            float(np.max(np.abs(prediction - expected_adamw["prediction"]))),
            abs(initial - expected_adamw["initial_loss"]),
            abs(final - expected_adamw["final_loss"]),
        )
        if not np.isfinite(error) or error > 5.0e-11:
            raise RuntimeError(f"FortML AdamW oracle mismatch: {error:.3e}")
        rows = []
        for phase, metric, value, timing_name in (
            ("fit", "final_loss", final, "mlp_adamw_fit"),
            ("predict", "prediction_sum", float(np.sum(prediction)), "mlp_adamw_predict"),
        ):
            rows.append(
                base_row(
                    details,
                    workload=workload,
                    phase=phase,
                    variant="adamw",
                    backend="fortml",
                    status="pass",
                    epochs=ADAMW_EPOCHS,
                    repetitions=ADAMW_REPETITIONS,
                    seconds_per_operation=parse_timing(completed.stdout, timing_name),
                    metric=metric,
                    value=value,
                    max_abs_error=error,
                    oracle="complete NumPy AdamW prediction/loss arrays",
                    notes=target,
                )
            )
        return rows

    value = actual.get(("value", 1), np.nan)
    gradient = np.array([actual.get(("gradient", index), np.nan) for index in (1, 2)])
    jvp = actual.get(("jvp", 1), np.nan)
    error = max(
        abs(value - expected_hyper["value"]),
        float(np.max(np.abs(gradient - expected_hyper["gradient"]))),
        abs(jvp - expected_hyper["jvp"]),
    )
    if not np.isfinite(error) or error > 2.0e-6:
        raise RuntimeError(f"FortML hypergradient oracle mismatch: {error:.3e}")
    timing = parse_timing(completed.stdout, "mlp_hypergradient_value_gradient")
    return [
        base_row(
            details,
            workload=workload,
            phase="value_gradient",
            variant="fixed_sgd_log_hyperparameters",
            backend="fortml",
            status="pass",
            n_samples=HYPER_TRAIN + HYPER_VALIDATION,
            steps=HYPER_STEPS,
            repetitions=HYPER_REPETITIONS,
            seconds_per_operation=timing,
            metric="validation_mse",
            value=value,
            max_abs_error=error,
            oracle="complete NumPy central-FD validation value/gradient/JVP arrays",
            notes=target,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--adamw-output", type=Path, default=Path("results/adamw_training.csv"))
    parser.add_argument("--hypergradient-output", type=Path, default=Path("results/mlp_hypergradient.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    parser.add_argument("--adamw-target", default="fortml_bench_adamw_training")
    parser.add_argument("--hypergradient-target", default="fortml_bench_mlp_hypergradient")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    adamw_output = args.adamw_output.resolve()
    hyper_output = args.hypergradient_output.resolve()
    ignored = (adamw_output, hyper_output)
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, ignored)
    numpy, expected_adamw, expected_hyper = numpy_rows(details)
    if args.skip_fortml:
        adamw_rows = unavailable_rows(details, "mlp_adamw_training", ("fit", "predict"), "--skip-fortml requested")
        hyper_rows = unavailable_rows(details, "mlp_hypergradient", ("value_gradient", "jvp"), "--skip-fortml requested")
    else:
        build = subprocess.run(
            ["fo", "build", "--flag", "-O3"],
            cwd=fortml,
            env={**os.environ, "FO_FC": os.environ.get("FO_FC", "gfortran")},
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
            adamw_rows = unavailable_rows(details, "mlp_adamw_training", ("fit", "predict"), f"build unavailable: {note}")
            hyper_rows = unavailable_rows(details, "mlp_hypergradient", ("value_gradient", "jvp"), f"build unavailable: {note}")
        else:
            adamw_rows = run_target(
                fortml, args.adamw_target, "FORTML_BENCH_ADAMW_ORACLE", "mlp_adamw_training",
                ("fit", "predict"), details, expected_adamw, expected_hyper
            )
            hyper_rows = run_target(
                fortml, args.hypergradient_target, "FORTML_BENCH_HYPERGRADIENT_ORACLE", "mlp_hypergradient",
                ("value_gradient", "jvp"), details, expected_adamw, expected_hyper
            )
    adamw_rows.extend(
        device_refusal_rows(details, "mlp_adamw_training", ("fit", "predict"))
    )
    hyper_rows.extend(
        device_refusal_rows(details, "mlp_hypergradient", ("value_gradient", "jvp"))
    )
    adamw_output.parent.mkdir(parents=True, exist_ok=True)
    hyper_output.parent.mkdir(parents=True, exist_ok=True)
    with adamw_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([row for row in numpy if row["workload"] == "mlp_adamw_training"] + adamw_rows)
    with hyper_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([row for row in numpy if row["workload"] == "mlp_hypergradient"] + hyper_rows)
    print(f"wrote {len([row for row in numpy if row['workload'] == 'mlp_adamw_training']) + len(adamw_rows)} rows to {adamw_output}")
    print(f"wrote {len([row for row in numpy if row['workload'] == 'mlp_hypergradient']) + len(hyper_rows)} rows to {hyper_output}")


if __name__ == "__main__":
    main()
