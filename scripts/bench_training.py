#!/usr/bin/env python3
"""Benchmark FortML's SGD/Nesterov trainer and differentiable imputer.

The NumPy paths below are deliberately independent behavioral oracles.  The
FortML release app must emit every prediction, fitted statistic, transform,
JVP, and VJP entry before any timing row is retained.  The CSV therefore
contains both mathematical evidence and reproducible release-build timings.
"""

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


MLP_N = 96
MLP_D = 3
MLP_HIDDEN = 8
MLP_OUTPUTS = 1
MLP_EPOCHS = 24
MLP_LR = 0.01
MLP_MOMENTUM = 0.8
MLP_L2 = 1.0e-4
MLP_REPETITIONS = 4
IMPUTER_N = 12
IMPUTER_D = 4
IMPUTER_FIT_REPETITIONS = 32
IMPUTER_PRODUCT_REPETITIONS = 128
IMPUTER_STRATEGIES = ("mean", "median", "constant")
IMPUTER_FILL = -0.25

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
    "repetitions",
    "seconds_per_operation",
    "metric",
    "value",
    "max_abs_error",
    "oracle",
    "python_version",
    "numpy_version",
    "sklearn_version",
    "torch_version",
    "jax_version",
    "xgboost_version",
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
        "sklearn_version": package_version("sklearn"),
        "torch_version": package_version("torch"),
        "jax_version": package_version("jax"),
        "xgboost_version": package_version("xgboost"),
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
            "n_samples": "",
            "n_features": "",
            "n_hidden": "",
            "epochs": "",
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
    rows = np.arange(1, MLP_N + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, MLP_D + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * rows + 0.13 * columns)
    x += 0.15 * np.cos(0.009 * rows * columns)
    target = 0.4 * np.sin(x[:, :1]) + 0.2 * x[:, 1:2] - 0.1 * x[:, 2:3]
    target += 0.03 * np.cos(2.0 * x[:, :1])
    return x, target


def initial_theta(seed: int = 23) -> np.ndarray:
    layers = ((MLP_D, MLP_HIDDEN), (MLP_HIDDEN, MLP_OUTPUTS))
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
    count = MLP_D * MLP_HIDDEN
    weight_1 = theta[position : position + count].reshape(
        (MLP_D, MLP_HIDDEN), order="F"
    )
    position += count
    bias_1 = theta[position : position + MLP_HIDDEN]
    position += MLP_HIDDEN
    count = MLP_HIDDEN * MLP_OUTPUTS
    weight_2 = theta[position : position + count].reshape(
        (MLP_HIDDEN, MLP_OUTPUTS), order="F"
    )
    position += count
    bias_2 = theta[position : position + MLP_OUTPUTS]
    return weight_1, bias_1, weight_2, bias_2


def mlp_value_gradient(
    theta: np.ndarray, x: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    weight_1, bias_1, weight_2, bias_2 = unpack_theta(theta)
    hidden = np.tanh(x @ weight_1 + bias_1)
    prediction = hidden @ weight_2 + bias_2
    residual = prediction - target
    n = float(x.shape[0])
    preactivation_bar = (residual / n) @ weight_2.T
    preactivation_bar *= 1.0 - hidden * hidden
    weight_1_bar = x.T @ preactivation_bar
    bias_1_bar = np.sum(preactivation_bar, axis=0)
    weight_2_bar = hidden.T @ (residual / n)
    bias_2_bar = np.sum(residual / n, axis=0)
    gradient = np.concatenate(
        (
            weight_1_bar.reshape(-1, order="F"),
            bias_1_bar,
            weight_2_bar.reshape(-1, order="F"),
            bias_2_bar,
        )
    )
    value = 0.5 * np.sum(residual * residual) / n + 0.5 * MLP_L2 * np.sum(theta**2)
    gradient += MLP_L2 * theta
    return float(value), gradient, prediction


def mlp_oracle(nesterov: bool) -> dict[str, Any]:
    x, target = mlp_inputs()
    theta = initial_theta()
    initial_loss, _, _ = mlp_value_gradient(theta, x, target)
    velocity = np.zeros_like(theta)
    for _epoch in range(MLP_EPOCHS):
        _loss, gradient, _prediction = mlp_value_gradient(theta, x, target)
        velocity = MLP_MOMENTUM * velocity + gradient
        direction = gradient + MLP_MOMENTUM * velocity if nesterov else velocity
        theta -= MLP_LR * direction
    final_loss, _gradient, prediction = mlp_value_gradient(theta, x, target)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "prediction": prediction[:, 0],
    }


def imputer_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty((IMPUTER_N, IMPUTER_D), dtype=np.float64)
    x_dot = np.empty_like(x)
    output_bar = np.empty_like(x)
    x[:, 0] = 0.2 * np.arange(1, IMPUTER_N + 1, dtype=np.float64)
    x[:, 1] = -0.3 + 0.15 * np.arange(1, IMPUTER_N + 1, dtype=np.float64)
    x[:, 2] = np.sin(0.3 * np.arange(1, IMPUTER_N + 1, dtype=np.float64))
    x[:, 3] = 0.1 * np.arange(1, IMPUTER_N + 1, dtype=np.float64) ** 2
    x[[1, 7], 0] = np.nan
    x[[0, 11], 1] = np.nan
    x[[3, 6], 2] = np.nan
    x[[2, 4, 9], 3] = np.nan
    rows = np.arange(1, IMPUTER_N + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, IMPUTER_D + 1, dtype=np.float64)[None, :]
    x_dot[:] = 0.01 * np.cos(0.2 * rows + 0.1 * columns)
    output_bar[:] = 0.2 * rows - 0.03 * columns
    return x, x_dot, output_bar


def imputer_oracle(strategy: str) -> dict[str, np.ndarray]:
    x, x_dot, output_bar = imputer_inputs()
    if strategy == "mean":
        statistic = np.nanmean(x, axis=0)
    elif strategy == "median":
        statistic = np.nanmedian(x, axis=0)
    else:
        statistic = np.full(IMPUTER_D, IMPUTER_FILL, dtype=np.float64)
    transformed = np.where(np.isnan(x), statistic[None, :], x)
    transformed_dot = np.where(np.isnan(x), 0.0, x_dot)
    input_bar = np.where(np.isnan(x), 0.0, output_bar)
    return {
        "statistic": statistic,
        "transform": transformed,
        "jvp": transformed_dot,
        "vjp": input_bar,
    }


def numpy_rows(details: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for nesterov, variant in ((False, "sgd"), (True, "nesterov")):
        x, target = mlp_inputs()
        expected = mlp_oracle(nesterov)
        started = time.perf_counter()
        for _ in range(MLP_REPETITIONS):
            mlp_oracle(nesterov)
        seconds = (time.perf_counter() - started) / MLP_REPETITIONS
        rows.append(
            base_row(
                details,
                workload="mlp_training",
                phase="fit",
                variant=variant,
                backend="numpy_oracle",
                status="pass",
                n_samples=MLP_N,
                n_features=MLP_D,
                n_hidden=MLP_HIDDEN,
                epochs=MLP_EPOCHS,
                repetitions=MLP_REPETITIONS,
                seconds_per_operation=seconds,
                metric="final_loss",
                value=expected["final_loss"],
                max_abs_error=0.0,
                oracle="independent NumPy tanh MSE gradient and SGD recurrence",
                notes=f"initial_loss={expected['initial_loss']:.16g}; momentum={MLP_MOMENTUM:g}; nesterov={nesterov}",
            )
        )
        for phase, metric, value in (
            ("initial", "initial_loss", expected["initial_loss"]),
            ("final", "final_loss", expected["final_loss"]),
        ):
            rows.append(
                base_row(
                    details,
                    workload="mlp_training",
                    phase=phase,
                    variant=variant,
                    backend="numpy_oracle",
                    status="pass",
                    n_samples=MLP_N,
                    n_features=MLP_D,
                    n_hidden=MLP_HIDDEN,
                    epochs=MLP_EPOCHS,
                    repetitions=1,
                    metric=metric,
                    value=value,
                    max_abs_error=0.0,
                    oracle="independent NumPy tanh MSE gradient and SGD recurrence",
                )
            )

    for strategy in IMPUTER_STRATEGIES:
        expected = imputer_oracle(strategy)
        x, x_dot, output_bar = imputer_inputs()
        fit_started = time.perf_counter()
        for _ in range(IMPUTER_FIT_REPETITIONS):
            imputer_oracle(strategy)
        fit_seconds = (time.perf_counter() - fit_started) / IMPUTER_FIT_REPETITIONS
        # Keep the timed loops separate from the checks and from each other.
        transformed = expected["transform"]
        started = time.perf_counter()
        for _ in range(IMPUTER_PRODUCT_REPETITIONS):
            np.where(np.isnan(x), expected["statistic"][None, :], x)
        transform_seconds = (
            time.perf_counter() - started
        ) / IMPUTER_PRODUCT_REPETITIONS
        started = time.perf_counter()
        for _ in range(IMPUTER_PRODUCT_REPETITIONS):
            np.where(np.isnan(x), 0.0, x_dot)
        jvp_seconds = (time.perf_counter() - started) / IMPUTER_PRODUCT_REPETITIONS
        started = time.perf_counter()
        for _ in range(IMPUTER_PRODUCT_REPETITIONS):
            np.where(np.isnan(x), 0.0, output_bar)
        vjp_seconds = (time.perf_counter() - started) / IMPUTER_PRODUCT_REPETITIONS
        for phase, seconds, metric, value in (
            ("fit", fit_seconds, "statistic_sum", np.sum(expected["statistic"])),
            ("transform", transform_seconds, "transform_sum", np.sum(transformed)),
            ("jvp", jvp_seconds, "jvp_sum", np.sum(expected["jvp"])),
            ("vjp", vjp_seconds, "vjp_sum", np.sum(expected["vjp"])),
        ):
            rows.append(
                base_row(
                    details,
                    workload="simple_imputer",
                    phase=phase,
                    variant=strategy,
                    backend="numpy_oracle",
                    status="pass",
                    n_samples=IMPUTER_N,
                    n_features=IMPUTER_D,
                    repetitions=(
                        IMPUTER_FIT_REPETITIONS
                        if phase == "fit"
                        else IMPUTER_PRODUCT_REPETITIONS
                    ),
                    seconds_per_operation=seconds,
                    metric=metric,
                    value=float(value),
                    max_abs_error=0.0,
                    oracle="independent NumPy NaN statistic and piecewise derivative rules",
                    notes=(
                        f"fill_value={IMPUTER_FILL:g}"
                        if strategy == "constant"
                        else "NaNs are missing; observed entries are identity"
                    ),
                )
            )
    return rows


def parse_stdout(stdout: str) -> dict[tuple[str, str], list[str]]:
    records: dict[tuple[str, str], list[str]] = {}
    for line in stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if fields and fields[0] == "mlp_training" and len(fields) == 9:
            records[("mlp_training", fields[1])] = fields[2:]
        if fields and fields[0] == "simple_imputer" and len(fields) == 6:
            records[("simple_imputer", f"{fields[1]}:{fields[2]}")] = fields[3:]
    return records


def parse_oracle(
    path: Path,
) -> dict[tuple[str, str, str], dict[tuple[int, int], float]]:
    values: dict[tuple[str, str, str], dict[tuple[int, int], float]] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            key = (record["workload"], record["variant"], record["quantity"])
            row = int(record["row"])
            column = int(record["column"])
            values.setdefault(key, {})[(row, column)] = float(record["value"])
    return values


def array_from_records(
    records: dict[tuple[int, int], float], shape: tuple[int, int]
) -> np.ndarray:
    result = np.full(shape, np.nan)
    for (row, column), value in records.items():
        if not (1 <= row <= shape[0] and 1 <= column <= shape[1]):
            raise RuntimeError("FortML training oracle index is out of range")
        result[row - 1, column - 1] = value
    if not np.isfinite(result).all():
        raise RuntimeError("FortML training oracle is incomplete")
    return result


def checked_fortml(
    stdout: str, oracle_path: Path, details: dict[str, str]
) -> list[dict[str, Any]]:
    records = parse_stdout(stdout)
    arrays = parse_oracle(oracle_path)
    rows: list[dict[str, Any]] = []
    for nesterov, variant in ((False, "sgd"), (True, "nesterov")):
        expected = mlp_oracle(nesterov)
        predicted = array_from_records(
            arrays.get(("mlp_training", variant, "prediction"), {}),
            (MLP_N, 1),
        )[:, 0]
        initial = arrays.get(("mlp_training", variant, "initial_loss"), {})
        final = arrays.get(("mlp_training", variant, "final_loss"), {})
        if len(initial) != 1 or len(final) != 1:
            raise RuntimeError(f"FortML {variant} loss oracle is incomplete")
        initial_value = next(iter(initial.values()))
        final_value = next(iter(final.values()))
        error = max(
            float(np.max(np.abs(predicted - expected["prediction"]))),
            abs(initial_value - expected["initial_loss"]),
            abs(final_value - expected["final_loss"]),
        )
        if error > 5.0e-12:
            raise RuntimeError(
                f"FortML {variant} training oracle mismatch: {error:.3e}"
            )
        fields = records.get(("mlp_training", variant))
        if fields is None:
            raise RuntimeError(f"FortML {variant} timing record is missing")
        rows.append(
            base_row(
                details,
                workload="mlp_training",
                phase="fit",
                variant=variant,
                backend="fortml",
                status="pass",
                n_samples=MLP_N,
                n_features=MLP_D,
                n_hidden=MLP_HIDDEN,
                epochs=MLP_EPOCHS,
                repetitions=MLP_REPETITIONS,
                seconds_per_operation=float(fields[-1]),
                metric="final_loss",
                value=final_value,
                max_abs_error=error,
                oracle="complete NumPy SGD/Nesterov prediction and loss arrays",
                notes=f"release app fortml_bench_training; initial_loss={initial_value:.16g}",
            )
        )

    for strategy in IMPUTER_STRATEGIES:
        expected = imputer_oracle(strategy)
        errors = []
        for quantity in ("statistic", "transform", "jvp", "vjp"):
            records_for_quantity = arrays.get(
                ("simple_imputer", strategy, quantity), {}
            )
            if quantity == "statistic":
                actual = array_from_records(records_for_quantity, (1, IMPUTER_D))[0]
                target = expected[quantity]
            else:
                actual = array_from_records(
                    records_for_quantity, (IMPUTER_N, IMPUTER_D)
                )
                target = expected[quantity]
            errors.append(float(np.max(np.abs(actual - target))))
        error = max(errors)
        if error > 5.0e-13:
            raise RuntimeError(
                f"FortML {strategy} imputer oracle mismatch: {error:.3e}"
            )
        for phase in ("fit", "transform", "jvp", "vjp"):
            fields = records.get(("simple_imputer", f"{strategy}:{phase}"))
            if fields is None:
                raise RuntimeError(
                    f"FortML {strategy} {phase} timing record is missing"
                )
            rows.append(
                base_row(
                    details,
                    workload="simple_imputer",
                    phase=phase,
                    variant=strategy,
                    backend="fortml",
                    status="pass",
                    n_samples=IMPUTER_N,
                    n_features=IMPUTER_D,
                    repetitions=(
                        IMPUTER_FIT_REPETITIONS
                        if phase == "fit"
                        else IMPUTER_PRODUCT_REPETITIONS
                    ),
                    seconds_per_operation=float(fields[-1]),
                    metric=f"{phase}_sum",
                    value=float(
                        np.sum(expected["statistic" if phase == "fit" else phase])
                    ),
                    max_abs_error=error,
                    oracle="complete NumPy NaN statistic/transform/JVP/VJP arrays",
                    notes="missing entries use fitted statistic; derivative entries are zero",
                )
            )
    return rows


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True
    )
    with tempfile.TemporaryDirectory(dir="/mnt/storage") as directory:
        oracle_path = Path(directory) / "fortml_training_oracle.csv"
        environment["FORTML_BENCH_TRAINING_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_training"],
            cwd=fortml,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return checked_fortml(completed.stdout, oracle_path, details)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/training_imputer.csv")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    details = metadata(root, args.fortml.resolve(), (output,))
    rows = numpy_rows(details)
    rows.extend(run_fortml(args.fortml.resolve(), details))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
