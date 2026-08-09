#!/usr/bin/env python3
"""Correctness-gated weighted multilabel MLP objective benchmark.

The Fortran probe supplies a fitted one-layer shared head and objective
products. NumPy recomputes weighted BCE, direct/log-L2 gradients, and mixed
Hessian-vector products from the emitted packed state. The release rows are
written only after both coordinate modes and the bounded optimizer statuses
pass their checks.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


N, P, LABELS = 6, 2, 2
L2 = 0.02
FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "seconds", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.column_stack((
        np.asarray((-1.0, -0.5, 0.0, 0.5, 1.0, 1.2), dtype=np.float64),
        np.asarray((-1.0, -0.2, 0.0, 0.2, 1.0, 0.8), dtype=np.float64),
    ))
    targets = np.column_stack((
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64),
        np.asarray((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), dtype=np.float64),
    ))
    sample = np.asarray((0.5, 1.0, 1.5, 2.0, 0.75, 1.25), dtype=np.float64)
    classes = np.asarray(((1.1, 0.9), (0.8, 1.4)), dtype=np.float64)
    return x, targets, sample, classes


def sigmoid(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, 1.0 / (1.0 + np.exp(-value)),
                    np.exp(value) / (1.0 + np.exp(value)))


def products(parameters: np.ndarray, direction: np.ndarray, log_mode: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    x, targets, sample, class_factors = fixture()
    theta = parameters[: LABELS * (P + 1)]
    coordinate = parameters[-1]
    l2 = float(np.exp(coordinate) if log_mode else coordinate)
    direction_theta = direction[: LABELS * (P + 1)]
    direction_coordinate = direction[-1]
    value = 0.0
    gradient = np.zeros_like(parameters)
    hvp = np.zeros_like(parameters)
    norm2 = float(np.dot(theta, theta))
    for label in range(LABELS):
        indices = np.asarray([label * P + k for k in range(P)] + [LABELS * P + label])
        block = theta[indices]
        block_dot = direction_theta[indices]
        logits = x @ block[:P] + block[P]
        residual = sigmoid(logits) - targets[:, label]
        weights = sample * np.where(targets[:, label] >= 0.5,
                                    class_factors[1, label], class_factors[0, label])
        normalization = float(np.sum(weights))
        value += float(np.sum(weights * (np.logaddexp(0.0, logits) - targets[:, label] * logits))) / normalization
        jacobian = np.column_stack((x, np.ones(N)))
        gradient[indices] = jacobian.T @ (weights * residual) / normalization
        score_dot = x @ block_dot[:P] + block_dot[P]
        curvature = weights * sigmoid(logits) * (1.0 - sigmoid(logits))
        hvp[indices] = jacobian.T @ (curvature * score_dot) / normalization
    value /= LABELS
    gradient[: LABELS * (P + 1)] /= LABELS
    hvp[: LABELS * (P + 1)] /= LABELS
    value += 0.5 * l2 * norm2
    gradient[: LABELS * (P + 1)] += l2 * theta
    hvp[: LABELS * (P + 1)] += l2 * direction_theta
    if log_mode:
        gradient[-1] = 0.5 * l2 * norm2
        hvp[: LABELS * (P + 1)] += l2 * direction_coordinate * theta
        hvp[-1] = l2 * float(np.dot(theta, direction_theta)) + 0.5 * l2 * norm2 * direction_coordinate
    else:
        gradient[-1] = 0.5 * norm2
        hvp[: LABELS * (P + 1)] += direction_coordinate * theta
        hvp[-1] = float(np.dot(theta, direction_theta))
    return value, gradient, hvp


def build_probe(fortml: Path, fixture_path: Path) -> tuple[str, float]:
    build = subprocess.run(["fo", "build", "--flag", "-O2"], cwd=fortml,
                           capture_output=True, text=True, check=False)
    if build.returncode:
        raise RuntimeError(build.stderr.strip() or build.stdout.strip())
    archives = list((fortml / "build" / "fo" / "lib").glob("*.a"))
    if not archives:
        raise RuntimeError("fo build produced no archive")
    archive = max(archives, key=lambda path: path.stat().st_mtime_ns)
    module_dir = fortml / "build" / "fo" / "mod"
    compiler = shlex.split(os.environ.get("FO_FC", "gfortran"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError(f"Fortran compiler unavailable: {compiler!r}")
    with tempfile.TemporaryDirectory(prefix="fortml-mlp-objective-", dir="/mnt/storage") as directory:
        directory_path = Path(directory)
        source = directory_path / fixture_path.name
        executable = directory_path / "mlp_multilabel_objective_probe"
        source.write_bytes(fixture_path.read_bytes())
        command = compiler + ["-O2", "-ffree-line-length-none", "-I", str(module_dir),
                              str(source), str(archive), "-o", str(executable)]
        link = subprocess.run(command, cwd=fortml, capture_output=True, text=True, check=False)
        if link.returncode:
            raise RuntimeError(link.stderr.strip() or link.stdout.strip())
        started = time.perf_counter()
        run = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        if run.returncode:
            raise RuntimeError(run.stderr.strip() or run.stdout.strip())
        return run.stdout, elapsed


def parse(stdout: str) -> dict[str, list[list[str]]]:
    rows: dict[str, list[list[str]]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields:
            rows.setdefault(fields[0], []).append(fields[1:])
    return rows


def token(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def vector(rows: dict[str, list[list[str]]], name: str) -> np.ndarray:
    return np.asarray([token(values[1]) for values in sorted(rows[name], key=lambda values: int(values[0]))])


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_multilabel_objective.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    fixture_path = root / "fixtures" / "mlp_multilabel_objective_probe.f90"
    stdout, elapsed = build_probe(fortml, fixture_path)
    observed = parse(stdout)
    required = {
        "objective_theta", "objective_gradient", "objective_hvp", "objective_value",
        "objective_jvp", "objective_vjp_dot", "log_objective_parameters",
        "log_objective_gradient", "log_objective_hvp", "log_objective_value",
        "direct_optimizer", "log_optimizer", "cuda_status",
    }
    if not required.issubset(observed):
        raise RuntimeError(f"probe rows missing: {sorted(required - set(observed))}\n{stdout}")
    direct_parameters = vector(observed, "objective_theta")
    direct_direction = 0.01 * np.arange(1, direct_parameters.size + 1, dtype=np.float64)
    expected_value, expected_gradient, expected_hvp = products(direct_parameters, direct_direction)
    direct_gradient = vector(observed, "objective_gradient")
    direct_hvp = vector(observed, "objective_hvp")
    direct_error = max(
        abs(token(observed["objective_value"][0][0]) - expected_value),
        np.max(np.abs(direct_gradient - expected_gradient)),
        np.max(np.abs(direct_hvp - expected_hvp)),
        abs(token(observed["objective_jvp"][0][0]) - float(np.dot(expected_gradient, direct_direction))),
        abs(token(observed["objective_vjp_dot"][0][0]) - 1.7 * float(np.dot(expected_gradient, direct_direction))),
    )
    log_parameters = vector(observed, "log_objective_parameters")
    _, expected_log_gradient, expected_log_hvp = products(log_parameters, direct_direction, log_mode=True)
    expected_log_value, _, _ = products(log_parameters, direct_direction, log_mode=True)
    log_error = max(
        abs(token(observed["log_objective_value"][0][0]) - expected_log_value),
        np.max(np.abs(vector(observed, "log_objective_gradient") - expected_log_gradient)),
        np.max(np.abs(vector(observed, "log_objective_hvp") - expected_log_hvp)),
    )
    direct_status, direct_converged = int(observed["direct_optimizer"][0][0]), observed["direct_optimizer"][0][1].upper() == "T"
    log_status, log_converged = int(observed["log_optimizer"][0][0]), observed["log_optimizer"][0][1].upper() == "T"
    cuda_status = int(observed["cuda_status"][0][0])
    if direct_error > 2.0e-10 or log_error > 2.0e-10:
        raise RuntimeError(f"multilabel objective oracle mismatch: direct={direct_error:.3e}, log={log_error:.3e}")
    if direct_status != 0 or not direct_converged or log_status != 0 or not log_converged:
        raise RuntimeError(f"optimizer status mismatch: direct={direct_status}/{direct_converged}, log={log_status}/{log_converged}")
    if cuda_status != 3:
        raise RuntimeError(f"expected typed CUDA refusal 3, got {cuda_status}")
    output = args.output.resolve()
    details = {
        "oracle": "independent NumPy weighted multilabel BCE and mixed L2/log-L2 HVP oracle",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O2",
    }
    records = [
        row(details, workload="mlp_multilabel_objective", phase="direct_l2_products", backend="fortml",
            device="cpu", status="pass", metric="max_abs_error", value=direct_error,
            max_abs_error=direct_error, seconds=elapsed,
            notes="weighted mean BCE, network parameters plus direct l2 coordinate"),
        row(details, workload="mlp_multilabel_objective", phase="log_l2_products", backend="fortml",
            device="cpu", status="pass", metric="max_abs_error", value=log_error,
            max_abs_error=log_error, seconds="",
            notes="positive exp(log_l2) coordinate with exact mixed HVP"),
        row(details, workload="mlp_multilabel_objective", phase="lbfgsb", backend="fortopt",
            device="cpu", status="pass", metric="optimizer_status", value=1.0,
            max_abs_error=0.0, seconds="", oracle="FortOpt bounded L-BFGS-B",
            notes="direct and positive log-L2 coordinates converged"),
        row(details, workload="mlp_multilabel_objective", phase="device_capability", backend="fortml",
            device="cuda", status="unavailable", metric="predict_proba", value="nan",
            max_abs_error="nan", oracle="typed device contract", seconds="",
            notes="FORTNUM_NOT_IMPLEMENTED; resident multilabel MLP graph is not linked"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
