#!/usr/bin/env python3
"""Correctness-gated multilabel MLP classifier benchmark.

The fixture has two independent sigmoid heads over the same two-feature data.
Each head performs one full-batch Adam step.  The NumPy oracle reproduces the
deterministic MLP initializer, weighted binary cross entropy, parameter
gradient, and exact parameter Hessian-vector product independently.  The
multilabel wrapper contract is a concatenation of head parameters and a mean
of the two per-head objectives/derivatives.
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


N, P, LABELS, SEED = 6, 2, 2, 29
LR, BETA1, BETA2, EPS, L2 = 0.03, 0.8, 0.95, 1.0e-7, 0.02
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
    dirty = False
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        if line[3:].split(" -> ")[-1].strip() not in ignored_names:
            dirty = True
            break
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack((
        np.asarray((-1.0, -0.5, 0.0, 0.5, 1.0, 1.2), dtype=np.float64),
        np.asarray((-1.0, -0.2, 0.0, 0.2, 1.0, 0.8), dtype=np.float64),
    ))
    targets = np.column_stack((
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64),
        np.asarray((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), dtype=np.float64),
    ))
    return x, targets


def sigmoid(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, 1.0 / (1.0 + np.exp(-value)),
                    np.exp(value) / (1.0 + np.exp(value)))


def objective(theta: np.ndarray, x: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    logits = x @ theta[:P] + theta[P]
    value = float(np.mean(np.logaddexp(0.0, logits) - target * logits)
                  + 0.5 * L2 * np.dot(theta, theta))
    residual = sigmoid(logits) - target
    jacobian = np.column_stack((x, np.ones(N)))
    gradient = jacobian.T @ residual / N + L2 * theta
    return value, gradient


def oracle() -> dict[str, np.ndarray | float | int]:
    x, targets = fixture()
    scale = np.sqrt(6.0 / (P + 1.0))
    initial = np.array([
        scale * np.sin(SEED + 1009 + 9176 * index) for index in (1, 2)
    ] + [0.01 * scale * np.sin(SEED + 1009 + 7919)], dtype=np.float64)
    theta_parts = []
    gradient_parts = []
    hvp_parts = []
    probability_parts = []
    prediction_parts = []
    losses = []
    direction = 0.01 * np.arange(1, LABELS * (P + 1) + 1, dtype=np.float64)
    for label in range(LABELS):
        theta = initial.copy()
        initial_loss, gradient = objective(theta, x, targets[:, label])
        del initial_loss
        # For the first Adam step, bias correction reduces to g/(|g|+eps).
        theta = theta - LR * gradient / (np.abs(gradient) + EPS)
        loss, final_gradient = objective(theta, x, targets[:, label])
        logits = x @ theta[:P] + theta[P]
        positive = sigmoid(logits)
        weights = positive * (1.0 - positive)
        jacobian = np.column_stack((x, np.ones(N)))
        hessian = (jacobian.T * weights) @ jacobian / N + L2 * np.eye(P + 1)
        theta_parts.append(theta)
        gradient_parts.append(final_gradient)
        direction_part = direction[label * (P + 1):(label + 1) * (P + 1)]
        hvp_parts.append(hessian @ direction_part)
        probability_parts.append(positive)
        prediction_parts.append((logits >= 0.0).astype(np.int64))
        losses.append(loss)
    return {
        "theta": np.concatenate(theta_parts),
        "probabilities": np.column_stack(probability_parts),
        "predicted": np.column_stack(prediction_parts),
        "loss": float(np.mean(losses)),
        "gradient": np.concatenate(gradient_parts) / LABELS,
        "hvp": np.concatenate(hvp_parts) / LABELS,
        "direction": direction,
    }


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
    with tempfile.TemporaryDirectory(prefix="fortml-mlp-multilabel-", dir="/mnt/storage") as directory:
        directory_path = Path(directory)
        source = directory_path / fixture_path.name
        executable = directory_path / "mlp_multilabel_classifier_probe"
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
        if fields and fields[0].startswith("mlp_multilabel_"):
            rows.setdefault(fields[0], []).append(fields[1:])
    return rows


def token(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_multilabel_classifier.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    fixture_path = root / "fixtures" / "mlp_multilabel_classifier_probe.f90"
    expected = oracle()
    stdout, elapsed = build_probe(fortml, fixture_path)
    observed = parse(stdout)
    required = {"mlp_multilabel_parameter_count", "mlp_multilabel_loss",
                "mlp_multilabel_theta", "mlp_multilabel_gradient",
                "mlp_multilabel_hvp", "mlp_multilabel_probability",
                "mlp_multilabel_prediction", "mlp_multilabel_cuda"}
    if not required.issubset(observed):
        raise RuntimeError(f"probe rows missing: {sorted(required - set(observed))}\n{stdout}")
    if int(observed["mlp_multilabel_parameter_count"][0][0]) != LABELS * (P + 1):
        raise RuntimeError("unexpected multilabel parameter count")
    theta = np.array([token(values[1]) for values in sorted(
        observed["mlp_multilabel_theta"], key=lambda values: int(values[0]))])
    gradient = np.array([token(values[1]) for values in sorted(
        observed["mlp_multilabel_gradient"], key=lambda values: int(values[0]))])
    hvp = np.array([token(values[1]) for values in sorted(
        observed["mlp_multilabel_hvp"], key=lambda values: int(values[0]))])
    probabilities = np.zeros((N, LABELS))
    predicted = np.zeros((N, LABELS), dtype=np.int64)
    for values in observed["mlp_multilabel_probability"]:
        probabilities[int(values[0]) - 1, int(values[1]) - 1] = token(values[2])
    for values in observed["mlp_multilabel_prediction"]:
        predicted[int(values[0]) - 1, int(values[1]) - 1] = int(values[2])
    errors = [
        np.max(np.abs(theta - expected["theta"])),
        abs(token(observed["mlp_multilabel_loss"][0][0]) - expected["loss"]),
        np.max(np.abs(gradient - expected["gradient"])),
        np.max(np.abs(hvp - expected["hvp"])),
        np.max(np.abs(probabilities - expected["probabilities"])),
    ]
    error = float(max(errors))
    if error > 3.0e-11 or not np.array_equal(predicted, expected["predicted"]):
        raise RuntimeError(f"multilabel MLP oracle mismatch: error={error:.3e}, predicted={predicted}")
    cuda_code = int(observed["mlp_multilabel_cuda"][0][0])
    if cuda_code != 3:
        raise RuntimeError(f"expected FORTNUM_NOT_IMPLEMENTED=3, got {cuda_code}")
    output = args.output.resolve()
    details = {
        "oracle": "independent NumPy Xavier initializer, per-head BCE/Adam/Hessian oracle",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O2",
    }
    records = [
        row(details, workload="mlp_multilabel_classifier", phase="fit_predict", backend="fortml",
            device="cpu", status="pass", metric="max_abs_error", value=error,
            max_abs_error=error, seconds=elapsed,
            notes="two sigmoid heads; layers=2->1 each; one full-batch Adam step; l2=0.02; seed=29"),
        row(details, workload="mlp_multilabel_classifier", phase="derivatives", backend="fortml",
            device="cpu", status="pass", metric="gradient_hvp_probability_error", value=error,
            max_abs_error=error, seconds="", notes="mean per-head BCE gradient and exact parameter HVP"),
        row(details, workload="mlp_multilabel_classifier", phase="device_capability", backend="fortml",
            device="cuda", status="unavailable", metric="predict_proba", value="nan",
            max_abs_error="nan", oracle="typed device contract", seconds="",
            notes="FORTNUM_NOT_IMPLEMENTED; resident multilabel MLP CUDA graph is not linked"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
