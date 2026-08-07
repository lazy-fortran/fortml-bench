#!/usr/bin/env python3
"""Correctness-gated binary MLP classifier release benchmark.

The fixture is a one-layer sigmoid head and one full-batch Adam update.  A
NumPy oracle independently reproduces the deterministic Xavier/trigonometric
initializer, weighted BCE derivatives, Adam bias correction, probabilities,
and the exact parameter Hessian-vector product.  This keeps the benchmark
small while exercising the production classifier and derivative APIs.
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


N, P, SEED = 6, 2, 29
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
    labels = np.asarray((-2, -2, -2, 5, 5, 5), dtype=np.int64)
    return x, (labels == 5).astype(np.float64)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return np.where(value >= 0.0, 1.0 / (1.0 + np.exp(-value)),
                    np.exp(value) / (1.0 + np.exp(value)))


def objective(theta: np.ndarray, x: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    logits = x @ theta[:P] + theta[P]
    value = float(np.mean(np.logaddexp(0.0, logits) - target * logits) + 0.5 * L2 * np.dot(theta, theta))
    residual = sigmoid(logits) - target
    gradient = np.empty(P + 1, dtype=np.float64)
    gradient[:P] = x.T @ residual / N
    gradient[P] = np.mean(residual)
    gradient += L2 * theta
    return value, gradient


def oracle() -> dict[str, np.ndarray | float | int]:
    x, target = fixture()
    scale = np.sqrt(6.0 / (P + 1.0))
    theta = np.array([
        scale * np.sin(SEED + 1009 + 9176 * index) for index in (1, 2)
    ] + [0.01 * scale * np.sin(SEED + 1009 + 7919)], dtype=np.float64)
    initial_loss, gradient = objective(theta, x, target)
    # First Adam update: bias correction exactly reduces to g/(|g|+eps).
    theta = theta - LR * gradient / (np.abs(gradient) + EPS)
    final_loss, final_gradient = objective(theta, x, target)
    logits = x @ theta[:P] + theta[P]
    probabilities = np.column_stack((1.0 - sigmoid(logits), sigmoid(logits)))
    predicted = np.where(logits >= 0.0, 5, -2)
    weights = sigmoid(logits) * (1.0 - sigmoid(logits))
    jacobian = np.column_stack((x, np.ones(N)))
    hessian = (jacobian.T * weights) @ jacobian / N + L2 * np.eye(P + 1)
    direction = 0.01 * np.arange(1, P + 2, dtype=np.float64)
    return {
        "theta": theta, "initial_loss": initial_loss, "final_loss": final_loss,
        "loss": final_loss, "gradient": final_gradient, "hvp": hessian @ direction,
        "scores": logits, "probabilities": probabilities, "predicted": predicted,
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
    with tempfile.TemporaryDirectory(prefix="fortml-mlp-binary-", dir="/mnt/storage") as directory:
        directory_path = Path(directory)
        source = directory_path / fixture_path.name
        executable = directory_path / "mlp_binary_classifier_probe"
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
        if fields and fields[0].startswith("mlp_binary_"):
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
    parser.add_argument("--output", type=Path, default=Path("results/mlp_binary_classifier.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    fixture_path = root / "fixtures" / "mlp_binary_classifier_probe.f90"
    expected = oracle()
    stdout, elapsed = build_probe(fortml, fixture_path)
    observed = parse(stdout)
    required = {"mlp_binary_parameter_count", "mlp_binary_classes_1", "mlp_binary_classes_2",
                "mlp_binary_epochs", "mlp_binary_updates", "mlp_binary_initial_loss",
                "mlp_binary_final_loss", "mlp_binary_loss", "mlp_binary_theta",
                "mlp_binary_gradient", "mlp_binary_hvp", "mlp_binary_score",
                "mlp_binary_probability", "mlp_binary_prediction", "mlp_binary_cuda"}
    if not required.issubset(observed):
        raise RuntimeError(f"probe rows missing: {sorted(required - set(observed))}\n{stdout}")
    if int(observed["mlp_binary_parameter_count"][0][0]) != P + 1:
        raise RuntimeError("unexpected parameter count")
    if [int(observed["mlp_binary_classes_1"][0][0]), int(observed["mlp_binary_classes_2"][0][0])] != [-2, 5]:
        raise RuntimeError("class ordering mismatch")
    if (int(observed["mlp_binary_epochs"][0][0]), int(observed["mlp_binary_updates"][0][0])) != (1, 1):
        raise RuntimeError("Adam accounting mismatch")
    theta = np.array([token(values[1]) for values in sorted(observed["mlp_binary_theta"], key=lambda values: int(values[0]))])
    gradient = np.array([token(values[1]) for values in sorted(observed["mlp_binary_gradient"], key=lambda values: int(values[0]))])
    hvp = np.array([token(values[1]) for values in sorted(observed["mlp_binary_hvp"], key=lambda values: int(values[0]))])
    scores = np.array([token(values[1]) for values in sorted(observed["mlp_binary_score"], key=lambda values: int(values[0]))])
    probabilities = np.array([[token(values[1]), token(values[2])] for values in sorted(observed["mlp_binary_probability"], key=lambda values: int(values[0]))])
    predicted = np.array([int(values[1]) for values in sorted(observed["mlp_binary_prediction"], key=lambda values: int(values[0]))])
    errors = [
        np.max(np.abs(theta - expected["theta"])),
        abs(token(observed["mlp_binary_initial_loss"][0][0]) - expected["initial_loss"]),
        abs(token(observed["mlp_binary_final_loss"][0][0]) - expected["final_loss"]),
        abs(token(observed["mlp_binary_loss"][0][0]) - expected["loss"]),
        np.max(np.abs(gradient - expected["gradient"])),
        np.max(np.abs(hvp - expected["hvp"])),
        np.max(np.abs(scores - expected["scores"])),
        np.max(np.abs(probabilities - expected["probabilities"])),
    ]
    error = float(max(errors))
    if error > 3.0e-11 or not np.array_equal(predicted, expected["predicted"]):
        raise RuntimeError(f"binary MLP oracle mismatch: error={error:.3e}, predicted={predicted}")
    cuda_code = int(observed["mlp_binary_cuda"][0][0])
    if cuda_code != 3:
        raise RuntimeError(f"expected FORTNUM_NOT_IMPLEMENTED=3, got {cuda_code}")
    output = args.output.resolve()
    details = {
        "oracle": "independent NumPy Xavier initializer, BCE/Adam/Hessian oracle",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O2",
    }
    records = [
        row(details, workload="mlp_binary_classifier", phase="fit_predict", backend="fortml",
            device="cpu", status="pass", metric="max_abs_error", value=error,
            max_abs_error=error, seconds=elapsed,
            notes="one full-batch Adam step; layers=2->1; l2=0.02; seed=29"),
        row(details, workload="mlp_binary_classifier", phase="derivatives", backend="fortml",
            device="cpu", status="pass", metric="gradient_hvp_score_probability_error", value=error,
            max_abs_error=error, seconds="", notes="BCE gradient and exact parameter HVP"),
        row(details, workload="mlp_binary_classifier", phase="device_capability", backend="fortml",
            device="cuda", status="unavailable", metric="predict_proba", value="nan",
            max_abs_error="nan", oracle="typed device contract", seconds="",
            notes="FORTNUM_NOT_IMPLEMENTED; resident MLP CUDA graph is not linked"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
