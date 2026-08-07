#!/usr/bin/env python3
"""Correctness-gated ARD-RBF kernel and exact-GP benchmark."""

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


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "seconds", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
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


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x1 = np.asarray(
        ((0.0, 1.0, -0.2), (0.5, 1.2, 0.9), (-0.4, -0.7, 0.4)),
        dtype=np.float64,
    )
    x2 = np.asarray(((0.2, 0.8, -0.6), (-0.1, 0.4, 0.3)), dtype=np.float64)
    return x1, x2


def expected_kernel() -> dict[str, np.ndarray | float]:
    x1, x2 = fixture()
    variance = 2.1
    lengths = np.asarray((0.8, 1.2, 1.6), dtype=np.float64)
    inv = 1.0 / lengths**2
    difference = x1[:, None, :] - x2[None, :, :]
    q = difference**2 * inv
    matrix = variance * np.exp(-0.5 * np.sum(q, axis=2))
    x = x1[1]
    z = x2[0]
    d = x - z
    value = float(variance * np.exp(-0.5 * np.sum(d**2 * inv)))
    gradient_x1 = -value * d * inv
    gradient_x2 = -gradient_x1
    mixed = value * (np.diag(inv) - np.outer(d * inv, d * inv))
    direction = np.asarray((0.17, -0.23, 0.11, 0.29), dtype=np.float64)
    matrix_jvp = matrix * (direction[0] + np.sum(q * direction[1:], axis=2))
    # Fortran's reshape fills columns first (the probe uses the same literal).
    matrix_bar = np.asarray(((0.4, 0.5), (-0.2, -0.7), (0.3, 0.1)), dtype=np.float64)
    parameter_vjp = np.empty(4, dtype=np.float64)
    parameter_vjp[0] = np.sum(matrix_bar * matrix)
    parameter_vjp[1:] = np.sum(matrix_bar[:, :, None] * matrix[:, :, None] * q, axis=(0, 1))
    tangent_log_kernel = direction[0] + np.sum(q * direction[1:], axis=2)
    parameter_hvp = np.empty(4, dtype=np.float64)
    parameter_hvp[0] = np.sum(matrix_bar * matrix * tangent_log_kernel)
    parameter_hvp[1:] = np.sum(
        matrix_bar[:, :, None] * matrix[:, :, None]
        * (q * tangent_log_kernel[:, :, None] - 2.0 * q * direction[1:]), axis=(0, 1),
    )
    gp_var, gp_noise = 1.5, 0.2
    gp_lengths = np.asarray((0.8, 1.4, 2.1), dtype=np.float64)
    query = np.asarray((1.0, -0.5, 0.25), dtype=np.float64)
    kstar = gp_var * np.exp(-0.5 * np.sum(query**2 / gp_lengths**2))
    denominator = gp_var + gp_noise
    mean = 2.0 * kstar / denominator
    posterior_variance = gp_var - kstar * kstar / denominator
    lml = -0.5 * 4.0 / denominator - 0.5 * np.log(denominator) - 0.5 * np.log(2.0 * np.pi)
    matrix_bar_gp = 0.5 * ((2.0 / denominator) ** 2 - 1.0 / denominator)
    gp_gradient = np.asarray((matrix_bar_gp * gp_var, 0.0, 0.0, 0.0, matrix_bar_gp * gp_noise))
    return {
        "matrix": matrix, "value": value, "gradient_x1": gradient_x1,
        "gradient_x2": gradient_x2, "mixed": mixed, "matrix_jvp": matrix_jvp,
        "parameter_vjp": parameter_vjp, "parameter_hvp": parameter_hvp,
        "mean": mean, "posterior_variance": posterior_variance, "lml": lml,
        "gp_gradient": gp_gradient,
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
    with tempfile.TemporaryDirectory(prefix="fortml-gp-ard-", dir="/mnt/storage") as directory:
        directory_path = Path(directory)
        source = directory_path / fixture_path.name
        executable = directory_path / "gp_ard_probe"
        source.write_bytes(fixture_path.read_bytes())
        command = compiler + ["-O2", "-ffree-line-length-none", "-I", str(module_dir),
                              str(source), str(archive), "-llapack", "-lblas", "-o", str(executable)]
        link = subprocess.run(command, cwd=fortml, capture_output=True, text=True, check=False)
        if link.returncode:
            raise RuntimeError(link.stderr.strip() or link.stdout.strip())
        started = time.perf_counter()
        run = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        if run.returncode:
            raise RuntimeError(run.stderr.strip() or run.stdout.strip())
        return run.stdout, elapsed


def parse(stdout: str) -> dict[str, list[tuple[int, ...] | float]]:
    rows: dict[str, list[tuple[int, ...] | float]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or not fields[0].startswith("gp_ard_"):
            continue
        key = fields[0]
        if key in {"gp_ard_matrix", "gp_ard_matrix_jvp", "gp_ard_input_mixed"}:
            rows.setdefault(key, []).append((int(fields[1]), int(fields[2]), float(fields[3].replace("D", "E"))))
        elif key in {"gp_ard_input_gradient_x1", "gp_ard_input_gradient_x2",
                     "gp_ard_parameter_vjp", "gp_ard_parameter_hvp", "gp_ard_gp_gradient"}:
            rows.setdefault(key, []).append((int(fields[1]), float(fields[2].replace("D", "E"))))
        elif key == "gp_ard_prediction":
            rows.setdefault(key, []).append((float(fields[1].replace("D", "E")), float(fields[2].replace("D", "E"))))
        elif key in {"gp_ard_input_value", "gp_ard_lml"}:
            rows.setdefault(key, []).append(float(fields[1].replace("D", "E")))
        elif key in {"gp_ard_parameter_count", "gp_ard_cuda"}:
            rows.setdefault(key, []).append(int(fields[1]))
    return rows


def max_error(observed: dict[str, list[tuple[int, ...] | float]], expected: dict[str, np.ndarray | float]) -> float:
    error = 0.0
    matrix = np.zeros((3, 2)); matrix_jvp = np.zeros((3, 2)); mixed = np.zeros((3, 3))
    for i, j, value in observed["gp_ard_matrix"]: matrix[i - 1, j - 1] = value
    for i, j, value in observed["gp_ard_matrix_jvp"]: matrix_jvp[i - 1, j - 1] = value
    for i, j, value in observed["gp_ard_input_mixed"]: mixed[i - 1, j - 1] = value
    for name, shape in (("gp_ard_input_gradient_x1", 3), ("gp_ard_input_gradient_x2", 3),
                        ("gp_ard_parameter_vjp", 4), ("gp_ard_parameter_hvp", 4), ("gp_ard_gp_gradient", 5)):
        values = np.zeros(shape)
        for i, value in observed[name]: values[i - 1] = value
        reference = {"gp_ard_input_gradient_x1": expected["gradient_x1"],
                     "gp_ard_input_gradient_x2": expected["gradient_x2"],
                     "gp_ard_parameter_vjp": expected["parameter_vjp"],
                     "gp_ard_parameter_hvp": expected["parameter_hvp"],
                     "gp_ard_gp_gradient": expected["gp_gradient"]}[name]
        error = max(error, float(np.max(np.abs(values - reference))))
    error = max(error, float(np.max(np.abs(matrix - expected["matrix"]))))
    error = max(error, float(np.max(np.abs(matrix_jvp - expected["matrix_jvp"]))))
    error = max(error, float(np.max(np.abs(mixed - expected["mixed"]))))
    error = max(error, abs(float(observed["gp_ard_input_value"][0]) - float(expected["value"])))
    prediction = observed["gp_ard_prediction"][0]
    error = max(error, abs(float(prediction[0]) - float(expected["mean"])),
                abs(float(prediction[1]) - float(expected["posterior_variance"])),
                abs(float(observed["gp_ard_lml"][0]) - float(expected["lml"])))
    return error


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}; result.update(details); result.update(values); return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/gp_ard.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; fortml = args.fortml.resolve()
    stdout, elapsed = build_probe(fortml, root / "fixtures" / "gp_ard_probe.f90")
    observed = parse(stdout); expected = expected_kernel(); error = max_error(observed, expected)
    if error > 2.0e-11: raise RuntimeError(f"ARD oracle mismatch: {error:.3e}")
    if observed["gp_ard_parameter_count"][0] != 4 or observed["gp_ard_cuda"][0] != 3:
        raise RuntimeError("ARD metadata/device contract mismatch")
    output = args.output.resolve()
    details = {"oracle": "independent NumPy anisotropic RBF, input derivatives, and exact-GP covariance",
               "python_version": platform.python_version(), "numpy_version": np.__version__,
               "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
               "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O2"}
    records = [
        row(details, workload="gp_ard_kernel", phase="value_input_parameter_products", backend="fortml",
            device="cpu", status="pass", metric="max_abs_error", value=error, max_abs_error=error,
            seconds=elapsed, notes="ARD RBF matrix, input gradient/mixed Hessian, JVP/VJP/HVP"),
        row(details, workload="gp_ard_exact_gp", phase="fit_predict_hypergradient", backend="fortml",
            device="cpu", status="pass", metric="max_abs_error", value=error, max_abs_error=error,
            seconds=elapsed, notes="exact GP posterior, LML, and five packed log-parameter gradients"),
        row(details, workload="gp_ard", phase="device_capability", backend="fortml", device="cuda",
            status="unavailable", metric="resident_ard_covariance", value="nan", max_abs_error="nan",
            oracle="typed device contract", notes="FORTNUM_NOT_IMPLEMENTED=3; resident ARD CUDA covariance is not linked"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}; max_abs_error={error:.3e}")


if __name__ == "__main__": main()
