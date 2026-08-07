#!/usr/bin/env python3
"""Correctness-gated exact-GP constant and linear mean benchmark."""

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


N, Q = 4, 2
VARIANCE, LENGTHSCALE, NOISE, JITTER = 1.3, 0.8, 0.15, 1.0e-10
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray((-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0), dtype=np.float64)[:, None]
    y = (0.3 + 0.2 * x[:, 0] + 0.05 * x[:, 0] ** 2)[:, None]
    query = np.asarray((-0.5, 1.5), dtype=np.float64)[:, None]
    return x, y, query


def model_oracle(kind: str) -> dict[str, np.ndarray | float]:
    x, y, query = fixture()
    basis = np.column_stack((np.ones(N),)) if kind == "constant" else np.column_stack((np.ones(N), x[:, 0]))
    query_basis = np.column_stack((np.ones(Q),)) if kind == "constant" else np.column_stack((np.ones(Q), query[:, 0]))
    coefficients = np.asarray((0.2,), dtype=np.float64) if kind == "constant" else np.asarray((0.1, 0.5), dtype=np.float64)
    direction = np.zeros(3 + coefficients.size, dtype=np.float64)
    direction[3:] = (0.07,) if kind == "constant" else (0.02, -0.03)

    def evaluate(parameters: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        variance_value, lengthscale_value, noise_value = np.exp(parameters[:3])
        mean_coefficients = parameters[3:]
        kernel_value = variance_value * np.exp(
            -((x[:, None, 0] - x[None, :, 0]) ** 2) / (2.0 * lengthscale_value**2)
        )
        covariance_value = kernel_value + (noise_value + JITTER) * np.eye(N)
        residual_value = y[:, 0] - basis @ mean_coefficients
        alpha_value = np.linalg.solve(covariance_value, residual_value)
        sign_value, logdet_value = np.linalg.slogdet(covariance_value)
        if sign_value <= 0.0:
            raise RuntimeError("GP covariance is not positive definite")
        lml_value = float(-0.5 * residual_value @ alpha_value - 0.5 * logdet_value -
                          0.5 * N * np.log(2.0 * np.pi))
        inverse_value = np.linalg.inv(covariance_value)
        matrix_bar_value = 0.5 * (np.outer(alpha_value, alpha_value) - inverse_value)
        r2_value = (x[:, None, 0] - x[None, :, 0]) ** 2
        gradient_value = np.empty(3 + mean_coefficients.size, dtype=np.float64)
        gradient_value[0] = np.sum(matrix_bar_value * kernel_value)
        gradient_value[1] = np.sum(matrix_bar_value * kernel_value * r2_value / lengthscale_value**2)
        gradient_value[2] = noise_value * np.trace(matrix_bar_value)
        gradient_value[3:] = basis.T @ alpha_value
        return lml_value, gradient_value, covariance_value, alpha_value

    parameters = np.concatenate((np.log((VARIANCE, LENGTHSCALE, NOISE)), coefficients))
    lml, gradient, covariance, alpha = evaluate(parameters)
    cross = VARIANCE * np.exp(-((x[:, None, 0] - query[None, :, 0]) ** 2) / (2.0 * LENGTHSCALE**2))
    mean = query_basis @ coefficients + cross.T @ alpha
    variance = VARIANCE - np.sum(cross * np.linalg.solve(covariance, cross), axis=0)
    step = 1.0e-5
    hvp = (evaluate(parameters + step * direction)[1] -
           evaluate(parameters - step * direction)[1]) / (2.0 * step)
    return {"parameters": parameters, "mean": mean, "variance": variance, "lml": lml,
            "gradient": gradient, "hvp": hvp, "parameter_count": parameters.size,
            "mean_parameter_count": coefficients.size}


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
    with tempfile.TemporaryDirectory(prefix="fortml-gp-mean-", dir="/mnt/storage") as directory:
        directory_path = Path(directory)
        source = directory_path / fixture_path.name
        executable = directory_path / "gp_mean_probe"
        source.write_bytes(fixture_path.read_bytes())
        command = compiler + ["-O2", "-ffree-line-length-none", "-I", str(module_dir),
                              str(source), str(archive), "-llapack", "-lblas",
                              "-o", str(executable)]
        link = subprocess.run(command, cwd=fortml, capture_output=True, text=True, check=False)
        if link.returncode:
            raise RuntimeError(link.stderr.strip() or link.stdout.strip())
        started = time.perf_counter()
        run = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        if run.returncode:
            raise RuntimeError(run.stderr.strip() or run.stdout.strip())
        return run.stdout, elapsed


def token(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def parse(stdout: str) -> dict[str, list[list[str]]]:
    rows: dict[str, list[list[str]]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0].startswith("gp_mean_"):
            rows.setdefault(fields[0], []).append(fields[1:])
    return rows


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/gp_mean.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    fixture_path = root / "fixtures" / "gp_mean_probe.f90"
    stdout, elapsed = build_probe(fortml, fixture_path)
    observed = parse(stdout)
    expected = {kind: model_oracle(kind) for kind in ("constant", "linear")}
    errors: dict[str, float] = {}
    for kind, oracle in expected.items():
        prefix = f"gp_mean_{kind}_"
        count = int(observed[prefix + "parameter_count"][0][0])
        mean_count = int(observed[prefix + "mean_parameter_count"][0][0])
        if (count, mean_count) != (oracle["parameter_count"], oracle["mean_parameter_count"]):
            raise RuntimeError(f"{kind} mean packing mismatch: {(count, mean_count)}")
        parameters = np.array([token(values[1]) for values in sorted(observed[prefix + "parameter"], key=lambda values: int(values[0]))])
        gradient = np.array([token(values[1]) for values in sorted(observed[prefix + "gradient"], key=lambda values: int(values[0]))])
        hvp = np.array([token(values[1]) for values in sorted(observed[prefix + "hvp"], key=lambda values: int(values[0]))])
        prediction = np.array([[token(values[1]), token(values[2])] for values in sorted(observed[prefix + "prediction"], key=lambda values: int(values[0]))])
        error = max(
            np.max(np.abs(parameters - oracle["parameters"])),
            abs(token(observed[prefix + "lml"][0][0]) - oracle["lml"]),
            np.max(np.abs(gradient - oracle["gradient"])),
            np.max(np.abs(hvp - oracle["hvp"])),
            np.max(np.abs(prediction[:, 0] - oracle["mean"])),
            np.max(np.abs(prediction[:, 1] - oracle["variance"])),
        )
        errors[kind] = float(error)
        if error > 4.0e-10:
            raise RuntimeError(f"{kind} GP mean oracle mismatch: {error:.3e}")
    cuda = int(observed["gp_mean_cuda"][0][0])
    if cuda != 3:
        raise RuntimeError(f"expected typed CUDA refusal code 3, got {cuda}")
    output = args.output.resolve()
    details = {
        "oracle": "independent NumPy RBF covariance, Cholesky-equivalent solve, and mean derivatives",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O2",
    }
    records = []
    for kind in ("constant", "linear"):
        records.append(row(details, workload=f"gp_{kind}_mean", phase="fit_predict_derivatives",
                            backend="fortml", device="cpu", status="pass", metric="max_abs_error",
                            value=errors[kind], max_abs_error=errors[kind], seconds=elapsed,
                            notes="trainable mean coefficients included in packed GP parameters"))
    records.append(row(details, workload="gp_mean", phase="device_capability", backend="fortml",
                       device="cuda", status="unavailable", metric="exact_gp_mean", value="nan",
                       max_abs_error="nan", oracle="typed device contract", seconds="",
                       notes="FORTNUM_NOT_IMPLEMENTED=3; exact GP factorization has no resident CUDA path"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
