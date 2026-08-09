#!/usr/bin/env python3
"""Correctness-gated benchmark for exact GP posterior covariance."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "seconds_per_operation", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = (-1.2 + 0.08*np.arange(32, dtype=np.float64))[:, None]
    y = np.column_stack((np.sin(1.1*x[:, 0]), np.cos(0.8*x[:, 0]) - 0.25))
    query = (-1.0 + 0.13*np.arange(16, dtype=np.float64))[:, None]
    return x, y, query


def covariance_for_parameters(parameters: np.ndarray) -> np.ndarray:
    x, _, query = fixture()
    signal, lengthscale, noise = np.exp(parameters)
    jitter = 1.0e-10
    train = signal*np.exp(-0.5*(x - x.T)**2/lengthscale**2)
    train += (noise + jitter)*np.eye(x.shape[0])
    cross = signal*np.exp(-0.5*(x - query.T)**2/lengthscale**2)
    prior = signal*np.exp(-0.5*(query - query.T)**2/lengthscale**2)
    solved = np.linalg.solve(train, cross)
    covariance = prior - cross.T @ solved
    covariance = 0.5*(covariance + covariance.T)
    return covariance


def oracle() -> dict[str, object]:
    parameters = np.log([1.2, 0.55, 0.08])
    direction = np.array([0.13, -0.19, 0.23], dtype=np.float64)
    x, _, query = fixture()
    signal, lengthscale, noise = np.exp(parameters)
    jitter = 1.0e-10
    train = signal*np.exp(-0.5*(x - x.T)**2/lengthscale**2)
    train += (noise + jitter)*np.eye(x.shape[0])
    cross = signal*np.exp(-0.5*(x - query.T)**2/lengthscale**2)
    prior = signal*np.exp(-0.5*(query - query.T)**2/lengthscale**2)
    train_dot = (train - (noise + jitter)*np.eye(x.shape[0])) * (
        direction[0] + (x - x.T)**2/lengthscale**2*direction[1])
    train_dot += noise*direction[2]*np.eye(x.shape[0])
    cross_dot = cross*(direction[0] + (x - query.T)**2/lengthscale**2*direction[1])
    prior_dot = prior*(direction[0] + (query - query.T)**2/lengthscale**2*direction[1])
    solved = np.linalg.solve(train, cross)
    solved_dot = np.linalg.solve(train, cross_dot - train_dot @ solved)
    covariance = prior - cross.T @ solved
    covariance = 0.5*(covariance + covariance.T)
    covariance_dot = prior_dot - cross_dot.T @ solved - cross.T @ solved_dot
    covariance_dot = 0.5*(covariance_dot + covariance_dot.T)
    covariance_bar = np.resize(
        np.array([0.4, -0.2, 0.3, 0.5, -0.7, 0.1, 0.6, -0.4, 0.2],
                 dtype=np.float64), query.shape[0]*query.shape[0]
    ).reshape((query.shape[0], query.shape[0]), order="F")
    h = 1.0e-6
    vjp = np.array([
        np.sum(covariance_bar * (covariance_for_parameters(parameters + h*np.eye(3)[i]) -
                                 covariance_for_parameters(parameters - h*np.eye(3)[i]))/(2*h))
        for i in range(3)
    ])
    return {
        "covariance_checksum": float(np.sum(covariance)),
        "variance_checksum": float(np.trace(covariance)),
        "symmetry_max": float(np.max(np.abs(covariance - covariance.T))),
        "covariance_dot_checksum": float(np.sum(covariance_dot)),
        "parameter_bar_checksum": float(np.sum(vjp)),
    }


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True,
    ).splitlines():
        relative = line[3:].split(" -> ")[-1].strip()
        if (repository / relative).resolve() not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def row(metadata: dict[str, object], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update(values)
    return result


def parse_app(stdout: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields[:2] == ["gp_posterior_covariance", "seconds"]:
            values["seconds"] = float(fields[2])
            values["covariance_checksum"] = float(fields[4])
        elif fields[:2] == ["gp_posterior_covariance", "variance_checksum"]:
            values["variance_checksum"] = float(fields[2])
        elif fields[:2] == ["gp_posterior_covariance_jvp", "seconds"]:
            values["jvp_seconds"] = float(fields[2])
            values["covariance_dot_checksum"] = float(fields[4])
        elif fields[:2] == ["gp_posterior_covariance_vjp", "parameter_bar_checksum"]:
            values["parameter_bar_checksum"] = float(fields[2])
        elif fields[:3] == ["gp_posterior_covariance_device", "cpu", "supported"]:
            values["cpu_supported"] = fields[3]
        elif fields[:3] == ["gp_posterior_covariance_device", "cuda", "refused"]:
            values["cuda_code"] = int(fields[3])
        elif fields[:3] == ["gp_posterior_covariance_jvp_device", "cpu", "supported"]:
            values["jvp_cpu_supported"] = fields[3]
        elif fields[:3] == ["gp_posterior_covariance_jvp_device", "cuda", "refused"]:
            values["jvp_cuda_code"] = int(fields[3])
        elif fields[:3] == ["gp_posterior_covariance_vjp_device", "cpu", "supported"]:
            values["vjp_cpu_supported"] = fields[3]
        elif fields[:3] == ["gp_posterior_covariance_vjp_device", "cuda", "refused"]:
            values["vjp_cuda_code"] = int(fields[3])
    required = {"seconds", "covariance_checksum", "variance_checksum",
                "cpu_supported", "cuda_code", "jvp_seconds",
                "covariance_dot_checksum", "parameter_bar_checksum",
                "jvp_cpu_supported", "jvp_cuda_code", "vjp_cpu_supported",
                "vjp_cuda_code"}
    if not required.issubset(values):
        raise RuntimeError(f"release app omitted covariance rows: {stdout}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_posterior_covariance.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/GP_POSTERIOR_COVARIANCE.md"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    report = (args.report if args.report.is_absolute() else root / args.report).resolve()
    expected = oracle()
    metadata = {
        "workload": "gp_posterior_covariance", "backend": "fortml", "device": "cpu",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = []
    for metric, value in expected.items():
        rows.append(row(metadata, phase="independent_oracle", status="pass",
                        metric=metric, value=value, max_abs_error=0.0,
                        oracle="independent NumPy dense Cholesky-equivalent solve"))
    if args.skip_fortml:
        rows.append(row(metadata, phase="behavioral_gate", status="skipped",
                        metric="tests_passed", value="nan", max_abs_error="nan",
                        oracle="test_gp_posterior_covariance", notes="--skip-fortml"))
    else:
        environment = os.environ.copy()
        environment.update({"FO_SCAN_FALLBACK": "regex", "OMP_NUM_THREADS": "1"})
        subprocess.run(["fo", "test", "test_gp_posterior_covariance"], cwd=fortml,
                       env=environment, check=True)
        rows.append(row(metadata, phase="behavioral_gate", status="pass",
                        metric="tests_passed", value=1.0, max_abs_error=0.0,
                        oracle="independent Fortran dense posterior covariance oracle"))
        started = time.perf_counter()
        app = subprocess.run(["fo", "exec", "fortml_bench_gp_posterior_covariance"],
                             cwd=fortml, env=environment, check=True,
                             capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        values = parse_app(app.stdout)
        for metric in ("covariance_checksum", "variance_checksum"):
            error = abs(float(values[metric]) - float(expected[metric]))
            if error > 3.0e-10:
                raise RuntimeError(f"{metric} checksum mismatch: {error:.3e}")
            rows.append(row(metadata, phase="release_app", status="pass", metric=metric,
                            value=values[metric], seconds_per_operation=values["seconds"],
                            max_abs_error=error, oracle="independent NumPy posterior covariance"))
        for metric in ("covariance_dot_checksum", "parameter_bar_checksum"):
            error = abs(float(values[metric]) - float(expected[metric]))
            if error > 3.0e-7:
                raise RuntimeError(f"{metric} checksum mismatch: {error:.3e}")
            rows.append(row(metadata, phase="release_app", status="pass", metric=metric,
                            value=values[metric],
                            seconds_per_operation=values["jvp_seconds"],
                            max_abs_error=error,
                            oracle="independent NumPy covariance hyperparameter product"))
        rows.append(row(metadata, phase="release_app", status="pass", metric="wall_seconds",
                        value=elapsed, max_abs_error=0.0,
                        oracle="fortml release app timing"))
        if values["cpu_supported"] != "T" or values["cuda_code"] != 3:
            raise RuntimeError(f"unexpected device boundary: {values}")
        rows.append(row(metadata, phase="device_boundary", backend="fortml", device="cuda",
                        status="refused", metric="posterior_covariance", value="nan",
                        max_abs_error=0.0, oracle="FORTNUM_NOT_IMPLEMENTED",
                        notes="resident CUDA covariance and Cholesky are not linked"))
        if values["jvp_cpu_supported"] != "T" or values["jvp_cuda_code"] != 3:
            raise RuntimeError(f"unexpected JVP device boundary: {values}")
        rows.append(row(metadata, phase="device_boundary", backend="fortml", device="cuda",
                        status="refused", metric="posterior_covariance_jvp", value="nan",
                        max_abs_error=0.0, oracle="FORTNUM_NOT_IMPLEMENTED",
                        notes="resident CUDA covariance derivative kernel is not linked"))
        if values["vjp_cpu_supported"] != "T" or values["vjp_cuda_code"] != 3:
            raise RuntimeError(f"unexpected VJP device boundary: {values}")
        rows.append(row(metadata, phase="device_boundary", backend="fortml", device="cuda",
                        status="refused", metric="posterior_covariance_vjp", value="nan",
                        max_abs_error=0.0, oracle="FORTNUM_NOT_IMPLEMENTED",
                        notes="resident CUDA covariance derivative kernel is not linked"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Exact GP posterior covariance\n\n"
        "This lane compares `gp_regression_t%predict_covariance` with an "
        "independent NumPy dense solve for a shared RBF query set. The full "
        "latent posterior matrix is checked for checksum, symmetry, and a "
        "diagonal match with the marginal variance path. CPU dispatch is the "
        "reference, and selected CUDA returns `FORTNUM_NOT_IMPLEMENTED` without "
        "claiming a host fallback. The same lane checks the full-matrix "
        "hyperparameter JVP and VJP against independent NumPy solve-state "
        "products; covariance derivative device calls have the same explicit "
        "CUDA refusal.\n\n"
        f"Oracle covariance checksum: `{expected['covariance_checksum']:.16g}`, "
        f"variance checksum: `{expected['variance_checksum']:.16g}`.\n\n"
        "Reproduce with:\n\n"
        "```text\n"
        "python scripts/bench_gp_posterior_covariance.py --fortml ../fortml "
        "--output results/gp_posterior_covariance.csv "
        "--report results/GP_POSTERIOR_COVARIANCE.md\n"
        "```\n\n"
        f"Pinned source revision: `{metadata['fortml_revision']}`.\n"
        f"Pinned benchmark revision: `{metadata['benchmark_revision']}`.\n"
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
