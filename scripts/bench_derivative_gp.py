#!/usr/bin/env python3
"""Correctness-gated derivative-observation GP query-product benchmark.

The NumPy path independently assembles value/first/mixed covariance blocks
from scalar periodic, rational-quadratic, cosine, polynomial, and
spectral-mixture formulas. It then finite-differences
the complete posterior query to obtain an oracle for the exact FortML
third-input products. CUDA is represented as an explicit typed refusal until a
resident derivative-GP graph is available.
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

import numpy as np


N, D, Q, REPETITIONS = 8, 2, 4, 16
NOISE, JITTER = 0.07, 1.0e-10
FIELDS = (
    "workload", "kernel", "operation", "backend", "device", "status",
    "n_samples", "n_features", "n_query", "seconds_per_operation",
    "value", "max_abs_error", "oracle", "python_version", "numpy_version",
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
    dirty = False
    for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines():
        if line[3:].strip() not in ignored_names:
            dirty = True
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty((N, D), dtype=np.float64)
    y = np.empty((N, 1), dtype=np.float64)
    components = np.mod(np.arange(N), 2).astype(np.int64)
    for i in range(N):
        x[i] = (-0.8 + 0.21 * i, 0.3 * np.sin(0.31 * (i + 1)))
        y[i, 0] = 0.6 * np.cos(x[i, 0]) + 0.1 * x[i, 1]
    query = np.empty((Q, D), dtype=np.float64)
    direction = np.empty((Q, D), dtype=np.float64)
    query_components = np.mod(np.arange(1, Q + 1), 2).astype(np.int64)
    mean_bar = np.empty((Q, 1), dtype=np.float64)
    variance_bar = np.empty(Q, dtype=np.float64)
    for i in range(Q):
        query[i] = (-0.53 + 0.27 * i, 0.2 * np.cos(0.37 * (i + 1)))
        direction[i] = (0.07 - 0.01 * (i + 1), -0.04 + 0.013 * (i + 1))
        mean_bar[i, 0] = 0.23 - 0.04 * (i + 1)
        variance_bar[i] = -0.09 + 0.02 * (i + 1)
    return x, y, components, query, direction, query_components, mean_bar, variance_bar


def scalar_kernel(a: np.ndarray, b: np.ndarray, name: str) -> float:
    if name == "polynomial":
        return 1.3 * (1.5 + 0.4 * float(np.dot(a, b))) ** 2.2
    if name == "ard_rbf":
        difference = a - b
        return float(1.3 * np.exp(-0.5 * np.sum(difference**2 / np.asarray((0.75, 1.25))**2)))
    squared = float(np.sum((a - b) ** 2))
    if name == "periodic":
        return 1.3 * np.exp(-2.0 * np.sin(np.pi * np.sqrt(squared) / 2.1) ** 2 / 0.8**2)
    if name == "cosine":
        return 1.3 * np.cos(np.sqrt(squared) / 0.8)
    denominator = 1.0 + squared / (2.0 * 1.7 * 0.8**2)
    return 1.3 * denominator ** (-1.7)


def spectral_covariance(a: np.ndarray, ca: int, b: np.ndarray, cb: int,
                        theta: np.ndarray | None = None) -> float:
    """Independent value/gradient/mixed block for two spectral components."""
    if theta is None:
        theta = np.array([
            np.log(1.15), np.log(0.31), np.log(0.22), 0.21, 0.48,
            np.log(0.63), np.log(0.57), np.log(0.44), -0.37, 0.16,
        ], dtype=np.float64)
    tau = a - b
    value = 0.0
    for q in range(2):
        base = 5 * q
        weight = np.exp(theta[base])
        scale = np.exp(theta[base + 1:base + 3])
        mean = theta[base + 3:base + 5]
        phase = 2.0 * np.pi * tau * mean
        envelope = np.exp(-0.5 * (2.0 * np.pi * tau * scale) ** 2)
        factor = envelope * np.cos(phase)
        lag_first = envelope * (
            -(2.0 * np.pi) ** 2 * tau * scale**2 * np.cos(phase)
            - 2.0 * np.pi * mean * np.sin(phase)
        )
        lag_second = envelope * (
            ((-(2.0 * np.pi) ** 2 * scale**2)
             + (-(2.0 * np.pi) ** 2 * tau * scale**2) ** 2) * np.cos(phase)
            + 2.0 * (-(2.0 * np.pi) ** 2 * tau * scale**2)
            * (-2.0 * np.pi * mean * np.sin(phase))
            - (2.0 * np.pi * mean) ** 2 * np.cos(phase)
        )
        if ca == 0 and cb == 0:
            component = np.prod(factor)
        elif ca > 0 and cb == 0:
            component = lag_first[ca - 1] * np.prod(np.delete(factor, ca - 1))
        elif ca == 0 and cb > 0:
            component = -lag_first[cb - 1] * np.prod(np.delete(factor, cb - 1))
        elif ca == cb:
            component = -lag_second[ca - 1] * np.prod(np.delete(factor, ca - 1))
        else:
            keep = [i for i in range(D) if i not in (ca - 1, cb - 1)]
            component = -lag_first[ca - 1] * lag_first[cb - 1] * np.prod(factor[keep])
        value += weight * component
    return float(value)


def scalar_partials(a: np.ndarray, b: np.ndarray, name: str,
                    theta: np.ndarray | None = None) -> tuple[float, float, float]:
    """Return F(s), F'(s), F''(s), s=||a-b||², independently."""
    squared = float(np.sum((a - b) ** 2))
    if name == "periodic":
        if theta is None:
            variance, lengthscale, period = 1.3, 0.8, 2.1
        else:
            variance, lengthscale, period = np.exp(theta[:3])
        frequency = np.pi / period
        radius = np.sqrt(squared)
        argument = frequency * radius
        if radius <= 1.0e-8:
            t1 = frequency**2
            t2 = -2.0 * frequency**4 / 3.0
        else:
            t1 = frequency * np.sin(2.0 * argument) / (2.0 * radius)
            t2 = frequency * (2.0 * frequency * radius * np.cos(2.0 * argument) -
                              np.sin(2.0 * argument)) / (4.0 * radius**3)
        t = np.sin(argument) ** 2
        value = variance * np.exp(-2.0 * t / lengthscale**2)
        bscale = 2.0 / lengthscale**2
        return value, -bscale * t1 * value, (bscale**2 * t1**2 - bscale * t2) * value
    if name == "cosine":
        if theta is None:
            variance, lengthscale = 1.3, 0.8
        else:
            variance, lengthscale = np.exp(theta[:2])
        radius = np.sqrt(squared)
        if radius <= 1.0e-8:
            return variance, -variance / (2.0 * lengthscale**2), variance / (12.0 * lengthscale**4)
        z = radius / lengthscale
        sine = np.sin(z)
        cosine = np.cos(z)
        value = variance * cosine
        p = -variance * sine / (2.0 * lengthscale * radius)
        p2 = variance * (sine - z * cosine) / (4.0 * lengthscale**4 * z**3)
        return value, p, p2
    if theta is None:
        variance, lengthscale, alpha = 1.3, 0.8, 1.7
    else:
        variance, lengthscale, alpha = np.exp(theta[:3])
    denominator = 1.0 + squared / (2.0 * alpha * lengthscale**2)
    value = variance * denominator ** (-alpha)
    p = -0.5 * value / (lengthscale**2 * denominator)
    p2 = value * (alpha + 1.0) / (4.0 * alpha * lengthscale**4 * denominator**2)
    return value, p, p2


def covariance(a: np.ndarray, ca: int, b: np.ndarray, cb: int, name: str,
               theta: np.ndarray | None = None) -> float:
    """Independent analytic value/gradient/mixed-Hessian covariance blocks."""
    if name == "spectral_mixture":
        return spectral_covariance(a, ca, b, cb, None if theta is None else theta[:10])
    if name == "ard_rbf":
        if theta is None:
            variance, lengths = 1.3, np.asarray((0.75, 1.25), dtype=np.float64)
        else:
            variance, lengths = np.exp(theta[0]), np.exp(theta[1:3])
        difference = a - b
        inverse = 1.0 / lengths**2
        value = variance * np.exp(-0.5 * np.sum(difference**2 * inverse))
        if ca == 0 and cb == 0:
            return float(value)
        if ca > 0 and cb == 0:
            return float(-value * difference[ca - 1] * inverse[ca - 1])
        if ca == 0 and cb > 0:
            return float(value * difference[cb - 1] * inverse[cb - 1])
        return float(value * (inverse[ca - 1] * (ca == cb) -
                             difference[ca - 1] * difference[cb - 1] *
                             inverse[ca - 1] * inverse[cb - 1]))
    if name == "polynomial":
        if theta is None:
            variance, scale, offset, degree = 1.3, 0.4, 1.5, 2.2
        else:
            variance, scale, offset, degree = np.exp(theta[:4])
        base = offset + scale * float(np.dot(a, b))
        value = variance * base ** degree
        coefficient = variance * degree * scale * base ** (degree - 1.0)
        curvature = variance * degree * (degree - 1.0) * scale**2 * base ** (degree - 2.0)
        if ca == 0 and cb == 0:
            return value
        if ca > 0 and cb == 0:
            return float(coefficient * b[ca - 1])
        if ca == 0 and cb > 0:
            return float(coefficient * a[cb - 1])
        return float(coefficient * (1.0 if ca == cb else 0.0) +
                     curvature * b[ca - 1] * a[cb - 1])
    difference = a - b
    value, p, p2 = scalar_partials(a, b, name, theta)
    if ca == 0 and cb == 0:
        return value
    if ca > 0 and cb == 0:
        return float(2.0 * p * difference[ca - 1])
    if ca == 0 and cb > 0:
        return float(-2.0 * p * difference[cb - 1])
    return float(-2.0 * p * (1.0 if ca == cb else 0.0) -
                 4.0 * p2 * difference[ca - 1] * difference[cb - 1])


def predict(x: np.ndarray, components: np.ndarray, y: np.ndarray,
            query: np.ndarray, query_components: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    k_train = np.empty((len(x), len(x)))
    cross = np.empty((len(x), len(query)))
    for i in range(len(x)):
        for j in range(len(x)):
            k_train[i, j] = covariance(x[i], int(components[i]), x[j], int(components[j]), name)
        for j in range(len(query)):
            cross[i, j] = covariance(x[i], int(components[i]), query[j], int(query_components[j]), name)
    k_train.flat[:: len(x) + 1] += NOISE + JITTER
    alpha = np.linalg.solve(k_train, y)
    mean = cross.T @ alpha
    variance = np.empty(len(query))
    for j in range(len(query)):
        prior = covariance(query[j], int(query_components[j]), query[j], int(query_components[j]), name)
        solve_cross = np.linalg.solve(k_train, cross[:, j])
        variance[j] = prior - cross[:, j] @ solve_cross
    return mean, variance


def joint_covariance(x: np.ndarray, components: np.ndarray, y: np.ndarray,
                     query: np.ndarray, query_components: np.ndarray, name: str,
                     theta: np.ndarray | None = None) -> np.ndarray:
    """Independent dense latent posterior covariance oracle."""
    k_train = np.empty((len(x), len(x)))
    cross = np.empty((len(x), len(query)))
    prior = np.empty((len(query), len(query)))
    for i in range(len(x)):
        for j in range(len(x)):
            k_train[i, j] = covariance(x[i], int(components[i]), x[j], int(components[j]), name, theta)
        for j in range(len(query)):
            cross[i, j] = covariance(x[i], int(components[i]), query[j], int(query_components[j]), name, theta)
    for i in range(len(query)):
        for j in range(len(query)):
            prior[i, j] = covariance(query[i], int(query_components[i]), query[j], int(query_components[j]), name, theta)
    noise = NOISE if theta is None else float(np.exp(theta[-1]))
    k_train.flat[:: len(x) + 1] += noise + JITTER
    work = np.linalg.solve(k_train, cross)
    return 0.5 * (prior - cross.T @ work + (prior - cross.T @ work).T)


def likelihood(x: np.ndarray, components: np.ndarray, y: np.ndarray,
               name: str, theta: np.ndarray) -> float:
    """Independent dense log marginal likelihood for the HVP oracle."""
    k_train = np.empty((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            k_train[i, j] = covariance(x[i], int(components[i]), x[j],
                                       int(components[j]), name, theta)
    k_train.flat[:: len(x) + 1] += np.exp(theta[-1]) + JITTER
    alpha = np.linalg.solve(k_train, y)
    sign, logdet = np.linalg.slogdet(k_train)
    if sign <= 0.0:
        raise RuntimeError("independent polynomial likelihood covariance is not SPD")
    return float(-0.5 * np.sum(y * alpha) - 0.5 * y.shape[1] * logdet -
                 0.5 * len(x) * y.shape[1] * np.log(2.0 * np.pi))


def likelihood_gradient(x: np.ndarray, components: np.ndarray, y: np.ndarray,
                        name: str, theta: np.ndarray) -> np.ndarray:
    """Central-difference only the independent likelihood oracle."""
    gradient = np.empty_like(theta)
    step = 2.0e-5
    for i in range(len(theta)):
        direction = np.zeros_like(theta)
        direction[i] = step
        gradient[i] = (likelihood(x, components, y, name, theta + direction) -
                       likelihood(x, components, y, name, theta - direction)) / (2.0 * step)
    return gradient


def derivative_hvp_oracle(name: str) -> float:
    """Return an independent mixed-HVP reduction for a smooth kernel."""
    x, y, components, *_ = fixture()
    values = {
        "polynomial": [1.3, 0.4, 1.5, 2.2],
        "ard_rbf": [1.3, 0.75, 1.25],
    }
    theta = np.log(np.array(values[name] + [NOISE], dtype=np.float64))
    direction = 0.08 - 0.017 * np.arange(1, len(theta) + 1)
    step = 2.0e-4
    product = (likelihood_gradient(x, components, y, name, theta + step * direction) -
               likelihood_gradient(x, components, y, name, theta - step * direction)) / (2.0 * step)
    return float(np.sum(product))


def oracle(name: str) -> tuple[float, float, float, float, float]:
    x, y, components, query, direction, query_components, mean_bar, variance_bar = fixture()
    h = 1.0e-5
    mean_plus, variance_plus = predict(x, components, y, query + h * direction, query_components, name)
    mean_minus, variance_minus = predict(x, components, y, query - h * direction, query_components, name)
    input_jvp = float(np.sum((mean_plus - mean_minus) / (2.0 * h)) +
                      np.sum((variance_plus - variance_minus) / (2.0 * h)))
    x_bar = np.empty_like(query)
    for i in range(Q):
        for j in range(D):
            delta = np.zeros_like(query); delta[i, j] = h
            mean_plus, variance_plus = predict(x, components, y, query + delta, query_components, name)
            mean_minus, variance_minus = predict(x, components, y, query - delta, query_components, name)
            x_bar[i, j] = (float(np.sum(mean_bar * mean_plus) + np.sum(variance_bar * variance_plus)) -
                           float(np.sum(mean_bar * mean_minus) + np.sum(variance_bar * variance_minus))) / (2.0 * h)
    covariance = joint_covariance(x, components, y, query, query_components, name)
    kernel_parameter_count = {"periodic": 3, "rational_quadratic": 3,
                              "cosine": 2, "polynomial": 4,
                              "ard_rbf": 3,
                              "spectral_mixture": 10}[name]
    if name == "spectral_mixture":
        kernel_theta = np.array([
            np.log(1.15), np.log(0.31), np.log(0.22), 0.21, 0.48,
            np.log(0.63), np.log(0.57), np.log(0.44), -0.37, 0.16,
        ], dtype=np.float64)
        theta = np.concatenate([kernel_theta, [np.log(NOISE)]])
    else:
        theta = np.log(np.array({"periodic": [1.3, 0.8, 2.1],
                                 "rational_quadratic": [1.3, 0.8, 1.7],
                                 "cosine": [1.3, 0.8],
                                 "polynomial": [1.3, 0.4, 1.5, 2.2],
                                 "ard_rbf": [1.3, 0.75, 1.25]}[name] + [NOISE]))
    parameter_direction = 0.08 - 0.017 * np.arange(1, kernel_parameter_count + 2)
    covariance_plus = joint_covariance(x, components, y, query, query_components, name,
                                       theta + 1.0e-5 * parameter_direction)
    covariance_minus = joint_covariance(x, components, y, query, query_components, name,
                                        theta - 1.0e-5 * parameter_direction)
    covariance_jvp = (covariance_plus - covariance_minus) / (2.0e-5)
    covariance_bar = np.empty((Q, Q))
    for i in range(Q):
        for j in range(Q):
            covariance_bar[i, j] = 0.15 + 0.02 * (i + 1) - 0.03 * (j + 1)
    covariance_vjp = np.empty(kernel_parameter_count + 1)
    for i in range(kernel_parameter_count + 1):
        unit = np.zeros_like(parameter_direction)
        unit[i] = 1.0e-5
        covariance_plus = joint_covariance(x, components, y, query, query_components, name,
                                           theta + unit)
        covariance_minus = joint_covariance(x, components, y, query, query_components, name,
                                            theta - unit)
        covariance_vjp[i] = np.sum(covariance_bar * (covariance_plus - covariance_minus)) / (2.0e-5)
    return (input_jvp, float(np.sum(x_bar)), float(np.sum(covariance)),
            float(np.sum(covariance_jvp)), float(np.sum(covariance_vjp)))


def read_app(output: str) -> dict[tuple[str, str], tuple[float, float]]:
    records: dict[tuple[str, str], tuple[float, float]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5 and fields[0] == "derivative_gp":
            records[(fields[1], fields[2])] = (float(fields[3]), float(fields[4]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path(__file__).resolve().parents[2] / "fortml")
    parser.add_argument("--output", type=Path, default=Path("results/derivative_gp.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (args.output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    expected = {name: oracle(name) for name in (
        "periodic", "rational_quadratic", "cosine", "polynomial", "spectral_mixture", "ard_rbf")}

    def row(**values: object) -> dict[str, str]:
        result = {field: "" for field in FIELDS}
        result.update({"workload": "derivative_gp", "backend": "fortml", "device": "cpu",
                       "status": "pass", "n_samples": str(N), "n_features": str(D),
                       "n_query": str(Q), "oracle": "independent NumPy analytic covariance and posterior finite-difference oracle",
                       **metadata})
        result.update({key: str(value) for key, value in values.items()})
        return result

    rows: list[dict[str, str]] = []
    for name, values in expected.items():
        rows.append(row(kernel=name, operation="input_jvp", backend="numpy_oracle",
                        seconds_per_operation="", value=values[0], max_abs_error="0.0"))
        rows.append(row(kernel=name, operation="input_vjp", backend="numpy_oracle",
                        seconds_per_operation="", value=values[1], max_abs_error="0.0"))
        rows.append(row(kernel=name, operation="joint_covariance", backend="numpy_oracle",
                        seconds_per_operation="", value=values[2], max_abs_error="0.0"))
        rows.append(row(kernel=name, operation="joint_covariance_jvp", backend="numpy_oracle",
                        seconds_per_operation="", value=values[3], max_abs_error="0.0"))
        rows.append(row(kernel=name, operation="joint_covariance_vjp", backend="numpy_oracle",
                        seconds_per_operation="", value=values[4], max_abs_error="0.0"))

    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(dir=fortml / "build"):
        completed = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_derivative_gp"],
                                   cwd=fortml, env=environment, capture_output=True, text=True, check=True)
    records = read_app(completed.stdout)
    for name, values in expected.items():
        for operation, expected_value in zip(("input_jvp", "input_vjp", "joint_covariance",
                                              "joint_covariance_jvp", "joint_covariance_vjp"), values):
            if (name, operation) not in records:
                raise RuntimeError(f"release app omitted {name}/{operation}")
            seconds, actual = records[(name, operation)]
            error = abs(actual - expected_value)
            if error > 5.0e-4 * max(1.0, abs(expected_value)):
                raise RuntimeError(f"{name}/{operation} oracle mismatch: {error:.3e}")
            rows.append(row(kernel=name, operation=operation, seconds_per_operation=seconds,
                            value=actual, max_abs_error=error, notes="exact CPU product; independent oracle passed"))
            rows.append(row(kernel=name, operation=operation, device="cuda", status="unavailable",
                            backend="fortml", seconds_per_operation="", value="", max_abs_error="",
                            oracle="typed_device_contract", notes="FORTNUM_NOT_IMPLEMENTED: resident derivative-GP graph not linked"))

    for name in ("polynomial", "ard_rbf"):
        expected_hvp = derivative_hvp_oracle(name)
        if (name, "hyperparameter_hvp") not in records:
            raise RuntimeError(f"release app omitted {name}/hyperparameter_hvp")
        seconds, actual = records[(name, "hyperparameter_hvp")]
        error = abs(actual - expected_hvp)
        if error > 5.0e-4 * max(1.0, abs(expected_hvp)):
            raise RuntimeError(f"{name}/hyperparameter_hvp oracle mismatch: {error:.3e}")
        rows.append(row(kernel=name, operation="hyperparameter_hvp",
                        seconds_per_operation=seconds, value=actual, max_abs_error=error,
                        notes="exact mixed-observation HVP; independent likelihood finite-difference oracle passed"))
        rows.append(row(kernel=name, operation="hyperparameter_hvp", device="cuda",
                        status="unavailable", backend="fortml", seconds_per_operation="",
                        value="", max_abs_error="", oracle="typed_device_contract",
                        notes="FORTNUM_NOT_IMPLEMENTED: resident derivative-GP graph not linked"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
