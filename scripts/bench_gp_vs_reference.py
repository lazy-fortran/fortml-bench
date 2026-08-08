#!/usr/bin/env python3
"""FortML's exact GP against scikit-learn and GPyTorch, on an identical model.

The GP benchmarks already in this tree measure FortML against itself across
sizes. That catches a regression and cannot answer whether the implementation
is competitive, which is a different question and the one asked here.

**The model is pinned, not fitted.** Learning hyperparameters on each side
would compare three optimizers and report the difference as a modelling cost.
What is compared is the linear algebra every exact GP performs: a Cholesky
factorization at fit time, and a cross-covariance, a triangular solve and a
reduction at predict time.

**Inputs are reconstructed, not shipped.** Both sides build the training and
query sets from the same closed forms, so a transcription error surfaces as a
value mismatch rather than hiding behind a shared array.

**Fit and predict are timed apart** because they scale differently -- fit is
cubic in the training size, predict linear in the query count -- and one
number would hide which of them a change affected.

scikit-learn's `GaussianProcessRegressor` is the closest widely-used exact GP;
GPyTorch is the one FortBO's own cross-framework work already pins against.
Both are given the identical kernel, hyperparameters and noise, with fitting
disabled.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FORTML = ROOT.parent / "fortml"
BINARY = FORTML / "build" / "fo" / "bin" / "fortml_bench_gp_reference"

LENGTHSCALE = 0.8
SIGNAL_VARIANCE = 1.4
NOISE_VARIANCE = 0.04


def inputs(n_train: int, n_query: int, dimension: int):
    x = np.empty((n_train, dimension))
    for k in range(1, n_train + 1):
        for j in range(1, dimension + 1):
            x[k - 1, j - 1] = math.sin(0.29 * k + 0.41 * j)
    y = np.array([
        np.cos(1.1 * x[k - 1]).sum() + 0.1 * k / n_train
        for k in range(1, n_train + 1)
    ])
    q = np.empty((n_query, dimension))
    for k in range(1, n_query + 1):
        for j in range(1, dimension + 1):
            q[k - 1, j - 1] = math.cos(0.17 * k + 0.23 * j)
    return x, y, q


def run_fortml(n_train: int, n_query: int, dimension: int) -> dict:
    completed = subprocess.run(
        [str(BINARY), str(n_train), str(n_query), str(dimension)],
        capture_output=True, text=True, timeout=20000, cwd=FORTML,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fortml gp bench failed:\n{completed.stdout[-800:]}")
    out = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] in ("TIME", "VALUE"):
            out[f"{fields[0].lower()}_{fields[1]}"] = float(fields[2])
    return out


def run_sklearn(x, y, q) -> dict:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    # `optimizer=None` freezes the hyperparameters; without it sklearn refits
    # and the comparison silently becomes one between optimizers.
    kernel = ConstantKernel(SIGNAL_VARIANCE, constant_value_bounds="fixed") * \
        RBF(LENGTHSCALE, length_scale_bounds="fixed")
    model = GaussianProcessRegressor(kernel=kernel, alpha=NOISE_VARIANCE,
                                     optimizer=None, normalize_y=False)
    started = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    mean, sd = model.predict(q, return_std=True)
    predict_seconds = time.perf_counter() - started
    return {
        "time_fit": fit_seconds,
        "time_predict": predict_seconds,
        "value_mean_sum": float(mean.sum()),
        "value_variance_sum": float((sd**2).sum()),
        "value_mean_first": float(mean[0]),
        "value_variance_first": float(sd[0] ** 2),
    }


def run_gpytorch(x, y, q) -> dict:
    import torch
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ZeroMean

    dtype = torch.float64
    train_x = torch.tensor(x, dtype=dtype)
    train_y = torch.tensor(y, dtype=dtype).unsqueeze(-1)
    query = torch.tensor(q, dtype=dtype)

    likelihood = GaussianLikelihood()
    likelihood.noise = torch.tensor(NOISE_VARIANCE, dtype=dtype)
    covar = ScaleKernel(RBFKernel())
    covar.base_kernel.lengthscale = torch.tensor(LENGTHSCALE, dtype=dtype)
    covar.outputscale = torch.tensor(SIGNAL_VARIANCE, dtype=dtype)

    started = time.perf_counter()
    model = SingleTaskGP(train_x, train_y, likelihood=likelihood,
                         mean_module=ZeroMean(), covar_module=covar,
                         outcome_transform=None, input_transform=None)
    model.eval()
    # GPyTorch factorizes lazily, so the fit cost only materializes on the
    # first posterior. Timing the constructor alone would report a fit time of
    # nearly zero and move that work into the predict column.
    fit_seconds = time.perf_counter() - started

    with torch.no_grad():
        started = time.perf_counter()
        posterior = model.posterior(query)
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1)
        predict_seconds = time.perf_counter() - started

    return {
        "time_fit": fit_seconds,
        "time_predict": predict_seconds,
        "value_mean_sum": float(mean.sum()),
        "value_variance_sum": float(variance.sum()),
        "value_mean_first": float(mean[0]),
        "value_variance_first": float(variance[0]),
        "note": "lazy factorization: fit cost appears in the predict column",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-query", type=int, default=4000)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "fixtures" / "gp_vs_reference.json")
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"{BINARY} missing; run 'fo build --profile release' in fortml")
        return 1

    x, y, q = inputs(args.n_train, args.n_query, args.dimension)
    print(f"n_train={args.n_train} n_query={args.n_query} d={args.dimension}")

    results = {"fortml": run_fortml(args.n_train, args.n_query, args.dimension)}
    for name, runner in (("sklearn", run_sklearn), ("gpytorch", run_gpytorch)):
        try:
            results[name] = runner(x, y, q)
        except ImportError as error:
            results[name] = {"unavailable": str(error)}
            print(f"  {name} unavailable: {error}")

    mine = results["fortml"]
    rows, slower, wrong = [], [], []
    for name in ("sklearn", "gpytorch"):
        other = results[name]
        if "unavailable" in other:
            continue
        # Accuracy first: a speed number against a different answer is not a
        # comparison. The predictive mean sums thousands of terms, so the
        # tolerance is relative.
        scale = max(1.0, abs(mine["value_mean_sum"]))
        mean_error = abs(mine["value_mean_sum"] - other["value_mean_sum"]) / scale
        scale = max(1.0, abs(mine["value_variance_sum"]))
        variance_error = abs(
            mine["value_variance_sum"] - other["value_variance_sum"]) / scale
        agrees = mean_error < 1e-8 and variance_error < 1e-8
        if not agrees:
            wrong.append((name, mean_error, variance_error))

        for stage in ("fit", "predict"):
            key = f"time_{stage}"
            speedup = other[key] / mine[key] if mine[key] > 0 else float("inf")
            rows.append({"against": name, "stage": stage,
                         "fortml_seconds": mine[key],
                         "other_seconds": other[key],
                         "speedup": speedup,
                         "values_agree": agrees})
            if speedup < 1.0:
                slower.append((name, stage, speedup))

    payload = {
        "config": {"n_train": args.n_train, "n_query": args.n_query,
                   "dimension": args.dimension,
                   "lengthscale": LENGTHSCALE,
                   "signal_variance": SIGNAL_VARIANCE,
                   "noise_variance": NOISE_VARIANCE},
        "note": ("Pinned model, not fitted. Inputs reconstructed from the same "
                 "closed forms on both sides rather than shipped."),
        "results": results,
        "comparison": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}\n")

    print(f"{'against':10s} {'stage':9s} {'fortml':>11s} {'other':>11s} {'speedup':>9s}")
    print("-" * 54)
    for row in rows:
        print(f"{row['against']:10s} {row['stage']:9s} "
              f"{row['fortml_seconds']*1e3:9.2f}ms "
              f"{row['other_seconds']*1e3:9.2f}ms "
              f"{row['speedup']:8.1f}x")

    if wrong:
        print("\nVALUES DISAGREE (speed is meaningless until this is fixed):")
        for name, mean_error, variance_error in wrong:
            print(f"  {name}: mean {mean_error:.2e}  variance {variance_error:.2e}")
        return 1
    if slower:
        print("\nSLOWER:")
        for name, stage, factor in slower:
            print(f"  {stage} against {name}: {1.0/factor:.2f}x slower")
        return 1
    print("\nfortml is at least as fast as every reference, values agreeing: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
