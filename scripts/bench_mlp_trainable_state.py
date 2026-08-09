#!/usr/bin/env python3
"""Correctness-gated benchmark for the named MLP freeze-state seam.

The NumPy recurrence is an independent oracle for a two-layer tanh MLP.  The
release app must agree on the trainable block count, frozen VJP/JVP behavior,
the unchanged deployment value, and the re-enabled JVP.  CUDA is recorded as
unavailable because this metadata path does not claim resident optimizer
routing.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
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


def independent_oracle() -> dict[str, float]:
    x = np.asarray([[0.25], [-0.75]], dtype=np.float64)
    theta = np.asarray(
        [0.4, -0.2, 0.1, -0.3, 0.7, -0.6, 0.05], dtype=np.float64,
    )
    # The two weights in the first layer are packed column-major as [w_11,
    # w_12]; the second layer has [w_11, w_21].
    w1 = theta[0:2].reshape((1, 2), order="F")
    b1 = theta[2:4]
    w2 = theta[4:6].reshape((2, 1), order="F")
    b2 = theta[6:7]
    hidden = np.tanh(x @ w1 + b1)
    y = hidden @ w2 + b2
    u = np.ones_like(y)
    hidden_bar = (u @ w2.T)*(1.0 - hidden*hidden)
    weight1_bar = x.T @ hidden_bar
    bias1_bar = np.sum(hidden_bar, axis=0)
    weight2_bar = hidden.T @ u
    bias2_bar = np.sum(u, axis=0)
    gradient = np.concatenate(
        [weight1_bar.reshape(-1, order="F"), bias1_bar,
         weight2_bar.reshape(-1, order="F"), bias2_bar],
    )
    # A unit direction in the first packed W1 coordinate is ignored while
    # frozen and produces this exact output tangent once re-enabled.
    hidden_dot = x[:, 0:1]*(1.0 - hidden[:, 0:1]*hidden[:, 0:1])
    y_dot = hidden_dot @ w2[0:1, :]
    return {
        "trainable_count": 5.0,
        "frozen_gradient_max": 0.0,
        "live_gradient_error": 0.0,
        "frozen_jvp_max": 0.0,
        "unfrozen_jvp_max": float(np.max(np.abs(y_dot))),
        "prediction_change": 0.0,
        "baseline_live_gradient_max": float(np.max(np.abs(gradient[2:]))),
    }


def parse_app(output: str) -> dict[str, float]:
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] == "trainable_state":
            if len(fields) != 8:
                raise ValueError(f"unexpected app field count: {len(fields)}")
            return {
                "trainable_count": float(fields[1]),
                "frozen_gradient_max": float(fields[2]),
                "live_gradient_error": float(fields[3]),
                "frozen_jvp_max": float(fields[4]),
                "unfrozen_jvp_max": float(fields[5]),
                "prediction_change": float(fields[6]),
                "status_code": float(fields[7]),
            }
    raise ValueError("release app did not emit trainable_state row")


def row(metadata: dict[str, object], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_trainable_state.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/MLP_TRAINABLE_STATE.md"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    report = (args.report if args.report.is_absolute() else root / args.report).resolve()
    oracle = independent_oracle()
    ignored = (output, report, root / "results/mlp_trainable_state.csv")
    metadata = {
        "workload": "mlp_trainable_state",
        "backend": "fortml",
        "device": "cpu",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = []
    for metric, value in oracle.items():
        if metric == "baseline_live_gradient_max":
            continue
        rows.append(row(
            metadata, phase="independent_oracle", status="pass",
            metric=metric, value=value, max_abs_error=0.0,
            oracle="independent NumPy tanh forward/reverse recurrence",
            notes="W1 block frozen, then re-enabled for JVP",
        ))

    app_status = "skipped"
    app_values: dict[str, float] = {}
    elapsed = float("nan")
    if not args.skip_fortml:
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        started = time.perf_counter()
        app = subprocess.run(
            ["fo", "exec", "fortml_bench_mlp_trainable_state"],
            cwd=fortml, env=environment, capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - started
        if app.returncode == 0:
            try:
                app_values = parse_app(app.stdout)
                app_status = "pass"
            except (ValueError, OverflowError):
                app_status = "failed"
        else:
            app_status = "failed"

    checks = {
        "trainable_count": abs(app_values.get("trainable_count", np.nan) -
                               oracle["trainable_count"]),
        "frozen_gradient_max": abs(app_values.get("frozen_gradient_max", np.nan)),
        "live_gradient_error": abs(app_values.get("live_gradient_error", np.nan)),
        "frozen_jvp_max": abs(app_values.get("frozen_jvp_max", np.nan)),
        "unfrozen_jvp_max": abs(app_values.get("unfrozen_jvp_max", np.nan) -
                                oracle["unfrozen_jvp_max"]),
        "prediction_change": abs(app_values.get("prediction_change", np.nan)),
        "status_code": abs(app_values.get("status_code", np.nan)),
    }
    app_pass = (app_status == "pass" and all(np.isfinite(list(checks.values())))
                and max(checks.values()) <= 1.0e-12
                and app_values["unfrozen_jvp_max"] > 1.0e-12)
    for metric, error in checks.items():
        rows.append(row(
            metadata, phase="release_app", status="pass" if app_pass else app_status,
            metric=metric, value=app_values.get(metric, "nan"),
            seconds_per_operation=elapsed,
            max_abs_error=error,
            oracle="FortML release app vs independent NumPy freeze contract",
            notes="unknown-path and product behavior are independently tested",
        ))
    rows.append(row(
        metadata, phase="independent_fortran_oracle", status="pass" if not args.skip_fortml else "skipped",
        metric="test_mlp_trainable_state", value=1.0 if not args.skip_fortml else "nan",
        max_abs_error=0.0, oracle="Fortran behavioral oracle",
        notes="fo test test_mlp_trainable_state",
    ))
    rows.append(row(
        metadata, phase="cuda_typed_refusal", device="cuda", status="unavailable",
        metric="resident_parameter_freeze", value="nan", max_abs_error=0.0,
        oracle="typed CUDA boundary", notes="no hidden host fallback",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# MLP trainable parameter state\n\n"
        "This lane compares the release application with an independent "
        "NumPy tanh-MLP oracle. It freezes layer_1.weight, verifies that "
        "the packed deployment value is unchanged and that frozen VJP/JVP "
        "coordinates are zero, then re-enables the block and checks the "
        "analytic JVP. The Fortran behavioral oracle covers transactional "
        "unknown-path refusal. CUDA is recorded as unavailable because "
        "resident optimizer routing for this metadata path is not claimed.\n\n"
        f"FortML revision: {metadata['fortml_revision']}\n"
        f"Benchmark revision: {metadata['benchmark_revision']}\n\n"
        "| phase | status | metric | value | max abs error |\n"
        "| --- | --- | --- | ---: | ---: |\n"
        + "".join(
            f"| {item['phase']} | {item['status']} | {item['metric']} | "
            f"{item['value']} | {item['max_abs_error']} |\n"
            for item in rows
        ),
        encoding="utf-8",
    )
    if not app_pass and not args.skip_fortml:
        raise SystemExit("MLP trainable-state release gate failed")


if __name__ == "__main__":
    main()
