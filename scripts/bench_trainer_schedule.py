#!/usr/bin/env python3
"""Correctness-gated benchmark for generic trainer-owned learning-rate schedules.

The scalar schedule and quadratic SGD recurrence below are independent Python
oracles. A release Fortran app must reproduce both before timing is retained.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


RATES = (0.15, 0.2, 0.1125, 0.025)
TOLERANCE = 3.0e-13
FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "seconds", "oracle", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        relative = line[3:].split(" -> ")[-1].strip()
        path = (repository / relative).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def independent_parameters() -> tuple[float, float]:
    parameters = [0.0, 1.0]
    for rate in RATES:
        gradient = [2.0*(parameters[0] - 1.5),
                    4.0*(parameters[1] + 0.5)]
        parameters = [parameters[0] - rate*gradient[0],
                      parameters[1] - rate*gradient[1]]
    return parameters[0], parameters[1]


def row(metadata: dict[str, str], **values: object) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update(values)
    return result


def parse_app(output: str) -> tuple[list[float], tuple[float, float], float]:
    rates: list[float] = []
    parameters: tuple[float, float] | None = None
    seconds = float("nan")
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if not fields:
            continue
        if fields[0] == "trainer_schedule":
            if len(fields) != 7:
                raise RuntimeError(f"unexpected trainer schedule row: {line}")
            rates = [float(item) for item in fields[2:6]]
            seconds = float(fields[6])
        elif fields[0] == "trainer_schedule_parameters":
            if len(fields) != 3:
                raise RuntimeError(f"unexpected trainer parameter row: {line}")
            parameters = (float(fields[1]), float(fields[2]))
    if len(rates) != 4 or parameters is None:
        raise RuntimeError(f"release app omitted schedule rows:\n{output}")
    return rates, parameters, seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/trainer_schedule.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/TRAINER_SCHEDULE.md"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    report = (args.report if args.report.is_absolute() else root / args.report).resolve()
    metadata = {
        "workload": "trainer_schedule",
        "backend": "fortml",
        "device": "cpu",
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    expected_parameters = independent_parameters()
    rows: list[dict[str, object]] = []
    for index, expected in enumerate(RATES, 1):
        rows.append(row(
            metadata, phase="independent_oracle", status="pass",
            metric=f"rate_{index}", value=expected, max_abs_error=0.0,
            oracle="independent one-cycle formula",
            notes="warmup=2,total=4,peak=2,final=0.25",
        ))

    app_status = "skipped"
    app_rates: list[float] = []
    app_parameters = (float("nan"), float("nan"))
    elapsed = float("nan")
    app_output = ""
    environment = os.environ.copy()
    environment.setdefault("FO_SCAN_FALLBACK", "regex")
    if not args.skip_fortml:
        result = subprocess.run(
            ["fo", "exec", "fortml_bench_trainer_schedule"],
            cwd=fortml, env=environment, capture_output=True, text=True,
        )
        app_output = result.stdout + result.stderr
        if result.returncode == 0:
            try:
                app_rates, app_parameters, elapsed = parse_app(result.stdout)
                app_status = "pass"
            except RuntimeError:
                app_status = "failed"
        else:
            app_status = "failed"
    rate_error = max(
        (abs(actual - expected) for actual, expected in zip(app_rates, RATES)),
        default=float("inf"),
    )
    parameter_error = max(
        (abs(actual - expected)
         for actual, expected in zip(app_parameters, expected_parameters)),
        default=float("inf"),
    )
    app_pass = app_status == "pass" and rate_error <= TOLERANCE and parameter_error <= TOLERANCE
    rows.append(row(
        metadata, phase="release_schedule", status="pass" if app_pass else app_status,
        metric="rate_max_abs_error", value=rate_error, max_abs_error=rate_error,
        seconds=elapsed/4.0 if app_pass else "",
        oracle="Fortran trainer vs independent schedule",
        notes="release app emits every optimizer update rate",
    ))
    rows.append(row(
        metadata, phase="release_recurrence", status="pass" if app_pass else app_status,
        metric="parameter_max_abs_error", value=parameter_error,
        max_abs_error=parameter_error, seconds=elapsed/4.0 if app_pass else "",
        oracle="Fortran SGD vs independent quadratic recurrence",
        notes=f"expected_parameters={expected_parameters}",
    ))
    rows.append(row(
        {**metadata, "device": "cuda"}, phase="cuda_typed_refusal",
        status="unavailable", metric="resident_schedule_optimizer",
        value="", max_abs_error=0.0, oracle="typed capability boundary",
        notes="no hidden host fallback; resident CUDA schedule lowering remains open",
    ))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generic trainer learning-rate schedules",
        "",
        "This lane uses an independent one-cycle formula and quadratic SGD",
        "recurrence. The release app is accepted only when all four schedule",
        "rates and both final parameters agree within the stated tolerance.",
        "",
        f"FortML revision: {metadata['fortml_revision']}",
        f"Benchmark revision: {metadata['benchmark_revision']}",
        "",
        "| phase | status | metric | value | max abs error |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for item in rows:
        lines.append(
            f"| {item['phase']} | {item['status']} | {item['metric']} | "
            f"{item['value']} | {item['max_abs_error']} |"
        )
    report.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} rows to {output}; report={report}")
    if not args.skip_fortml and not app_pass:
        print(app_output)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
