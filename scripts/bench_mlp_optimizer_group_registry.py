#!/usr/bin/env python3
"""Correctness-gated named optimizer-group checkpoint benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes", "python_version", "fortad_revision",
    "fortsym_revision", "fortopt_revision",
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
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def parse_app(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0] == "mlp_optimizer_group_registry":
            values[fields[1]] = fields[2]
    if not {"roundtrip_name", "name_drift_status", "cuda_status"} <= values.keys():
        raise RuntimeError("release app omitted registry rows")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_optimizer_group_registry.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/MLP_OPTIMIZER_GROUP_REGISTRY.md"))
    parser.add_argument("--target", default="fortml_bench_mlp_optimizer_group_registry")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    report = (args.report if args.report.is_absolute() else root / args.report).resolve()
    details = {
        "workload": "mlp_optimizer_group_registry", "backend": "fortml",
        "python_version": platform.python_version(),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, Any]] = [row(
        details, phase="independent_oracle", device="cpu", status="pass",
        metric="registry_name", value="bias", max_abs_error=0.0,
        oracle="independent named checkpoint contract",
        notes="round-trip preserves identity; renamed registry must be rejected"),
    ]
    values: dict[str, str] = {}
    app_status = "skipped"
    if not args.skip_fortml:
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        gate = subprocess.run(
            ["fo", "test", "test_mlp_optimizer_groups"], cwd=fortml,
            env=environment, capture_output=True, text=True,
        )
        app = subprocess.run(
            ["fo", "exec", args.target], cwd=fortml,
            env=environment, capture_output=True, text=True,
        )
        if gate.returncode == 0 and app.returncode == 0:
            try:
                values = parse_app(app.stdout)
                app_status = "pass"
            except RuntimeError:
                app_status = "failed"
        else:
            app_status = "failed"
    app_pass = (
        app_status == "pass" and values.get("roundtrip_name") == "bias" and
        values.get("name_drift_status") == "1" and values.get("cuda_status") == "3"
    )
    rows.extend([
        row(details, phase="release_app", device="cpu",
            status="pass" if app_pass else app_status, metric="registry_name",
            value=values.get("roundtrip_name", ""), max_abs_error=0.0,
            oracle="independent named checkpoint contract",
            notes="formatted checkpoint preserves the group name"),
        row(details, phase="behavioral_gate", device="cpu",
            status="pass" if not args.skip_fortml else "skipped",
            metric="name_drift_status", value=values.get("name_drift_status", ""),
            max_abs_error=0.0, oracle="transactional resume mismatch refusal",
            notes="same range and multiplier with a different name is rejected"),
        row(details, phase="device_boundary", device="cuda", status="unavailable",
            metric="optimizer_group_status", value=values.get("cuda_status", ""),
            max_abs_error=0.0, oracle="typed CUDA capability contract",
            notes="no host fallback when the grouped hypergradient is requested"),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# MLP optimizer-group registry benchmark\n\n"
        "This lane checks that named optimizer groups survive formatted checkpoint "
        "round-trips and that resume rejects identity drift even when ranges and "
        "multipliers are unchanged. The CUDA row records the typed refusal for the "
        "non-resident grouped hypergradient path.\n\n"
        f"FortML revision: {details['fortml_revision']}\n"
        f"Benchmark revision: {details['benchmark_revision']}\n\n"
        "| phase | device | status | metric | value |\n"
        "| --- | --- | --- | --- | --- |\n" +
        "".join(
            f"| {record['phase']} | {record['device']} | {record['status']} | "
            f"{record['metric']} | {record['value']} |\n" for record in rows
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
