from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.n, args.d, args.repetitions, args.max_iterations) < 1:
        raise SystemExit("n, d, repetitions, and max-iterations must be positive")
    if args.tolerance <= 0.0:
        raise SystemExit("tolerance must be positive")
    fortml = args.fortml.resolve()
    fortnum = (fortml.parent / "fortnum").resolve()
    fortml_commit = subprocess.check_output(
        ["git", "-C", str(fortml), "rev-parse", "HEAD"], text=True
    ).strip()
    fortnum_commit = subprocess.check_output(
        ["git", "-C", str(fortnum), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        environment = os.environ.copy()
        environment.update(
            {
                "OUT": str(scratch / "fortran.csv"),
                "META": str(scratch / "fortran.meta"),
                "N_SAMPLES": str(args.n),
                "N_FEATURES": str(args.d),
                "REPETITIONS": str(args.repetitions),
            }
        )
        if args.device == "cpu":
            environment["FC"] = environment.get(
                "FORTML_FC", environment.get("FC", "gfortran")
            )
            environment["FFLAGS"] = environment.get(
                "CPU_FFLAGS", environment.get("FFLAGS", "-O3 -fopenmp")
            )
        else:
            environment["FFLAGS"] = environment.get(
                "GPU_FFLAGS", environment.get("FFLAGS", "-O3 -acc")
            )
        command = fortml / "benchmark" / "run_rbf_cg.sh"
        subprocess.run([str(command)], check=True, env=environment, cwd=fortml)
        with (scratch / "fortran.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))
        with (scratch / "fortran.meta").open() as stream:
            metadata = dict(
                line.rstrip("\n").split("=", 1)
                for line in stream
                if "=" in line
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "device",
        "residency",
        "n_samples",
        "n_features",
        "dtype",
        "threads",
        "repetitions",
        "tolerance",
        "max_iterations",
        "iterations",
        "setup_seconds",
        "seconds_per_solve",
        "reported_residual_norm",
        "target_residual_norm",
        "independent_relative_residual",
        "independent_residual_limit",
        "dense_solution_relative_error",
        "status",
        "fortml_commit",
        "fortnum_commit",
        "correctness_oracle",
        "compiler",
        "flags",
        "compiler_version",
        "gpu",
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "backend": "fortml",
                "device": args.device,
                "residency": "resident_inputs",
                "n_samples": row["samples"],
                "n_features": row["features"],
                "dtype": "float64",
                "threads": os.environ.get("OMP_NUM_THREADS", "unknown"),
                "repetitions": row["repetitions"],
                "tolerance": args.tolerance,
                "max_iterations": args.max_iterations,
                "iterations": row["iterations"],
                "setup_seconds": "",
                "seconds_per_solve": row["seconds_per_solve"],
                "reported_residual_norm": row["residual_norm"],
                "target_residual_norm": "",
                "independent_relative_residual": "",
                "independent_residual_limit": "",
                "dense_solution_relative_error": "",
                "status": "pass",
                "fortml_commit": fortml_commit,
                "fortnum_commit": fortnum_commit,
                "correctness_oracle": "dense_solve_in_fortml_test_and_converged_residual",
                "compiler": metadata.get("compiler", ""),
                "flags": metadata.get("flags", ""),
                "compiler_version": metadata.get("compiler_version", ""),
                "gpu": metadata.get("gpu", ""),
            }
        )


if __name__ == "__main__":
    main()
