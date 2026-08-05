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
    parser.add_argument("--backend", choices=("openacc", "native"), default="openacc")
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--rhs", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.n, args.d, args.rhs, args.repetitions) < 1:
        raise SystemExit("n, d, rhs, and repetitions must be positive")
    if args.backend == "native" and args.device != "cuda":
        raise SystemExit("the native backend requires the CUDA device")
    if args.backend == "native" and args.rhs > 8:
        raise SystemExit("the native backend supports at most eight RHS")
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
                "N_RHS": str(args.rhs),
                "REPETITIONS": str(args.repetitions),
                "FORTML_NATIVE_CUDA": "1" if args.backend == "native" else "0",
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
        command = fortml / "benchmark" / "run_rbf_matmat.sh"
        subprocess.run([str(command)], check=True, env=environment, cwd=fortml)
        with (scratch / "fortran.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
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
        "rhs",
        "repetitions",
        "dtype",
        "seconds_per_operation",
        "status",
        "fortml_commit",
        "fortnum_commit",
        "correctness_oracle",
        "compiler",
        "flags",
        "compiler_version",
        "native_cuda_kernel",
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "backend": f"fortml_{args.backend}",
                    "device": args.device,
                    "residency": row["residency"],
                    "n_samples": row["samples"],
                    "n_features": row["features"],
                    "rhs": row["rhs"],
                    "repetitions": row["repetitions"],
                    "dtype": "float64",
                    "seconds_per_operation": row["seconds_per_operation"],
                    "status": "pass",
                    "fortml_commit": fortml_commit,
                    "fortnum_commit": fortnum_commit,
                    "correctness_oracle": metadata.get(
                        "correctness_oracle", ""
                    ),
                    "compiler": metadata.get("compiler", ""),
                    "flags": metadata.get("flags", ""),
                    "compiler_version": metadata.get("compiler_version", ""),
                    "native_cuda_kernel": metadata.get("native_cuda_kernel", "0"),
                }
            )


if __name__ == "__main__":
    main()
