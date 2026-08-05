# Benchmark provenance

The benchmark compares source revisions, not moving development trees. Each
run records the `fortml` and `fortnum` commit, compiler versions, Python
package versions, CUDA runtime, driver, GPU model, precision, problem shape,
residency mode, warm-up count, repetitions, and correctness error.

The independent RBF oracle evaluates the stated pairwise formula in blocked
NumPy loops. It is separate from the dense PyTorch, KeOps, GPyTorch, and
Fortran implementations.

The KeOps and GPyTorch adapters follow their public PyTorch interfaces. The
Fortran adapter invokes the pinned `fortml` benchmark entry point and records
the source revision it used. No competitor source is linked into the MIT
Fortran libraries.

The first recorded run uses 2048 samples, 8 features, float64, and 12 timed
MVM repetitions. Its CPU lane uses 16 physical CPU cores and its GPU lane uses an
RTX 5060 Ti with resident and transfer-inclusive timings. The direct oracle
must pass before a timing is written to the CSV.
