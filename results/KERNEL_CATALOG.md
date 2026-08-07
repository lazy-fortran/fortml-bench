# Periodic and rational-quadratic kernels

`bench_kernel_catalog.py` checks the dense covariance matrix, parameter JVP,
parameter VJP, parameter HVP, and value/input derivative checksum for the two
new `fortml_kernels` leaves. The NumPy calculations are independent formulas;
the parameter VJP uses the closed logarithmic-parameter derivatives and the HVP
uses a central difference of that independent gradient. Input mixed Hessians
use a separate central finite-difference oracle.

The release app uses 256 three-dimensional points and 24 repetitions. Results
are retained only after every checksum is within the recorded relative
criterion (`2e-6`). The CSV includes both release CPU timings and explicit
CUDA `unavailable` rows. The CUDA rows are a typed capability refusal: the
resident kernel postfix ABI currently has only variance/lengthscale payloads,
while both new leaves require a third positive parameter (period or alpha).
No host timing is relabeled as GPU evidence.

```bash
python -B scripts/bench_kernel_catalog.py \
  --fortml ../fortml --output results/kernel_catalog.csv
```

The recorded source and benchmark revisions are in every CSV row.
