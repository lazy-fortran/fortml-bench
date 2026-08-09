# Wave 12 parity evidence

The nine-row CSV covers three production slices:

- class-weighted LDA and QDA with sorted arbitrary labels and class/sample
  weight equivalence;
- exact multi-output ICM GP hyperparameter fitting through FortOpt L-BFGS-B;
- weighted pending-microbatch mass persisted in the MLP checkpoint cursor.

The NumPy rows are independent moment and dense-covariance calculations. The
FortML rows run the corresponding behavioral tests. Each workload carries a
typed CUDA boundary because no resident implementation is claimed for these
three operations.

Reproduce the rows with:

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_wave12_parity.py \
  --fortml ../fortml --output results/wave12_parity.csv
```

Both repository revisions are recorded in the CSV. The result is correctness
evidence and gate timing. It is not a cross-library throughput claim.
