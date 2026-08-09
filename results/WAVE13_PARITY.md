# Wave 13 parity evidence

The ten-row CSV covers three derivative and state contracts:

- weighted transactional OVO logistic `partial_fit` with arbitrary sorted
  labels and rollback;
- scalar GP-classification hyperparameter JVP/VJP products with logistic and
  probit finite-difference coverage;
- deterministic MLP batch-iterator cursor capture and suffix replay.

The NumPy rows cover the probability-simplex, sorted-label, logistic-envelope,
and seeded-permutation invariants. The FortML rows run the independent
behavioral tests. CUDA remains a typed unavailable result for each operation.

Reproduce the rows with:

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
python -B scripts/bench_wave13_parity.py \
  --fortml ../fortml --output results/wave13_parity.csv
```

The evidence records clean source and benchmark revisions. It is a correctness
gate with timing for the tests, not a cross-library throughput claim.
