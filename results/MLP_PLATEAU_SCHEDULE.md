# Metric-aware plateau schedule

The release lane checks the stateless `MLP_SCHEDULE_PLATEAU` contract. Each
scenario supplies the metric mode, current metric, best value, consecutive
bad-observation count, and reduction count. The independent Python oracle
checks patience resets, `min_delta` comparisons, compounded `factor` rates,
the exact base-rate and factor products, and the documented zero products for
comparison decisions.

The FortML app emits 132 values for each scenario before timing. The committed
CSV therefore contains 132 Python-oracle rows, 132 FortML CPU rows, and two
explicit CUDA capability rows. The CPU rows pass with zero maximum error in
the release run. CUDA is recorded as `unavailable` because no resident
metric-aware optimizer lowering is linked.

Reproduce from this repository with:

```sh
python3 -B scripts/bench_mlp_plateau_schedule.py \
    --fortml ../fortml --output results/mlp_plateau_schedule.csv
```

The application uses the release Fortran archive produced by `fo build` and a
temporary executable under `/mnt/storage`. No host timing is relabeled as a
CUDA measurement.
