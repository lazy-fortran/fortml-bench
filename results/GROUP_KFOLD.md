# Grouped K-fold validation

This lane measures the index-only `group_kfold_splitter_t` contract on ten
rows, six integer groups, and three folds. Group sizes are deliberately
uneven (`3,2,1,2,1,1`). The independent NumPy oracle orders groups by
decreasing size and assigns each to the currently lightest fold, preserving
all group boundaries while producing test folds of sizes `4,3,3`.

The FortML release app exports every test index before timing. The benchmark
requires exact agreement with the NumPy fold assignment and independently
checks that each group occurs in only one test fold. The raw record is
[`group_kfold.csv`](group_kfold.csv). On the recorded host, the NumPy oracle
took `2.7642e-05 s` per complete three-fold generation and FortML took
`1.6211e-07 s`; these are tiny index-workload timings, not end-to-end model
training measurements.

Reproduce with:

```bash
python -B scripts/bench_group_kfold.py \
    --fortml ../fortml --output results/group_kfold.csv
```

CUDA is recorded as `unavailable`: the splitter owns CPU index metadata and
does not claim a resident accelerator implementation or derivative path.
