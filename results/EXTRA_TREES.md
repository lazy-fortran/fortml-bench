# Extra-Trees classification

`bench_extra_trees.py` exercises the deterministic `extra_trees_classifier_t`
contract on a three-class, three-feature fixture.  The independent NumPy
oracle is a direct first-feature threshold rule; it does not import
scikit-learn and does not reproduce FortML's randomized tree construction.
The release gate requires all six query labels, a probability-simplex error
below `2e-12`, finite non-negative timings, and a typed CUDA refusal with no
host fallback.

The FortML lane uses 32 trees, depth six, two randomly selected features per
node, 64 random thresholds per selected feature, and seed 1729.  Trees use the
complete sample (no bootstrap), while each node keeps the best randomized
Gini split.  CPU fit and prediction timings are reported beside the NumPy
oracle; the CUDA row is `unavailable` until a resident tree kernel is linked.

Run from this repository:

```bash
python scripts/bench_extra_trees.py --fortml ../fortml
```

The generated `results/extra_trees.csv` records both repository revisions,
compiler flags, oracle provenance, and the explicit device-contract row.
