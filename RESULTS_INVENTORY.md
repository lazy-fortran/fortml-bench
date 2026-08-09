# Benchmark result inventory

The benchmark tree is now inventoried rather than treating every old CSV as a
release claim. The 2026-08-09 audit found 318 tracked top-level result files:

- 164 pass `scripts/validate_result_schema.py` without `--allow-dirty`.
- 154 are legacy/non-release records with missing v1 fields, incomplete
  capability metadata, or dirty historical provenance.
- The current release lanes are listed explicitly in `BENCHMARK.md` and are
  validated independently. A historical CSV is never promoted by filename
  alone.

Run the strict release gate for the current evidence set with:

```bash
python -B scripts/validate_result_schema.py \
  results/gp_classification_implicit_prediction.csv \
  results/mlp_optimizer_group_registry.csv \
  results/cuda_boosted_tree.csv \
  results/mlp_pca_initializer.csv \
  results/ovr_logistic_partial_fit.csv \
  results/mlp_rmsprop_weighted_hypergradient.csv \
  results/multi_output_gp_hypergradients.csv \
  results/ranking_metrics.csv
```

Use `--all` only when auditing migration debt. A failing historical row must
be rerun with an independent oracle and v1 provenance, or moved to the
recoverable Trash area. It must not be cited as a completed parity feature.
