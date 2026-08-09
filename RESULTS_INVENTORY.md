# Benchmark result inventory

The benchmark tree is now inventoried rather than treating every old CSV as a
release claim. The 2026-08-09 audit found 327 tracked CSV result files (324
top-level and three nested):

- 182 pass `scripts/validate_result_schema.py` without `--allow-dirty`.
- 145 are legacy/non-release records with missing v1 fields, incomplete
  capability metadata, or dirty historical provenance.
- The current release lanes are listed explicitly in `BENCHMARK.md` and are
  validated independently. A historical CSV is never promoted by filename
  alone.

The migration-invalid rows are retained only while their benchmark scripts or
historical reports still reference them. They are not release evidence. The
superseded unreferenced categorical-GP snapshot was archived during this audit;
the current categorical-GP evidence is `gp_categorical_likelihood.csv`.

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
  results/ranking_metrics.csv \
  results/gp_categorical_likelihood.csv \
  results/gaussian_nb_partial_fit.csv \
  results/lightgbm_leafwise.csv \
  results/mlp_loss_scaling.csv \
  results/adagrad_hypergradient_hvp.csv \
  results/gp_ordinal_likelihood.csv \
  results/gp_student_t_likelihood.csv \
  results/lightgbm_multiclass_log_proba.csv \
  results/xgboost_categorical_partition.csv \
  results/xgboost_multiclass_log_proba.csv \
  results/xgboost_cuda.csv \
  results/polynomial_svm.csv \
  results/adamw_beta_hypergradient.csv \
  results/gp_derivative_kernel_matrix.csv \
  results/naive_bayes_partial_fit.csv \
  results/basis_linear_regression.csv \
  results/wave11_derivative_products.csv \
  results/wave12_parity.csv \
  results/wave13_parity.csv
```

Use `--all` only when auditing migration debt. A failing historical row must
be rerun with an independent oracle and v1 provenance, or moved to the
recoverable Trash area. It must not be cited as a completed parity feature.
