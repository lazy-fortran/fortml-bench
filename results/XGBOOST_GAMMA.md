# Fixed-shape Gamma XGBoost lane

`xgboost_gamma.csv` is a correctness-gated release record for FortML's
fixed-shape Gamma objective. The target is strictly positive and the margin is
the logarithm of the mean. The independent NumPy oracle evaluates

\[
  \ell_i(\eta_i)=\alpha\,[\eta_i+y_i\exp(-\eta_i)],\qquad
  g_i=\alpha\,[1-y_i\exp(-\eta_i)],\qquad
  h_i=\alpha y_i\exp(-\eta_i),
\]

with shape `alpha=2`. The release app checks a four-row exact depth-one split,
then records fit/predict and weighted-quantile histogram CPU rows on a
deterministic 256-row fixture. The CUDA row is `unavailable` with
`FORTNUM_NOT_IMPLEMENTED`: no host timing is relabeled as accelerator work.

The source revision is recorded in every CSV row; rerun with:

```bash
python -B scripts/bench_xgboost_gamma.py \
  --fortml ../fortml --output results/xgboost_gamma.csv
```
