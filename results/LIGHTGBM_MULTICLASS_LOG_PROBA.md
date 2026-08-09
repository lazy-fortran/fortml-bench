# Multiclass LightGBM log-probability products

This lane checks stable sorted-label one-vs-rest `predict_log_proba`, input and packed leaf-coordinate JVP/VJP products, and explicit CPU/CUDA dispatch. The independent NumPy oracle exercises a probability tail that would underflow under `log(predict_proba)` and checks the simplex.

Reproduce:

```bash
python -B scripts/bench_lightgbm_multiclass_log_proba.py --fortml ../fortml --output results/lightgbm_multiclass_log_proba.csv --report results/LIGHTGBM_MULTICLASS_LOG_PROBA.md
```

The CUDA row is `unavailable` with typed `FORTNUM_NOT_IMPLEMENTED`. No host fallback timing is reported.
