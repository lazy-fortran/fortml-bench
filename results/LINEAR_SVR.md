# Weighted linear SVR

`bench_linear_svr.py` fits the dense primal `linear_svr_regression_t` on a
deterministic 192-by-4 fixture with arbitrary real targets, nonnegative sample
weights, feature-only L2 regularization, and epsilon `0.08`. Its independent
SciPy/NumPy oracle minimizes the identical weighted squared
epsilon-insensitive objective and checks every target, weight, fitted packed
parameter, and prediction before retaining timings.

The release app uses FortOpt L-BFGS-B. The recorded CPU rows are fit and
fixed-state affine prediction; the CUDA row is an explicit unavailable
capability record because no resident linear-SVR kernel is linked. This is not
a host fallback or a GPU performance claim.

The ordinary epsilon-insensitive objective is available as
`SVR_LOSS_EPSILON`. Its exact product returns a typed refusal at either
residual kink; fitting uses a small C1 continuation so FortOpt's Armijo
callback remains deterministic. The default squared epsilon-insensitive loss
has a continuous first derivative. Independent tests also cover packed affine
JVP/VJP products, weighted fitting, hyperparameter derivatives, and malformed
input refusals.

Run:

```bash
python -B scripts/bench_linear_svr.py \
    --fortml ../fortml --output results/linear_svr.csv
```

The checked-in CSV records source and benchmark revisions, compiler, NumPy and
SciPy versions, correctness error, and explicit CUDA status.
