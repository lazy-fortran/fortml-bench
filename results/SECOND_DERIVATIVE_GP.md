# RBF/Matérn-5/2 second-derivative GP benchmark

`bench_second_derivative_gp.py` checks the bounded `second_derivative_gp_t`
reference against independent NumPy implementations of scalar one-dimensional
RBF and Matérn-5/2 covariance derivatives. The fixture mixes value,
first-derivative, and second-derivative observations and queries. It records
posterior mean/variance, dense latent joint covariance, input JVP
finite-difference error, and input VJP duality for each kernel. The JVP row
compares an independently coded order-five analytic derivative with a central
difference. The FortML test is run before the public-contract timing row and
also checks malformed orders, the Matérn coincidence refusal, non-RBF kernels,
and the typed CUDA refusal.

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_second_derivative_gp.py \
  --fortml ../fortml --output results/second_derivative_gp.csv
```

The benchmark is a CPU reference. No GPU timing is reported: prediction and
joint covariance dispatch selected CPU contexts, while selected CUDA contexts
return `FORTNUM_NOT_IMPLEMENTED` until a resident derivative covariance and
factorization path is linked. Hyperparameter products and orders above two are
also outside this bounded lane; Matérn-5/2 fifth products at coincidence are
explicitly refused because that derivative is discontinuous.
