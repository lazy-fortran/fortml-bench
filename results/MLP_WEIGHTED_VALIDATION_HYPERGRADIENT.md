# Weighted validation trajectory benchmark

This lane checks the fixed full-batch SGD momentum trajectory with validation
weights `[1, 2, 4]` against an independent NumPy recurrence and central
finite-difference products.  The packed outer vector is
`[log(learning_rate), log(l2), momentum]`; the fixture uses four updates,
`learning_rate=0.11`, `l2=0.06`, and `momentum=0.29`.

The NumPy oracle reports weighted validation MSE `0.014998256378050192` and
packed weighted hypergradient
`[-0.0610401804644091, 0.00165088250231663, -0.0802196233914802]`.  The
directional JVP for `[0.23, -0.17, 0.11]` is `-0.0231440501021786`.

The affine outer HVP is independently checked on the uniform validation path;
the oracle components are `[0.0379430071650692, -0.000846766233519869,
0.0337074129459075]`.  A non-uniform HVP request is recorded as a typed
`FORTNUM_NOT_IMPLEMENTED` boundary, as is the resident CUDA path.  This is a
capability record, not a claim that weighted HVPs are silently available.

The machine-readable companion is
`results/mlp_weighted_validation_hypergradient.csv`; its provenance columns
pin both repository revisions and reject dirty source pins.
