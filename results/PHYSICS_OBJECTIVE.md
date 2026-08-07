# Composable physics-residual objective benchmark

`bench_physics_objective.py` uses an independent affine-residual fixture with
four weighted slots: data, differential-equation residual, boundary/initial,
and conservation/invariant. It checks the normalized squared-residual value,
gradient, directional JVP, scalar VJP, and central finite differences. A
nonlinear two-parameter fixture independently checks the exact
reverse-over-forward HVP against a central difference of the gradient. The
FortML gate additionally checks the FortOpt callback adapter, malformed shape
and weight refusal, and both the typed no-provider refusal and the exact HVP
provider path.

Run:

```bash
python3 -B scripts/bench_physics_objective.py \
  --fortml ../fortml --output results/physics_objective.csv
```

The CSV also keeps a separate passing refusal row for a provider that omits
`hvp_proc`; this boundary is distinct from the exact nonlinear HVP row.

The HVP row is a passing exact product when a provider supplies
`physics_residual_hvp_proc`; providers without that callback still receive
`FORTNUM_NOT_IMPLEMENTED`, not a hidden finite-difference fallback. The
objective seam is callback-based and has no built-in resident CUDA dispatch,
so a CUDA capability boundary is retained explicitly until a callback-backed
device adapter is benchmarked.
