# Composable physics-residual objective benchmark

`bench_physics_objective.py` uses an independent affine-residual fixture with
four weighted slots: data, differential-equation residual, boundary/initial,
and conservation/invariant. It checks the normalized squared-residual value,
gradient, directional JVP, scalar VJP, and central finite differences. The
FortML gate additionally checks the FortOpt callback adapter, malformed shape
and weight refusal, and the explicit residual-HVP refusal.

Run:

```bash
python3 -B scripts/bench_physics_objective.py \
  --fortml ../fortml --output results/physics_objective.csv
```

The HVP row is a passing typed refusal (`FORTNUM_NOT_IMPLEMENTED`), not a
missing result: no finite-difference fallback is hidden. The objective seam
is callback-based and has no built-in resident CUDA dispatch, so a CUDA
capability boundary is retained explicitly until a callback-backed device
adapter is benchmarked.
