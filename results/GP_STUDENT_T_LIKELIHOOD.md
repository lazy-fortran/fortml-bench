# Fixed-latent Student-t GP likelihood products

`bench_gp_student_t_likelihood.py` checks the normalized Student-t observation likelihood over stable `[log(scale), log(nu)]` coordinates. The independent NumPy scalar density uses central differences for the gradient and directional HVP, then checks the JVP/VJP products and the FortOpt objective callback.

Run:

```bash
python -B scripts/bench_gp_student_t_likelihood.py --fortml ../fortml --output results/gp_student_t_likelihood.csv --report results/GP_STUDENT_T_LIKELIHOOD.md
```

The recorded maximum product error is `1.445e-08` for 7 fixed latent rows. FortOpt decreased the negative log likelihood from `5.1133920696782367e+00` to `-1.4742870480566501e+00`. The release source revision is `9b06d473f5d09417fe513af18f0276fe81d5ff5f` and the benchmark revision is `3dda8f3093ef93a2c8ca6e8e3d8088b369bf6126`. CUDA is an explicit typed refusal until resident latent batches and special functions are linked; no GPU timing or hidden host fallback is claimed.
