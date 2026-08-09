# Weighted Gamma GP likelihood

`bench_gp_gamma_likelihood.py` checks the positive-target Gamma density over the joint latent and transformed-shape coordinate space. An independent NumPy scalar density supplies central-difference gradient and directional-HVP oracles. The harness also checks the bounded FortOpt shape fit against SciPy.

Run:

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex python -B scripts/bench_gp_gamma_likelihood.py --fortml ../fortml --output results/gp_gamma_likelihood.csv --report results/GP_GAMMA_LIKELIHOOD.md
```

The maximum product error is `1.399e-06`. The fitted log shape differs from the SciPy optimum by `4.274e-09`. FortML took `1.227e-06` seconds for one joint value-gradient and HVP pair. NumPy/SciPy took `2.318e-05` seconds in the Python loop, giving a NumPy-to-FortML time ratio of `18.886`. The source revision is `a3a28a4b8623086565679cf7a71e0e6237686738` and the benchmark revision is `566fbc1dc67db94601b6c00e45db93d0ca3ee14c`. CUDA has a typed refusal until resident special functions and reductions are linked.
