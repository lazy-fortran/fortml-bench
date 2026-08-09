# OVR logistic partial-fit benchmark

This lane checks sorted arbitrary labels, deferred class completion, deterministic replay, malformed-batch rollback, fixed-state JVP products, and the typed CUDA boundary. The Python stream state machine is independent of the Fortran metadata implementation.

FortML revision: 124a9b430085eeb5fbf2343aa36bbc0a0a08adb5
Benchmark revision: 954969ab1ff1dec0a0680d93f073d86c074ac1fe

| phase | device | status | metric | value | max abs error |
| --- | --- | --- | --- | ---: | ---: |
| independent_metadata | cpu | pass | batch_count | 2.0 | 0.0 |
| independent_metadata | cpu | pass | sample_count | 9.0 | 0.0 |
| release_app | cpu | pass | replay_probability_max_abs_error | 0.0 | 0.0 |
| release_app | cpu | pass | batch_count | 2.0 | 0.0 |
| behavioral_gate | cpu | pass | test_ovr_logistic_partial_fit | 1.0 | 0.0 |
| device_boundary | cuda | unavailable | predict_proba_device_status | 3.0 | 0.0 |
