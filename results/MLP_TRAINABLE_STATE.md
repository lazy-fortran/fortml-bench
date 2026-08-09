# MLP trainable parameter state

This lane compares the release application with an independent NumPy tanh-MLP oracle. It freezes layer_1.weight, verifies that the packed deployment value is unchanged and that frozen VJP/JVP coordinates are zero, then re-enables the block and checks the analytic JVP. The Fortran behavioral oracle covers transactional unknown-path refusal. CUDA is recorded as unavailable because resident optimizer routing for this metadata path is not claimed.

FortML revision: 2f26f444beb04342b04ebeefe0675475644ecedb
Benchmark revision: 2d8e19bb65f0d2868c93d673e2bd3468c3634626

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | trainable_count | 5.0 | 0.0 |
| independent_oracle | pass | frozen_gradient_max | 0.0 | 0.0 |
| independent_oracle | pass | live_gradient_error | 0.0 | 0.0 |
| independent_oracle | pass | frozen_jvp_max | 0.0 | 0.0 |
| independent_oracle | pass | unfrozen_jvp_max | 0.5045475660572112 | 0.0 |
| independent_oracle | pass | prediction_change | 0.0 | 0.0 |
| release_app | pass | trainable_count | 5.0 | 0.0 |
| release_app | pass | frozen_gradient_max | 0.0 | 0.0 |
| release_app | pass | live_gradient_error | 0.0 | 0.0 |
| release_app | pass | frozen_jvp_max | 0.0 | 0.0 |
| release_app | pass | unfrozen_jvp_max | 0.5045475660572112 | 0.0 |
| release_app | pass | prediction_change | 0.0 | 0.0 |
| release_app | pass | status_code | 0.0 | 0.0 |
| independent_fortran_oracle | pass | test_mlp_trainable_state | 1.0 | 0.0 |
| cuda_typed_refusal | unavailable | resident_parameter_freeze | nan | 0.0 |
