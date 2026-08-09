# Generic trainer value clipping

This lane compares the generic trainer's per-coordinate gradient clipping against an independent NumPy quadratic oracle. The release app checks the exact SGD update, diagnostic counter, and schema-8 checkpoint round trip.

FortML revision: edead3b99081f8a28f57b23841cd61a2aeccbaeb
Benchmark revision: 1b5d12e05771f451193cc2bfeeda33efe8af27c5+dirty

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | parameter_1 | 0.1 | 0.0 |
| independent_oracle | pass | parameter_2 | 0.9 | 0.0 |
| release_app | pass | parameter_max_abs_error | 0.0 | 0.0 |
| release_app | pass | value_clipped_steps | 1.0 | 0.0 |
| release_app | pass | checkpoint_equal | 1.0 | 0.0 |
| independent_fortran_oracle | pass | test_trainer | 1.0 | 0.0 |
| device_boundary | unavailable | resident_trainer | nan | 0.0 |
