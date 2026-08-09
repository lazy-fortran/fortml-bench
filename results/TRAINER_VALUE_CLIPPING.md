# Generic trainer value clipping

This lane compares the generic trainer's per-coordinate gradient clipping against an independent NumPy quadratic oracle. The release app checks the exact SGD update, diagnostic counter, and schema-8 checkpoint round trip.

FortML revision: edead3b99081f8a28f57b23841cd61a2aeccbaeb
Benchmark revision: 2ccb52c87a158f9047b1a5b86c13ea0294c0bffd

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | parameter_1 | 0.1 | 0.0 |
| independent_oracle | pass | parameter_2 | 0.9 | 0.0 |
| release_app | pass | parameter_max_abs_error | 0.0 | 0.0 |
| release_app | pass | value_clipped_steps | 1.0 | 0.0 |
| release_app | pass | checkpoint_equal | 1.0 | 0.0 |
| independent_fortran_oracle | pass | test_trainer | 1.0 | 0.0 |
| device_boundary | unavailable | resident_trainer | nan | 0.0 |
