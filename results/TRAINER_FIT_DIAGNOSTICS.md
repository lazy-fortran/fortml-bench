# Generic trainer fit diagnostics

This lane gates the schema-7 generic trainer diagnostics against an independent NumPy quadratic oracle. The bounded FortOpt L-BFGS-B row records iteration, line-search, and curvature counters. A callback-stopped Adam row checks fit-call/update counters and the zero L-BFGS-B-specific boundary.

FortML revision: 233edb720a12342f9ea46fe34d5f205287926f82
Benchmark revision: b002753eba3ddf2b321944a58770c81659b1bd60+dirty

| phase | status | metric | value | max abs error |
| --- | --- | --- | ---: | ---: |
| independent_oracle | pass | optimum_parameter_error | 0.0 | 0.0 |
| independent_oracle | pass | adam_iterations | 1.0 | 0.0 |
| release_app | pass | lbfgsb_parameter_error | 0.0 | 0.0 |
| release_app | pass | lbfgsb_line_search_evaluations | 2.0 | 0.0 |
| release_app | pass | lbfgsb_curvature_updates | 2.0 | 0.0 |
| release_app | pass | adam_fit_calls | 1.0 | 0.0 |
| independent_fortran_oracle | pass | test_trainer_fit_diagnostics | 1.0 | 0.0 |
| device_boundary | unavailable | resident_trainer | nan | 0.0 |
