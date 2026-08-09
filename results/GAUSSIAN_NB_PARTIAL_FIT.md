# Gaussian Naive Bayes partial-fit benchmark

The independent NumPy oracle accumulates a sorted class vocabulary, population moments, and Gaussian log densities across two batches. The first batch omits one declared class, so fitting is deferred until the second batch. The Fortran behavioral gate checks transactional unknown-label rollback and CPU/CUDA dispatch. CUDA is an explicit resident sufficient-statistic refusal.

FortML revision: 9b06d473f5d09417fe513af18f0276fe81d5ff5f
Benchmark revision: 1b9f96aa86fc42231f628115f1c84d3399ae0e99

| phase | device | status | metric | value | max abs error |
| --- | --- | --- | --- | ---: | ---: |
| independent_moments | cpu | pass | mean_variance_max_abs_error | 0.0 | 0.0 |
| independent_stream | cpu | pass | batch_count | 2.0 | 0.0 |
| independent_stream | cpu | pass | sample_count | 6.0 | 0.0 |
| behavioral_gate | cpu | pass | test_gaussian_nb_partial_fit | 1.0 | 0.0 |
| device_boundary | cuda | unavailable | partial_fit_device_status | 3.0 | 0.0 |
