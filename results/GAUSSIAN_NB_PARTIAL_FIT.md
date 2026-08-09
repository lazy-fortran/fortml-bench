# Gaussian Naive Bayes partial-fit benchmark

The independent NumPy oracle accumulates a sorted class vocabulary, population moments, and Gaussian log densities across two batches. The first batch omits one declared class, so fitting is deferred until the second batch. The Fortran behavioral gate checks transactional unknown-label rollback and CPU/CUDA dispatch. CUDA is an explicit resident sufficient-statistic refusal.

FortML revision: fb28717d296030d94416ceea2ec4519a83c93e3f
Benchmark revision: 3f896b62c61c3a1d0bb9267f47aa9066331ef1f7+dirty

| phase | device | status | metric | value | max abs error |
| --- | --- | --- | --- | ---: | ---: |
| independent_moments | cpu | pass | mean_variance_max_abs_error | 0.0 | 0.0 |
| independent_stream | cpu | pass | batch_count | 2.0 | 0.0 |
| independent_stream | cpu | pass | sample_count | 6.0 | 0.0 |
| behavioral_gate | cpu | pass | test_gaussian_nb_partial_fit | 1.0 | 0.0 |
| device_boundary | cuda | unavailable | partial_fit_device_status | 3.0 | 0.0 |
