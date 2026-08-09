# Derivative-observation kernel matrix

FortML revision: `e369b7bc8cf38c1c4711db1d37e3ba64d02a6fc2`

Benchmark revision: `88942f83f5f10f22b729ff4bb9cfebc9e1d412d9+dirty`

The independent Fortran oracle covered 14 kernel families with maximum central-difference error `5.079e-07`. It exercised mixed value/first-derivative GP prediction for every family. CUDA is recorded as a typed `FORTNUM_NOT_IMPLEMENTED` capability row because the resident derivative-GP factorization is not linked.
