# Derivative-observation kernel matrix

FortML revision: `e369b7bc8cf38c1c4711db1d37e3ba64d02a6fc2`

Benchmark revision: `36b7f35c4bafcc036adb3308d166a598c4f3293f`

The independent Fortran oracle covered 14 kernel families with maximum central-difference error `5.079e-07`. It exercised mixed value/first-derivative GP prediction for every family. CUDA is recorded as a typed `FORTNUM_NOT_IMPLEMENTED` capability row because the resident derivative-GP factorization is not linked.
