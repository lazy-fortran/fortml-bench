# Student-t process benchmark

`bench_student_t_process.py` checks the Student-t process regression contract
against an independent NumPy Cholesky oracle. The large-degree-of-freedom row
must converge to an exact GP with the same kernel and noise. A second row uses
the same inputs and kernel with calm and surprising observations: Student-t
predictive covariance must widen for surprising data, while the GP reference
variance remains unchanged. This is the data-dependent covariance distinction
from Shah, Wilson, and Ghahramani's Student-t-process construction.

Run the lane against the pinned FortML source:

```bash
FO_SCAN_FALLBACK=regex python3 -B scripts/bench_student_t_process.py \
  --fortml ../fortml --output results/student_t_process.csv
```

The CSV records independent large-`nu` mean/variance errors, the wild-to-calm
covariance-scale ratio, the FortML behavioral-gate timing, and the explicit
invalid-`nu` refusal row. `nu <= 2` is reported as `refused` with
`FORTNUM_DOMAIN_ERROR`, because a Student-t process has no finite covariance in
that range. No GPU timing is reported: this API currently has no resident
device implementation.
