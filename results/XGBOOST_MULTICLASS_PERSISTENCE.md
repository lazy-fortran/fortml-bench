# XGBoost multiclass text persistence

`bench_xgboost_multiclass_persistence.py` exercises the one-vs-rest
`xgboost_multiclass_t` snapshot contract on a deterministic nine-row fixture
with sorted arbitrary labels `[-8, 2, 11]`. The release app fits three
one-vs-rest logistic ensembles, writes one self-contained text artifact, loads
it into a fresh model, and compares every query probability before and after
the round trip.

The release app also reconstructs each probability independently from the
public class margins with a stable sigmoid and checks the normalized simplex.
The Python gate independently checks the row shape, class metadata, total
probability sum (`3` for three queries), and the zero-error round trip before
retaining the timing. The CUDA row is explicit `unavailable`: persistence is
CPU text I/O and no resident CUDA tree kernel is linked.

Run:

```bash
python3 -B scripts/bench_xgboost_multiclass_persistence.py \
  --fortml ../fortml \
  --output results/xgboost_multiclass_persistence.csv
```

The current run records zero maximum absolute error for both the independent
probability oracle and the save/load round trip. The raw CSV records source
revisions, compiler flags, and the typed CUDA boundary.
