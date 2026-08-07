# Multinomial and neural classification workload

`bench_classification_models.py` writes the raw
[`classification_models.csv`](classification_models.csv) and is the
correctness-gated benchmark for the
two classifier families that were previously missing from the cross-engine
record: regularized multinomial softmax regression and a multiclass tanh MLP.
It uses 192 rows, six features, three sorted arbitrary integer labels
`[-7, 3, 11]`, and a 12-unit hidden layer.  The fixture is deterministic and
the labels are deliberately interleaved rather than class-contiguous.

The NumPy reference is behavioral, not a state checksum. It independently
implements a damped-Newton solve for the softmax objective (coefficient L2
`0.05`) and full-batch Adam for the MLP (80 epochs, learning rate `0.03`,
L2 `0.001`). It checks complete predicted-label arrays, every class
probability, probability-simplex normalization, accuracy, and log loss before
any competitor timing.  The MLP initializer reproduces the documented
platform-independent phase initializer only in the oracle, so each FortML row
is compared against an independently calculated result rather than a
copied Fortran checksum.

Run the lane serially from this repository:

```bash
.venv/bin/python -B scripts/bench_classification_models.py \
    --fortml ../fortml --output results/classification_models.csv
```

The current CSV records four NumPy-oracle rows, four scikit-learn context rows,
four FortML CPU passes, four explicit FortML CUDA capability refusals, and
four resident PyTorch CPU/CUDA passes. The NumPy and
FortML references pass with softmax accuracy `0.53125` and MLP accuracy
`0.640625`. The FortML maximum probability error is `2.46e-6` for softmax and
`2.33e-15` for the MLP. PyTorch agrees with the independent MLP oracle to
`4.44e-16` on both devices. All probability normalization errors are at most
`2.22e-16`. Fit and prediction timing fields remain in the raw CSV. The
scikit-learn rows use multinomial `lbfgs`
and a tanh
`MLPClassifier` with matched exposed settings.  They are contextual timings,
not a claim that optimizer stopping or regularization conventions are
bit-for-bit identical.  The optional PyTorch MLP rows use resident float64
tensors, the same initialized weights, explicit L2, and `foreach=False` Adam.
CPU and CUDA are separate records. If either device or the optional package is
unavailable, the harness retains an explicit machine-readable `unavailable`
row instead of omitting that comparison.

FortML integration remains an explicit gate for every release run. The
harness invokes the release target `fortml_bench_classifiers` and sets
`FORTML_BENCH_CLASSIFIER_ORACLE`. If that target is absent it writes four
machine-readable `unavailable` rows rather than silently dropping FortML. The
current release app satisfies this protocol and produces four passing rows.
The app writes a CSV with quantities `label`,
`softmax_probability`, `softmax_prediction`, `mlp_probability`, and
`mlp_prediction` (one-based row/column indices).  It must also emit these
stdout records, whose final field is seconds per operation:

```text
softmax_fit,n_samples,n_features,n_classes,seconds
softmax_predict,n_samples,n_features,n_classes,seconds
mlp_classifier_fit,n_samples,n_features,n_hidden,n_classes,seconds
mlp_classifier_predict,n_samples,n_features,n_hidden,n_classes,seconds
```

The harness refuses incomplete output, checks the complete arrays against the
NumPy oracle, and records compiler flags, Python/NumPy versions, FortML,
FortNum, and benchmark revisions, plus explicit build/execution refusals.  A
FortML pass is
therefore a behavioral result, not merely a successful process exit.

The FortML CUDA rows are intentionally `unavailable` and contain no timing.
The classifier release app has no device-resident implementation, so the
benchmark refuses a CUDA execution rather than relabeling a host measurement.
PyTorch CUDA rows demonstrate the independent resident-device oracle only;
they do not imply CUDA support for FortML's classifier or MLP trainer.
