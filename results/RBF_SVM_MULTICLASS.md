# Multiclass RBF SVM

This correctness-gated lane compares FortML's transactional sorted-label
one-vs-rest finite-basis RBF SVM against independent per-class SciPy
L-BFGS-B weighted squared-hinge solves. The release rows retain margins,
normalized probabilities, packed child parameters, and predictions only
after all arrays agree. CUDA is an explicit typed-unavailable row until
resident batched RBF kernels are linked.

Maximum retained CPU absolute error: 4.689e-07.

Raw data: [`rbf_svm_multiclass.csv`](rbf_svm_multiclass.csv).
