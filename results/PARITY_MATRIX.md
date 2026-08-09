# Parity matrix

This table is a release index, not a claim that every reference-library
feature is finished. A completed row has a report, an independent correctness
oracle, and a machine-readable CSV with source and toolchain provenance.

| Family | FortML lane | Independent evidence | CPU state | CUDA state | Derivatives |
| --- | --- | --- | --- | --- | --- |
| Linear and preprocessing | Ridge, elastic-net, SVR, GLM, imputer, one-hot, PCA, basis/pipeline, joint basis-pipeline training | Reports and analytic or finite-difference oracles | Pass | Typed refusals for model paths | Fixed-fit and joint-objective JVP/VJP/HVP where documented |
| Classification | OVR/OVO logistic, Naive Bayes, ordinal, kNN/radius, LDA/QDA, forests, Extra-Trees, calibration | Classification reports and seeded fixtures | Pass | kNN and forest resident prediction gates, other paths explicit refusals | Continuous models expose products; discrete split/neighbor paths refuse boundaries |
| Classification margins | Binary and deterministic sorted-label OVR finite-basis RBF SVM | `RBF_SVM.md`, `RBF_SVM_MULTICLASS.md`, `rbf_svm.csv`, `rbf_svm_multiclass.csv` with SciPy/NumPy replay | Pass for CPU score/probability maps | Typed `FORTNUM_NOT_IMPLEMENTED` until resident batched RBF kernels | Fixed-state query and packed-parameter JVP/VJP products; fit/hinge boundaries refuse |
| Classification metrics | Multilabel precision/recall/F1/F-beta, Jaccard, Hamming, ROC-AUC, PR-AUC, plus existing scalar metrics | `multilabel_metrics.csv`, `roc_auc.csv` | Pass | Explicit unavailable rows until ranking/reduction kernels are linked | Metrics are hard, nondifferentiable contracts |
| Boosting | Exact and histogram XGBoost-style squared/logistic/Poisson/squared-log/Huber/quantile, monotonic constraints, additive tree contributions | XGBoost reports, `training_core.csv`, and independent tree fixtures | Pass for declared objectives | Typed refusals for resident tree execution | Fixed-tree products and split-boundary refusals |
| Dense neural | MLP, shared multilabel head, chain, grouped objectives, schedules, Adam/AdamW/Adagrad/RMSprop/SGD, resident dense-affine value/JVP inference | MLP, multilabel, grouped-training, device-contract, trajectory, checkpoint, and `training_core.csv` oracles | Pass for declared paths | Resident optimizer and one-layer dense-affine value/JVP primitives; full network graph remains open | Value/JVP/VJP/HVP and selected trajectory hypergradients |
| Exact GP | Kernel catalog, exact and derivative observations, multi-output and large-data fixtures | GP reports and dense covariance oracles | Pass for declared kernels | Resident covariance, factorization, and derivative-query kernels open | Kernel and query products are capability-specific |
| Approximate GP | Sparse, SKI, local experts, variational regression and Bernoulli classification | Large-GP, GP-features, and `gp_variational_classification.csv` reports | Pass for declared CPU slices | Typed refusal for variational resident graph | ELBO gradient/JVP on the CPU slice; natural-gradient/HVP products remain open |
| Physics models | Hamiltonian MLP prototype and PCA/linear-autoencoder initialization | [`PHYSICS_MODELS.md`](PHYSICS_MODELS.md), `physics_models.csv` | Partial | Explicit unavailable row; no resident physics graph | PINN, symplectic-GP, HNN/LNN, and GP-limit initializers remain open |

## Required next benchmark lanes

The remaining parity matrix needs matched scikit-learn estimator and pipeline
fixtures, XGBoost and LightGBM histogram/ranking/categorical fixtures,
PyTorch and JAX compiled and distributed neural training, GPyTorch and GPflow
exact/variational/derivative GP comparisons, and Flux/Lux module-tree checks.
Each lane must use the same data, precision, initialization, stopping rule,
device placement, warmup policy, and correctness tolerance. Resident timings,
transfer-inclusive timings, and typed refusals stay in separate rows.
