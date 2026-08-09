# Registered linear-operator GP

This lane checks a named first-order operator registry with columns `[value, d/dx]`. Three mixed operator rows (value, gradient, and two Robin combinations) are fitted and queried through an exact dense RBF GP.

The independent NumPy oracle assembles value, gradient, and mixed-Hessian covariance blocks directly, solves the noisy dense system, and central-differences query operator coefficients for JVPs. The release Fortran test additionally checks the JVP/VJP adjoint identity. CUDA is an explicit typed refusal until a resident operator covariance graph is linked.

Maximum prediction error: `1.347e-10`. CPU prediction time: `3.316e-06` seconds per query batch.

FortML revision: `66879d11ba7b261fab1922c7e37c67b65fee5b5c`. Benchmark revision: `874d4ad43290caecbbc45846877672ae0415cf21`.
