# OptimizationProblems

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://numoptim.github.io/OptimizationProblems.jl/dev/)
[![CI](https://github.com/numoptim/OptimizationProblems.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/numoptim/OptimizationProblems.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/numoptim/OptimizationProblems.jl/graph/badge.svg?token=CR7AFXRO0E)](https://codecov.io/gh/numoptim/OptimizationProblems.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)



`OptimizationProblems.jl` is a research-tier library for the Julia language 
that provides a set of optimization problems with a focus on those arising 
from data science problems. 


## Installation

This package is registered in [NumOptimRegistry](https://github.com/numoptim/NumOptimRegistry).
Add the registry once per Julia installation, then install the package normally:

```julia
] registry add https://github.com/numoptim/NumOptimRegistry
] add OptimizationProblems
```

Alternatively, install directly from the repository URL without adding the registry:

```julia
] add https://github.com/numoptim/OptimizationProblems.jl
```

It is also possible to clone the repository into a local directory.
In that case, refer to [Julia Pkg instructions](
    https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project
).

## Roadmap

### Generalized Linear Models

- [x] Bernoulli (Logistic) Regression
- [x] Binomial Regression
- [x] Exponential Regression
- [x] Normal (Linear) Regression
- [x] Poisson Regression
- [ ] Geometric Regression
- [ ] Negative Binomial Regression
- [ ] Inverse Gaussian Regression

### Quasilikelihood / Wedderburn Models

Regression models in which only the mean–variance relationship is specified
rather than a full distributional family. Implemented via a general
`WedderburnModel` type that combines a link function with a variance function,
using adaptive numerical integration to evaluate the quasi-deviance.

**Link functions:** Identity, Logistic, Inverse Complementary Log-Log, Log,
Inverse, and others.

**Variance functions:**
- [ ] Shifted Monomial: `V(μ) = μ^(2p) + c` (covers Quasi-Poisson at `p=0.5, c=0`, Gamma at `p=1, c=0`, Inverse Gaussian at `p=1.5, c=0`)
- [ ] Sinusoidal: `V(μ) = 1 + μ + sin(2πμ)`
- [ ] Negative Binomial: `V(μ) = μ + κμ²`
- [ ] Tweedie: `V(μ) = μ^p` with `p ∈ (1, 2)`

### Unconstrained Test Problems

Classic benchmark problems for unconstrained optimization algorithms, natively
implemented in Julia (no dependency on the Fortran/C CUTEst infrastructure).

**Fixed-size problems:**
- [ ] Rosenbrock (n=2)
- [ ] Beale (n=2)
- [ ] Himmelblau (n=2)
- [ ] Three-Hump Camel (n=2)

**Parameterized problems (from the CUTE/CUTEst catalogue):**
- [ ] ARGLINA
- [ ] ARWHEAD
- [ ] BDQRTIC
- [ ] BROYDN7D
- [ ] CHNROSNB
- [ ] ENGVAL1

### Penalized Regression

- [ ] Ridge Regression (`‖y - Xβ‖² + λ‖β‖²`)
- [ ] LASSO (`‖y - Xβ‖² + λ‖β‖₁`, smooth-part methods; proximal operator for the L1 term is left to the optimization algorithm)
- [ ] Elastic Net (`‖y - Xβ‖² + λ₁‖β‖₁ + λ₂‖β‖²`)

### Survival Analysis

- [ ] Cox Proportional Hazards (partial log-likelihood)

### Hierarchical Models

- [ ] Linear Mixed Models (LMMs): marginal likelihood optimization over fixed effects `β` and variance components `θ`, supporting diagonal and unstructured random-effects covariance structures
- [ ] Generalized Linear Mixed Models (GLMMs): Laplace approximation to the marginal likelihood for non-Normal responses

## License

MIT License
