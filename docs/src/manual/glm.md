# Generalized Linear Models
A Generalized Linear Model (GLM) relates a response ``y`` to explanatory
variables ``x`` through a linear predictor, a canonical link function,
and a distribution. The GLM functionality in this package is
implemented based on the following set of assumptions:

- **Independence**: observations ``(y_1, x_1), \ldots, (y_n, x_n)``
  are independent of each other.

- **Linear effect**: a linear predictor
  ``\eta_i = \beta^T x_i``.

- **Canonical Link**: the canonical link function for the given
  distribution is used to connect the linear predictor to the mean
  of the response.

- **Distribution**: the response follows a distribution from the exponential family.

Fitting a GLM means finding the ``\beta`` that maximizes the
likelihood of the observed data, equivalently minimizing the
negative log-likelihood ``\ell(\beta) = -\log L(\beta)``. While the functionality below does not directly enable estimation of ``\beta``, it provides the necessary functionality for evaluating the negative log-likelihood and its derivatives. 

!!! note
    Every negative log-likelihood below is stated up to an
    additive constant that does not depend on ``\beta`` (for
    example, terms like ``\log(\sigma^2)`` or ``\log(y_i!)``).
    Dropping such constants does not change where the minimum of
    ``\ell(\beta)`` occurs, since they vanish under differentiation.

## Linear (Normal) Regression

Predicts a continuous value by fitting the line that minimizes the
total squared distance between the line and the observed data.

**Assumptions**

- Response: ``y_i \in \mathbb{R}``.
- Linear effect: a linear predictor
  ``\eta_i = \beta^T x_i``.
- Link (identity): ``\mu_i = \eta_i``.
- Distribution: ``y_i \sim \text{Normal}(\mu_i, \sigma^2)``.

**Negative log-likelihood** (up to a constant not depending on
``\beta``):

```math
\ell(\beta) = \sum_i \left[ -y_i (x_i^T \beta)
  + \tfrac{1}{2} (x_i^T \beta)^2 \right]
```

**Gradient**: ``\sum_i (x_i^T \beta - y_i)\, x_i``

**Hessian**: ``\sum_i x_i x_i^T``

**Example**

```julia
using OptimizationProblems

problem = LinearRegression(Float32, num_param=10, num_obs=100)
store = allocate(problem, type=Float64, hess=true)

x = randn(Float64, 10) # Randomly generated argument.

obj!(problem, store=store, x=x)
grad!(problem, store=store, x=x)
hess!(problem, store=store, x=x)

# Evaluate using only a subset of observations (stochastic evaluation)
obj!(problem, store=store, x=x, batch=1:10)

# Accumulate onto the existing stored value instead of resetting first
obj!(problem, store=store, x=x, batch=11:20, reset=false)
```

The evaluated negative log-likelihood, gradient, and Hessian are in
`store[:obj]`, `store[:grad]`, and `store[:hess]`, respectively. The
`batch` argument restricts evaluation to a subset of observations,
and `reset=false` accumulates onto the existing stored value rather
than clearing it first.

## Logistic (Bernoulli) Regression

Predicts a binary outcome. The linear predictor is passed through
the logistic function so the output is always a valid probability.

**Assumptions**

- Response: ``y_i \in \{0, 1\}``.
- Linear effect: a linear predictor
  ``\eta_i = \beta^T x_i``.
- Link (logit): ``\mu_i = \dfrac{1}{1+\exp(-\eta_i)}``.
- Distribution: ``y_i \sim \text{Bernoulli}(\mu_i)``.

**Negative log-likelihood** (up to a constant):

```math
\ell(\beta) = \sum_i \left[ -y_i (x_i^T \beta)
  + \log\!\left(1+\exp(x_i^T \beta)\right) \right]
```

**Gradient**: ``\sum_i (\mu_i - y_i)\, x_i``

**Hessian**: ``\sum_i \mu_i (1-\mu_i)\, x_i x_i^T``

**Example**

```julia
using OptimizationProblems

problem = LogisticRegression(Float32, num_param=10, num_obs=100)
store = allocate(problem, type=Float64, hess=true)

x = randn(Float64, 10)

obj!(problem, store=store, x=x)
grad!(problem, store=store, x=x)
hess!(problem, store=store, x=x)

# Evaluate using only a subset of observations (stochastic evaluation)
obj!(problem, store=store, x=x, batch=1:10)

# Accumulate onto the existing stored value instead of resetting first
obj!(problem, store=store, x=x, batch=11:20, reset=false)
```

The evaluated negative log-likelihood, gradient, and Hessian are in
`store[:obj]`, `store[:grad]`, and `store[:hess]`, respectively. The
`batch` argument restricts evaluation to a subset of observations,
and `reset=false` accumulates onto the existing stored value rather
than clearing it first.

## Binomial Regression

Like logistic regression, but each observation records ``k``
successes out of ``n`` trials, rather than a single 0/1 outcome.

**Assumptions**

- Response: ``y_i = (k_i, n_i)``, with ``0 \le k_i \le n_i``.
- Linear effect: a linear predictor
  ``\eta_i = \beta^T x_i``.
- Link (logit, same as Bernoulli): ``\mu_i = \dfrac{1}{1+\exp(-\eta_i)}``.
- Distribution: ``k_i \sim \text{Binomial}(n_i, \mu_i)``.

**Negative log-likelihood** (up to a constant):

```math
\ell(\beta) = \sum_i \left[ -k_i (x_i^T \beta)
  + n_i \log\!\left(1+\exp(x_i^T \beta)\right) \right]
```

**Gradient**: ``\sum_i (n_i \mu_i - k_i)\, x_i``

**Hessian**: ``\sum_i n_i \mu_i (1-\mu_i)\, x_i x_i^T``

**Example**

```julia
using OptimizationProblems

problem = BinomialRegression(Float32, num_param=10, num_obs=100)
store = allocate(problem, type=Float64, hess=true)

x = randn(Float64, 10)

obj!(problem, store=store, x=x)
grad!(problem, store=store, x=x)
hess!(problem, store=store, x=x)

# Evaluate using only a subset of observations (stochastic evaluation)
obj!(problem, store=store, x=x, batch=1:10)

# Accumulate onto the existing stored value instead of resetting first
obj!(problem, store=store, x=x, batch=11:20, reset=false)
```

The evaluated negative log-likelihood, gradient, and Hessian are in
`store[:obj]`, `store[:grad]`, and `store[:hess]`, respectively. The
`batch` argument restricts evaluation to a subset of observations,
and `reset=false` accumulates onto the existing stored value rather
than clearing it first.

## Poisson Regression

Predicts count data (non-negative integers), such as the number of
events occurring in a fixed interval.

**Assumptions**

- Response: ``y_i \in \{0, 1, 2, \ldots\}``.
- Linear effect: a linear predictor
  ``\eta_i = \beta^T x_i``.
- Link (log): ``\mu_i = \exp(\eta_i)``.
- Distribution: ``y_i \sim \text{Poisson}(\mu_i)``.

**Negative log-likelihood** (up to a constant):

```math
\ell(\beta) = \sum_i \left[ -y_i (x_i^T \beta)
  + \exp(x_i^T \beta) \right]
```

**Gradient**: ``\sum_i (\mu_i - y_i)\, x_i``

**Hessian**: ``\sum_i \mu_i\, x_i x_i^T``

**Example**

```julia
using OptimizationProblems

problem = PoissonRegression(Float32, num_param=10, num_obs=100)
store = allocate(problem, type=Float64, hess=true)

x = randn(Float64, 10)

obj!(problem, store=store, x=x)
grad!(problem, store=store, x=x)
hess!(problem, store=store, x=x)

# Evaluate using only a subset of observations (stochastic evaluation)
obj!(problem, store=store, x=x, batch=1:10)

# Accumulate onto the existing stored value instead of resetting first
obj!(problem, store=store, x=x, batch=11:20, reset=false)
```

The evaluated negative log-likelihood, gradient, and Hessian are in
`store[:obj]`, `store[:grad]`, and `store[:hess]`, respectively. The
`batch` argument restricts evaluation to a subset of observations,
and `reset=false` accumulates onto the existing stored value rather
than clearing it first.

## Exponential Regression

Predicts a positive continuous value, commonly time until an event
occurs, assuming a constant event rate.

**Assumptions**

- Response: ``y_i > 0``.
- Linear effect: a linear predictor
  ``\eta_i = \beta^T x_i``.
- Link: ``\mu_i = -1/\eta_i``.
- Distribution: ``y_i \sim \text{Exponential}(\text{rate} = -\eta_i)``.

!!! warning
    The linear effect must satisfy ``\eta_i \le 0``. The
    negative log-likelihood contains the term
    ``-\log(-\eta_i)``, which is only defined when ``\eta_i``
    is negative.

**Negative log-likelihood** (up to a constant):

```math
\ell(\beta) = \sum_i \left[ -y_i (x_i^T \beta)
  - \log(-x_i^T \beta) \right]
```

**Gradient**: ``\sum_i \left(-y_i - \dfrac{1}{x_i^T \beta}\right) x_i``

**Hessian**: ``\sum_i \dfrac{1}{(x_i^T \beta)^2}\, x_i x_i^T``

**Example**

```julia
using OptimizationProblems

problem = ExponentialRegression(Float32, num_param=10, num_obs=100)
store = allocate(problem, type=Float64, hess=true)

x = randn(Float64, 10)

obj!(problem, store=store, x=x)
grad!(problem, store=store, x=x)
hess!(problem, store=store, x=x)

# Evaluate using only a subset of observations (stochastic evaluation)
obj!(problem, store=store, x=x, batch=1:10)

# Accumulate onto the existing stored value instead of resetting first
obj!(problem, store=store, x=x, batch=11:20, reset=false)
```

The evaluated negative log-likelihood, gradient, and Hessian are in
`store[:obj]`, `store[:grad]`, and `store[:hess]`, respectively. The
`batch` argument restricts evaluation to a subset of observations,
and `reset=false` accumulates onto the existing stored value rather
than clearing it first.