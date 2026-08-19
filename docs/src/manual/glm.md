# Generalized Linear Models

A Generalized Linear Model (GLM) relates a response `y` to explanatory
variables `x` through a linear predictor, a link function, and a
distribution. Every GLM in this package follows the same structure:

- **Independence**: observations are independent of each other.
- **Linear effect**: a linear predictor ``\eta_i = \beta^T x_i``.
- **Link**: a function connecting the linear predictor to the mean
  response.
- **Distribution**: the response follows a distribution whose mean is
  given by the link.

## Linear (Normal) Regression

Predicts a continuous value by fitting the line that minimizes the
total squared distance between the line and the observed data.

- Model: `y_i = beta^T x_i + epsilon_i`, `epsilon_i ~ Normal(0, sigma^2)`
- Negative log-likelihood:
  `sum_i [ -y_i*(x_i^T beta) + 1/2*(x_i^T beta)^2 ]`

## Logistic (Bernoulli) Regression

Predicts a binary (0/1) outcome. The linear predictor is passed
through the logistic function so the output is always a valid
probability between 0 and 1.

- Model: `y_i ~ Bernoulli(mu_i)`, `mu_i = 1/(1+exp(-x_i^T beta))`
- Negative log-likelihood:
  `sum_i [ -y_i*(x_i^T beta) + log(1+exp(x_i^T beta)) ]`

## Binomial Regression

Like logistic regression, but each observation records `k` successes
out of `n` trials, rather than a single 0/1 outcome.

- Model: `y_i ~ Binomial(n_i, mu_i)`, same logit link as logistic
- Negative log-likelihood:
  `sum_i [ -k_i*(x_i^T beta) + n_i*log(1+exp(x_i^T beta)) ]`

## Poisson Regression

Predicts count data (non-negative integers), such as the number of
events occurring in a fixed interval.

- Model: `y_i ~ Poisson(mu_i)`, `mu_i = exp(x_i^T beta)`
- Negative log-likelihood:
  `sum_i [ -y_i*(x_i^T beta) + exp(x_i^T beta) ]`

## Exponential Regression

Predicts a positive continuous value, commonly time until an event
occurs, assuming a constant event rate.

- Model: `y_i ~ Exponential(rate = -x_i^T beta)`, requires
  `x_i^T beta <= 0`
- Negative log-likelihood:
  `sum_i [ -y_i*(x_i^T beta) - log(-x_i^T beta) ]`
