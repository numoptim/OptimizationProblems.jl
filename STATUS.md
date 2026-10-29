# OptimizationProblems.jl Status

Updated 2026-10-29.

## Current focus

Bring all docstrings in line with [BlueStyle](https://github.com/invenia/BlueStyle)
and add the mathematical definition of each GLM family (response parametrization,
objective, gradient, Hessian) to the exported constructor docstrings.

## State of the docstring work

- Plan approved (Gate 1) and committed on
  [`vp/docstring-style`](https://github.com/numoptim/OptimizationProblems.jl/tree/vp/docstring-style):
  [plan.md](https://github.com/numoptim/OptimizationProblems.jl/blob/vp/docstring-style/docs/src/dev/docstring-style/plan.md)
  holds the files, functions, decisions, status checklist, and out-of-scope issues.
  Not yet merged into `main`.
- No docstring has been rewritten yet. Next is Gate 2: the `GLMFamily` and family
  struct docstrings in `src/glm/generalized_linear_models.jl`.
- Key decisions: one merged docstring per family constructor; extra `# Model` and
  `# Data Generation` headings; math only in exported constructors; per-family
  distribution check added to each existing `*_partition.jl`; no docstring lint test.

## Other open work in the repo

- `tpk/inverse-gaussian`: adds Inverse Gaussian regression. Its docstrings should
  follow the new style once merged.
- `tpk/distributions-sampling`: samples Geometric and Negative Binomial responses with
  `Distributions.jl`. Overlaps files in the docstring plan (`Data Generation` text).
- Suspected bugs found while planning (listed in the plan, not fixed): the
  `LogisticRegression` random response has `P(y=1) = 1 - p`; `BinomialRegression`
  default `name` is `"LogisticRegression"`; `LinearRegression` ignores `σ` in the
  objective.

## Next steps

| # | Step | Suggested due |
|:--|:--|:--|
| 1 | Review and merge the plan branch `vp/docstring-style` | 2026-10-30 |
| 2 | Gate 2: `GLMFamily` and family struct docstrings | 2026-11-03 |
| 3 | Rewrite `GeneralizedLinearModel`, `allocate`, `obj!`, `grad!`, `objgrad!`, `hess!` docstrings | 2026-11-10 |
| 4 | Geometric and Negative Binomial constructor docstrings plus distribution testsets (after `tpk/distributions-sampling` lands) | 2026-11-17 |
| 5 | Remaining five families (Bernoulli, Binomial, Exponential, Normal, Poisson) | 2026-12-01 |
| 6 | Triage the out-of-scope issues into GitHub issues | 2026-11-06 |
