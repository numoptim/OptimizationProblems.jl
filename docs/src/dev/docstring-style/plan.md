# Docstring Style and Mathematical Descriptions

## Goal

Rewrite every docstring in `src/` to follow the
[BlueStyle](https://github.com/invenia/BlueStyle) docstring rules, and add the
mathematical definition of each GLM family (what the response is, its
distribution, link, negative log-likelihood, gradient, Hessian) to the
docstrings of the exported constructors, so the docstrings alone answer
questions such as "does Geometric count failures or trials?".

## BlueStyle Rules Being Applied

- Wrap docstring lines at 92 characters. Project preference: aim for 80 and treat 92
  as a hard stop (tables excluded).
- Signature line indented four spaces, with arguments, keywords, and the return type
  after `->` (e.g. `f(x::Int; k::Bool=true) -> Int`).
- Sections as Markdown headings: `# Arguments`, `# Keywords`, `# Returns`, `# Throws`.
  Each bullet is a backticked name/type followed by the description; wrapped bullet
  lines are indented four spaces.
- Types: signature line, one-line description, `# Fields` (backticked bullets).
- Document the function rather than each method; list several signatures in one
  docstring when methods share behavior.
- Only exported functions are required to be documented; internal ones are optional.
- Docstrings are concise Markdown.

Current docstrings deviate in these ways: no `->` return type on the signature line;
prose "hanging indent" instead of sections (`# Arguments` appears only in `allocate`);
no `# Keywords`/`# Returns`/`# Throws`; the two constructor methods of each family have
separate docstrings; internal interface functions have none.

## Mathematical Conventions (to be stated in the docstrings)

Observation ``i`` has features ``x_i`` and linear effect ``\eta_i = x_i^\top \beta``.
Each per-observation objective is the negative log-likelihood with terms free of
``\beta`` dropped (same convention as `docs/src/manual/glm.md`).

| Family | Response | Distribution | Canonical parameter / mean | ``\ell_i(\beta)`` |
|:--|:--|:--|:--|:--|
| Normal | ``y\in\mathbb{R}`` | ``N(\mu,\sigma^2)`` | ``\mu=\eta`` | ``-y\eta+\eta^2/2`` |
| Bernoulli | ``y\in\{0,1\}`` | Bernoulli(``\mu``) | ``\mu=1/(1+e^{-\eta})`` | ``-y\eta+\log(1+e^\eta)`` |
| Binomial | ``(k,n)``, ``0\le k\le n`` | ``k\sim`` Bin(``n,\mu``) | ``\mu=1/(1+e^{-\eta})`` | ``-k\eta+n\log(1+e^\eta)`` |
| Poisson | ``y\in\{0,1,\dots\}`` | Poisson(``\mu``) | ``\mu=e^\eta`` | ``-y\eta+e^\eta`` |
| Exponential | ``y>0`` | rate ``-\eta`` | ``\mu=-1/\eta``, ``\eta<0`` | ``-y\eta-\log(-\eta)`` |
| Geometric | ``y\in\{0,1,\dots\}`` = **failures before the first success** | ``P(y)=(1-p)^y p`` | ``e^\eta=1-p\in(0,1)``, ``\eta<0``, ``\mu=e^\eta/(1-e^\eta)`` | ``-y\eta-\log(1-e^\eta)`` |
| Negative Binomial | ``(y,r)`` = **(failures, target successes)**, ``r\ge1`` a known per-observation count (data, not estimated) | ``P(y)=\binom{y+r-1}{y}(1-p)^y p^r`` | ``e^\eta=1-p``, ``\mu=r e^\eta/(1-e^\eta)`` | ``-y\eta-r\log(1-e^\eta)`` |

Geometric semantics were confirmed against the code, not assumed: the sampler
`floor(log(1-U)/η)` satisfies ``P(Y\ge y)=e^{y\eta}``, i.e. it counts failures before
the first success (support starts at 0), and the objective `-y*η - log(1 - exp(η))`
is the NLL of ``P(y)=(e^\eta)^y(1-e^\eta)``. This matches `Distributions.Geometric`.
Negative Binomial is the sum of ``r`` such variables, matching
`Distributions.NegativeBinomial(r, p)`. Gradient/Hessian for Geometric:
``\sum_i(\mu_i-y_i)x_i`` and ``\sum_i\mu_i(1+\mu_i)x_ix_i^\top``.

## Proposed Docstring Templates

**Constructor** (one docstring per family, attached to the first method, listing both
signatures; this replaces the two separate docstrings):

```
    GeometricRegression(::Type{T}; num_param, num_obs, name) -> GeneralizedLinearModel
    GeometricRegression(; resp, feat, name) -> GeneralizedLinearModel

One-line summary.

# Model
Math block: response meaning, pmf/pdf, link, NLL, gradient, Hessian, domain restriction.

# Arguments / # Keywords / # Returns / # Throws
...

# Data Generation
How the random-problem method draws `feat`, the oracle parameter, and `resp`.
```

**Family struct**: signature line, one sentence naming the response and its
parametrization (e.g. "failures before the first success"), and a
``[`XRegression`](@ref)`` pointer to the constructor docstring that holds the math.
`GLMFamily` points to `GeneralizedLinearModel`.

**Public API (`obj!`, etc.)**: signature with `->`, the formula being computed over
`batch`/`block`, `# Arguments`, `# Keywords`, `# Returns`, counter side effects.

## Files

| File | Purpose | Planned change |
|:--|:--|:--|
| `src/glm/generalized_linear_models.jl` | Core type, interface functions | Restyle all docstrings; add generic stubs with docstrings for the four internal interface functions |
| `src/glm/partition_functions/bernoulli.jl` | Logistic regression | Merge constructor docstrings, restyle, add math |
| `src/glm/partition_functions/binomial.jl` | Binomial regression | same |
| `src/glm/partition_functions/exponential.jl` | Exponential regression | same |
| `src/glm/partition_functions/geometric.jl` | Geometric regression | same; states failures-before-success |
| `src/glm/partition_functions/negative_binomial.jl` | Negative binomial regression | same; states (failures, successes) pair |
| `src/glm/partition_functions/normal.jl` | Linear regression | same |
| `src/glm/partition_functions/poisson.jl` | Poisson regression | same |
| `test/glm/partition_functions/<family>_partition.jl` (7 existing files) | Per-family likelihood/score/information tests | Add one testset each checking the documented distribution (see Tests) |
| `test/Project.toml` | Test environment | Add `Distributions` (currently only a transitive entry in the Manifest) |
| `docs/make.jl` | Docs nav | Add "Development" entry for this plan (placement to confirm) |

## Functions

Line counts are for the docstring (the deliverable); no function logic changes.

`src/glm/generalized_linear_models.jl`
- `GLMFamily` and seven family structs: one-line description plus response
  parametrization (about 3 lines each, about 25 total).
- `GeneralizedLinearModel`: rewrite `# Fields` wrapping, complete the `# Constructors`
  list, add a general `# Model` section (``\ell(\beta)=\sum_i\ell_i(\beta)``,
  ``\eta_i=x_i^\top\beta``, canonical link, dropped constants, role of `batch`/`block`,
  family tag selects ``\ell_i``; no per-family formulas) (about 45).
- `allocate`: signature with `->`, `# Keywords` instead of `# Arguments`, fix
  `num_param` typo in jacobian line (about 30).
- `obj!`, `grad!`, `objgrad!`, `hess!`: formulas over `batch`/`block`, sections,
  counter effects, fix `hess!` signature (comma should be `;`) (about 30 each, 120 total).
- New stubs `function likelihood end`, `score!`, `likelihoodscore!`, `information!`
  with interface-contract docstrings (about 25 each, 100 total). These document the
  extension point from CLAUDE.md and carry no math; per-family math lives in the
  constructors.

Each family file (7 files)
- Merged constructor docstring with Model, Arguments, Keywords, Returns, Throws, Data
  Generation, domain warnings (about 60 to 80 each, about 500 total).
- Delete the second method's docstring (content folded into the first).

Tests (no new files, nothing to register in `test/test.txt`)
- Each `<family>_partition.jl` gets one added `@testset` (about 20 lines each) that,
  for several random `x`, `feat` and a few responses, checks
  `likelihood(x1) - likelihood(x2) ≈ -(logpdf(x1) - logpdf(x2))` against the matching
  `Distributions` distribution. Differences cancel the dropped ``\beta``-free
  constants. This pins down the documented response parametrization (e.g. Geometric
  counts failures, Negative Binomial is `(failures, successes)`).
- `Distributions` is loaded with `import Distributions` in the test module, because
  `Distributions.Geometric`, `Binomial`, `Exponential`, `Normal` and `Poisson` clash
  with the package's own family type names.
- No docstring lint test for now; line-length and coverage checks are left to a future
  linting/formatting step.

## Suggested Order

1. `generalized_linear_models.jl` (sets the template everything else copies).
2. `geometric.jl` and `negative_binomial.jl` (the motivating ambiguity).
3. Remaining five families.
4. Docs nav. The distribution-semantics testset for each family is added at the same
   gate as that family's docstring, so each family is reviewed as one unit.

## Status Checklist

Because the docstring is the deliverable, "stub" and "logic" collapse: the first box is
the docstring plus test outline, the second is the filled-in test.

- [x] Gate 1: this plan approved
- [ ] `generalized_linear_models.jl`: family structs and `GLMFamily`
- [ ] `generalized_linear_models.jl`: `GeneralizedLinearModel`
- [ ] `generalized_linear_models.jl`: `allocate`
- [ ] `generalized_linear_models.jl`: `obj!`
- [ ] `generalized_linear_models.jl`: `grad!`
- [ ] `generalized_linear_models.jl`: `objgrad!`
- [ ] `generalized_linear_models.jl`: `hess!`
- [ ] `generalized_linear_models.jl`: `likelihood`, `score!`, `likelihoodscore!`,
  `information!` stubs
- [ ] `geometric.jl` + `geometric_partition.jl` testset
- [ ] `negative_binomial.jl` + `negative_binomial_partition.jl` testset
- [ ] `bernoulli.jl` + `bernoulli_partition.jl` testset
- [ ] `binomial.jl` + `binomial_partition.jl` testset
- [ ] `exponential.jl` + `exponential_partition.jl` testset
- [ ] `normal.jl` + `normal_partition.jl` testset
- [ ] `poisson.jl` + `poisson_partition.jl` testset
- [ ] `test/Project.toml`: add `Distributions`
- [ ] `docs/make.jl` nav entry

Each family/function item has two sub-boxes in practice: docstring + test outline
approved, then filled-in test approved.

## Out of Scope / Potential Issues

Noticed while reading; not to be fixed under this plan.

- `LogisticRegression` random method: `resp = rand(num_obs) .>= p` gives
  ``P(y=1)=1-p``, the opposite of the model it is documented to follow.
- `BinomialRegression` default `name` is `"LogisticRegression"`, probably meant to be
  `"Binomial Regression"`.
- `ExponentialRegression` docstring uses `!!! warn` (Documenter admonition should be
  `warning`), and both it and the manual allow ``\eta_i = 0`` ("non-positive"), but
  the ``-\log(-\eta_i)`` term needs strictly negative. Check against the code before
  rewording.
- Docstring typos: "distirbution" (exponential), "wih" (binomial), "nothign"
  (allocate), unmatched brace in the Binomial return type, `num_params` in
  `LinearRegression`.
- `docs/make.jl` has "Genearlized" in the API nav label.
- `docs/src/manual/glm.md` has no Geometric or Negative Binomial sections.
- `gnn_weight`, `gnn_constant`, `gnn_coefficient!` exist only for Bernoulli and are
  undocumented; the README roadmap still shows Geometric and Negative Binomial
  unchecked.
- `LinearRegression` takes `σ` but only uses it when generating data; the objective,
  gradient and Hessian assume ``\sigma^2=1``. The docstring will say so, but the
  behavior may deserve an issue.
- Per-observation loop allocates `view(problem.feat, i, :)` each iteration; unrelated
  to docs.

## Decisions Log

- Math placement: general framework in `GeneralizedLinearModel`; per-family math only in
  the constructors; family structs carry one parametrization sentence plus a pointer.
- Table entries must be verified against the code, not the manual, when each docstring
  is written. Known case: `likelihood(::Normal)` is ``(y-\eta)^2/2`` (``\sigma^2=1``),
  which differs from the table's ``-y\eta+\eta^2/2`` by the ``\beta``-free ``y^2/2``.

- Approved (option A): merge each family's two constructor docstrings into one using
  BlueStyle's multi-signature form, attached to the first method.
- Approved: extra `# Model` and `# Data Generation` headings (extensions of BlueStyle's
  section list).
- Approved: math lives only in the exported constructor docstrings, not on the internal
  `likelihood`/`score!`/`information!` stubs.
- Approved: "Development" entry in `docs/make.jl` for this plan.
- Approved: Geometric and Negative Binomial manual sections stay out of scope.
- Approved: distribution-semantics checks go inside each family's existing
  `*_partition.jl`, not in a new file.
- Approved: no docstring lint test now (deferred to a future linting step). Width
  target is 80 characters with 92 as the hard stop, tables excluded.
