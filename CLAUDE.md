# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**
```
julia --project=test test/runtests.jl
```

**Run a single test file** (prefix lines in `test/test.txt` with `%` to skip them, or temporarily add a single path):
```
julia --project=test -e 'using Test, OptimizationProblems; include("test/glm/partition_functions/bernoulli_partition.jl")'
```

**Build documentation:**
```
julia --project=docs docs/make.jl
```

**Instantiate dependencies** (first-time setup — requires the NumOptim private registry):
```
julia --project=. -e 'using Pkg; Pkg.Registry.add(RegistrySpec(url="https://github.com/numoptim/NumOptimRegistry.git")); Pkg.instantiate()'
```

## Architecture

`OptimizationProblems.jl` is a Julia package exposing optimization problems (negative log-likelihoods) from statistical models, designed to be consumed by numerical optimization algorithms.

**Dependency on `OptimizationModels`:** The package depends on `OptimizationModels` (from the private NumOptim registry). That package defines the `OptimizationProblem` abstract type, `Counter`, and counter utilities (`increment_batch!`, `increment_block!`, `reset!`). `GeneralizedLinearModel` subtypes `OptimizationProblem`.

**Core struct:** `GeneralizedLinearModel{R, F, G<:GLMFamily}` in `src/glm/generalized_linear_models.jl` is the single concrete type for all GLMs. It holds typed `resp` and `feat` fields (types vary by family) and a `family::G` tag. All arithmetic dispatch is via the `family` tag.

**GLM family dispatch pattern:** Each family file under `src/glm/partition_functions/` implements four internal functions dispatching on its `GLMFamily` subtype:
- `likelihood(family; x, resp, feat)` → scalar loss for one observation
- `score!(family; gradient, x, resp, feat, params)` → in-place gradient accumulation
- `likelihoodscore!(family; gradient, x, resp, feat, params)` → combined (avoids recomputing η)
- `information!(family; hessian, x, resp, feat, params)` → in-place Hessian accumulation

The public API (`obj!`, `grad!`, `objgrad!`, `hess!`) loops over observations and dispatches to these internal functions. The `params` (block) and `batch` arguments enable coordinate/mini-batch subproblems — important for stochastic and block-coordinate optimization algorithms.

**Counter tracking:** Every call to `obj!`/`grad!`/`hess!` increments counters in `problem.counters` (`:obj`, `:grad`, `:hess`, `:residual`, `:jacobian`). Counters track `batch_equivalent` and `block_equivalent` as fractions of total observations/parameters, enabling algorithm cost accounting. Call `reset!(counter)` to reset between tests.

**Memory preallocation:** `allocate(problem; type, obj, grad, hess, weights, residual, jacobian)` returns a `Dict{Symbol,Any}` that all in-place functions write into. Keys: `:obj`, `:grad`, `:hess`, `:weights`, `:residual`, `:jacobian`.

**Test structure:** `test/test.txt` is the test manifest — paths relative to `test/`, lines starting with `%` are comments/disabled. `test/runtests.jl` reads this file and `include`s each active path. Each test file is wrapped in its own module to avoid name collisions.

**Adding a new GLM family:**
1. Add a `struct NewFamily <: GLMFamily end` in a new file under `src/glm/partition_functions/`
2. Implement `likelihood`, `score!`, `likelihoodscore!`, `information!` (and optionally `gnn_weight`, `gnn_constant`, `gnn_coefficient!` for GGN support)
3. Add a constructor function returning `GeneralizedLinearModel{RespType, FeatType, NewFamily}`
4. `include` the file in `src/glm/generalized_linear_models.jl`
5. Export the constructor from `src/OptimizationProblems.jl`
6. Add test files to `test/glm/partition_functions/` and register them in `test/test.txt`
