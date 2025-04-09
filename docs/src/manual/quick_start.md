# Quick Start

The library provides an interface for creating optimization problems, and then 
evaluating the objective function, gradient function and Hessian function.

The following example creates a random [`LogisticRegression`](@ref) problem
with 10 parameters and 100 observations (features are stored as `Float32`); 
creates a `store` object to retain evaluated information (stored as `Float64`); 
evaluates the objective function (the negative log-likelihood) at a randomly generated 
argument `x`;
and evaluates the gradient function at the same argument.

```julia
using OptimizationProblems

problem = LogisticRegression(Float32, num_param=10, num_obs=100)
store = allocate(problem, type=Float64)

x = randn(Float64, 10) # Randomly generated argument. 

obj!(problem, store=store, x=x)
grad!(problem, store=store, x=x)
```
The evaluated objective and gradient are in `store[:obj]` and `store[:grad]`,
respectively.

If we have a new argument, `z`, for which we want to evaluate the objective 
and gradient function, we do the following evaluations.
```julia
z = randn(Float64, 10) # New argument

obj!(problem, store=store, x=z)
grad!(problem, store=store, x=z)
```
The evaluated objective and gradient are in `store[:obj]` and `store[:grad]`,
respectively.

See the API for examples of problems; 
for more information about [`allocate`](@ref);
and more information about [`obj!`](@ref), [`grad!`](@ref), [`objgrad!`](@ref),
and [`hess!`](@ref).
