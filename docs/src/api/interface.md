# Optimization Interface

The functions below provide overviews of the interfaces for accessing 
objective function, gradient function, Hessian and related information
for an `OptimizationProblem` from the [`OptimizationModels.jl` package
](https://github.com/numoptim/OptimizationModels.jl).

# Allocation 

```@docs 
allocate
```

# Objective Information 

```@docs 
obj!

grad!

objgrad!

hess!
```