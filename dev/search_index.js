var documenterSearchIndex = {"docs":
[{"location":"api/interface/#Optimization-Interface","page":"Optimization Interface","title":"Optimization Interface","text":"","category":"section"},{"location":"api/interface/","page":"Optimization Interface","title":"Optimization Interface","text":"The functions below provide overviews of the interfaces for accessing  objective function, gradient function, Hessian and related information for an OptimizationProblem from the OptimizationModels.jl package .","category":"page"},{"location":"api/interface/#Allocation","page":"Optimization Interface","title":"Allocation","text":"","category":"section"},{"location":"api/interface/","page":"Optimization Interface","title":"Optimization Interface","text":"allocate","category":"page"},{"location":"api/interface/#OptimizationProblems.allocate","page":"Optimization Interface","title":"OptimizationProblems.allocate","text":"allocate(\n    problem::GeneralizedLinearModel;\n    type::DataType=Float64,\n    obj::Bool=true, \n    grad::Bool=true,\n    hess::Bool=false,\n    weights::Bool=false,\n    residual::Bool=false,\n    jacobian::Bool=false,\n)\n\nAllocates storage necessary in a Dict for calculating objective information to reduce      allocations during runtime. \n\nArguments\n\nproblem::GeneralizedLinearModel, specifies the problem that is being considered\ntype::DataType, specifies the target DataType for all calculations. Defaults to     Float64.\nobj::Bool, if true adds a key :obj to the storage dictionary with value type(0.0).   Otherwise, does nothing to the storage dictionary. \ngrad::Bool, if true adds a key :grad to the storage dictionary with value    zeros(type, problem.num_param). Otherwise, does nothing to storage dictionary. \nhess::Bool, if true adds a key :hess to the storage dictionary with value    zeros(type, problem.num_param, problem.num_param). Otherwise, does nothing to    storage dictionary. \nweights::Bool, if true adds a key :weights to the storage dictionary with value    zeros(type, problem.num_obs). Otherwise, does nothing to storage dictionary. \nresidual::Bool, if true adds a key :residual to the storage dictionary with value    zeros(type, problem.num_obs). Otherwise, does nothing to storage dictionary.\njacobian::Bool, if true adds a key :jacobian to the storage dictionary with value    zeros(type, problem.num_obs, num_param). Otherwise, does nothign to storage   dictionary. \n\nReturns\n\nObject of type Dict{Symbol, Any}.\n\n\n\n\n\n","category":"function"},{"location":"api/interface/#Objective-Information","page":"Optimization Interface","title":"Objective Information","text":"","category":"section"},{"location":"api/interface/","page":"Optimization Interface","title":"Optimization Interface","text":"obj!\n\ngrad!\n\nobjgrad!\n\nhess!","category":"page"},{"location":"api/interface/#OptimizationProblems.obj!","page":"Optimization Interface","title":"OptimizationProblems.obj!","text":"obj!(problem::GeneralizedLinearModel; store::Dict{Symbol, Any},\n    x::Vector{T}, reset::Bool=true, batch::AbstractVector{Int64}=\n    Base.OneTo(problem.num_obs)\n) where T\n\nUpdates the value of the objective function stored in store[:obj] for      the problem specified in problem using the observations specified in      batch. If reset is true, then the value of the objective is      first set to zero. Returns nothing.\n\n\n\n\n\n","category":"function"},{"location":"api/interface/#OptimizationProblems.grad!","page":"Optimization Interface","title":"OptimizationProblems.grad!","text":"grad!(problem::GeneralizedLinearModel; store::Dict{Symbol, Any},\n    x::Vector{T}, reset::Bool=true, batch::AbstractVector{Int64}=\n    Base.OneTo(problem.num_obs), block::AbstractVector{Int64}=eachindex(x)\n) where T\n\nUpdates the value of the gradient function stored in store[:grad] for the      problem specified in problem using the observation specified in      batch and only updates the entries in block. If reset==true,     sets store[:grad][block] to zero. Returns nothing.\n\n\n\n\n\n","category":"function"},{"location":"api/interface/#OptimizationProblems.objgrad!","page":"Optimization Interface","title":"OptimizationProblems.objgrad!","text":"objgrad!(problem::GeneralizedLinearModel; store::Dict{Symbol, Any},\n    x::Vector{T}, reset::Bool=true, batch::AbstractVector{Int64}=\n    Base.OneTo(problem.num_obs), block::AbstractVector{Int64}=eachindex(x)\n) where T\n\nUpdates the objective value and gradient value in store for the problem      specified in problem using the observations specified in batch.     If block is specified, then, for the gradient update,      only the values in store[:grad][block]. block does not impact the      objective update. If reset==true, then the objective and      store[:grad][block] are set to zeros. Returns nothing.\n\n\n\n\n\n","category":"function"},{"location":"api/interface/#OptimizationProblems.hess!","page":"Optimization Interface","title":"OptimizationProblems.hess!","text":"hess!(problem::GeneralizedLinearModel, store::Dict{Symbol, Any},\n    x::Vector{T}, reset::Bool=true, batch::AbstractVector{Int64}=\n    Base.OneTo(problem.num_obs), block::AbstractVector{Int64}=eachindex(x)\n) where T\n\nUpdates the hessian in store[:hess] for the problem specified in problem     using the observations specified in batch. If block is specified      then only the entries of the sub-array, store[:hess][block, block],     are updated. If reset==true, then the entries of store[:hess][block,block]     are first set to zero. Returns nothing.\n\n\n\n\n\n","category":"function"},{"location":"api/glm/#Generalized-Linear-Models","page":"Genearlized Linear Models","title":"Generalized Linear Models","text":"","category":"section"},{"location":"api/glm/","page":"Genearlized Linear Models","title":"Genearlized Linear Models","text":"OptimizationProblems.GeneralizedLinearModel\n\nOptimizationProblems.GLMFamily","category":"page"},{"location":"api/glm/#OptimizationProblems.GeneralizedLinearModel","page":"Genearlized Linear Models","title":"OptimizationProblems.GeneralizedLinearModel","text":"GeneralizedLinearModel{R, F, G<:GLMFamily} <: OptimizationProblem\n\nData for specifying the negative log-likelihood objective function for a     generalized linear model. \n\nFields\n\nname::String, the name of the problem\ncounters::Dict{Symbol, Counter}, a dictionary of symbols that identify a   counter \nnum_param::Int64, dimension of the optimization parameter.\nnum_obs::Int64, the total number of observations.\nresp::R, the responses of the data of type R, which will depend on the type    of model being considered.\nfeat::F, the features or explanatory variables of type F, which will depend    on the way features are stored. \nfamily::G, specifies the GLM family being considered\n\nConstructors\n\nLogisticRegression\n\nFor details, see the docstrings for each function listed. \n\n\n\n\n\n","category":"type"},{"location":"api/glm/#OptimizationProblems.GLMFamily","page":"Genearlized Linear Models","title":"OptimizationProblems.GLMFamily","text":"GLMFamily\n\nAn abstract type specifying the partition function for a GLM.\n\n\n\n\n\n","category":"type"},{"location":"api/glm/#Logistic-Regression-Problem","page":"Genearlized Linear Models","title":"Logistic Regression Problem","text":"","category":"section"},{"location":"api/glm/","page":"Genearlized Linear Models","title":"Genearlized Linear Models","text":"LogisticRegression\n\nOptimizationProblems.Bernoulli","category":"page"},{"location":"api/glm/#OptimizationProblems.LogisticRegression","page":"Genearlized Linear Models","title":"OptimizationProblems.LogisticRegression","text":"LogisticRegression(::Type{T}; num_param::Int64, num_obs::Int64, \n    name::String=\"Logistic Regression\") where T <: Real\n\nConstructs a Logistic Regression Problem where the given number of parameters      is num_param and the number of observations is num_obs. The feature      matrix is a matrix of type T.     It has a column of ones followed by num_param-1 columns of      independent random normal vectors with mean zero and variance      1/num_param. The response vector is a randomly generated BitVector     corresponding to a logistic regression model.      Returns a GeneralizedLinearModel{BitVector, Matrix{T}, Bernoulli}.\n\n\n\n\n\nLogisticRegression(;resp::BitVector, feat::Matrix{T},\n    name::String=\"Logistic Regression\") where T <: Real\n\nConstructs a Logistic Regression Problem with the user-supplied response      vector, resp, and feature matrix, feat.       Returns a GeneralizedLinearModel{BitVector, Matrix{T}, Bernoulli}.\n\n\n\n\n\n","category":"function"},{"location":"api/glm/#OptimizationProblems.Bernoulli","page":"Genearlized Linear Models","title":"OptimizationProblems.Bernoulli","text":"Bernoulli <: GLMFamily\n\nA structure specifying a Bernoulli response GLM.\n\n\n\n\n\n","category":"type"},{"location":"api/glm/#Binomial-Regression-Problem","page":"Genearlized Linear Models","title":"Binomial Regression Problem","text":"","category":"section"},{"location":"api/glm/","page":"Genearlized Linear Models","title":"Genearlized Linear Models","text":"BinomialRegression\n\nOptimizationProblems.Binomial","category":"page"},{"location":"api/glm/#OptimizationProblems.BinomialRegression","page":"Genearlized Linear Models","title":"OptimizationProblems.BinomialRegression","text":"BinomialRegression(::Type{T}; num_param::Int64, num_obs::Int64,\n    max_trials::Int64=100, name::String=\"Binomial Regression\") where T<:Real\n\nConstructs a Binomial Regression problem where the given number of parameters is      num_param and the number of observations is num_obs. The feature      matrix is of type T.     It has a column of ones followed by num_param-1 columns of independent      random normal vectors wih mean zero and variance 1/num_param.      The response is a vector of integer pairs. The first integer in the      pair is a non-negative integer representing the number of successes.     The second integer in the pair is positive integer representing the      number of trials which is randomly selected from 1 to max_trials.     Returns a GeneralizedLinearModel{Vector{Tuple{Int64, Int64}, Matrix{T},      Binomial}.\n\n\n\n\n\nBinomialRegression(;resp::Vector{Tuple{Int64, Int64}}, feat::Matrix{T},\n    name::String=\"Binomial Regression\") where T<:Real\n\nConstructors a Binomial Regression problem with the user-supplied response      vector, resp, and feature matrix, feat.     Each entry of resp should be a pair with the first number indicating      the number of successes and the second number indicating the number of      trials.      Returns a GeneralizedLinearModel{Vector{Tuple{Int64, Int64}, Matrix{T},      Binomial}.\n\n\n\n\n\n","category":"function"},{"location":"api/glm/#OptimizationProblems.Binomial","page":"Genearlized Linear Models","title":"OptimizationProblems.Binomial","text":"Binomial <: GLMFamily\n\nA structure specifying a Binomial response GLM.\n\n\n\n\n\n","category":"type"},{"location":"#Overview","page":"Overview","title":"Overview","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"OptimizationProblems is a research-tier library for the Julia language that provides a set of  optimization problems with a focus on those arising from data science problems.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"The library is still in its infancy and will continue to evolve rapidly.  The documentation has several useful components that are outlined below.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"Manual: the manual pages will help you understand how to use the library to specify   optimization problems, preallocate memory for a problem, and evaluate    objective functions or derivative functions.\nAPI: the API pages will explain individual problems and their functionality in    detail.","category":"page"},{"location":"manual/quick_start/#Quick-Start","page":"Quick Start","title":"Quick Start","text":"","category":"section"},{"location":"manual/quick_start/","page":"Quick Start","title":"Quick Start","text":"The library provides an interface for creating optimization problems, and then  evaluating the objective function, gradient function and Hessian function.","category":"page"},{"location":"manual/quick_start/","page":"Quick Start","title":"Quick Start","text":"The following example creates a random LogisticRegression problem with 10 parameters and 100 observations (features are stored as Float32);  creates a store object to retain evaluated information (stored as Float64);  evaluates the objective function (the negative log-likelihood) at a randomly generated  argument x; and evaluates the gradient function at the same argument.","category":"page"},{"location":"manual/quick_start/","page":"Quick Start","title":"Quick Start","text":"using OptimizationProblems\n\nproblem = LogisticRegression(Float32, num_param=10, num_obs=100)\nstore = allocate(problem, type=Float64)\n\nx = randn(Float64, 10) # Randomly generated argument. \n\nobj!(problem, store=store, x=x)\ngrad!(problem, store=store, x=x)","category":"page"},{"location":"manual/quick_start/","page":"Quick Start","title":"Quick Start","text":"The evaluated objective and gradient are in store[:obj] and store[:grad], respectively.","category":"page"},{"location":"manual/quick_start/","page":"Quick Start","title":"Quick Start","text":"If we have a new argument, z, for which we want to evaluate the objective  and gradient function, we do the following evaluations.","category":"page"},{"location":"manual/quick_start/","page":"Quick Start","title":"Quick Start","text":"z = randn(Float64, 10) # New argument\n\nobj!(problem, store=store, x=z)\ngrad!(problem, store=store, x=z)","category":"page"},{"location":"manual/quick_start/","page":"Quick Start","title":"Quick Start","text":"The evaluated objective and gradient are in store[:obj] and store[:grad], respectively.","category":"page"},{"location":"manual/quick_start/","page":"Quick Start","title":"Quick Start","text":"See the API for examples of problems;  for more information about allocate; and more information about obj!, grad!, objgrad!, and hess!.","category":"page"}]
}
