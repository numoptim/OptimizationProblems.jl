"""
    GLMFamily

An abstract type specifying the partition function for a GLM.
"""
abstract type GLMFamily end 

"""
    Bernoulli <: GLMFamily

A structure specifying a Bernoulli response GLM.
"""
struct Bernoulli <: GLMFamily end 
#struct Binomial <: GLMFamily end 
#struct Exponential <: GLMFamily end 
#struct MultinomialNominal <: GLMFamily end 
#struct MultinomialOrdinal <: GLMFamily end 
#struct NegativeBinomial <: GLMFamily end 
#struct Normal <: GLMFamily end 
#struct Poisson <: GLMFamily end 

"""
    GeneralizedLinearModel{R, F, G<:GLMFamily} <: OptimizationProblem

Data for specifying the negative log-likelihood objective function for a
    generalized linear model. 

# Fields
- `name::String`, the name of the problem
- `counters::Dict{Symbol, Counter}`, a dictionary of symbols that identify a
    counter 
- `num_param::Int64`, dimension of the optimization parameter.
- `num_obs::Int64`, the total number of observations.
- `resp::R`, the responses of the data of type `R`, which will depend on the type 
    of model being considered.
- `feat::F`, the features or explanatory variables of type `F`, which will depend 
    on the way features are stored. 
- `family::G`, specifies the GLM family being considered

# Constructors 

    LogisticRegression

For details, see the docstrings for each function listed. 
"""
struct GeneralizedLinearModel{R, F, G<:GLMFamily} <: OptimizationProblem
    name::String
    counters::Dict{Symbol, Counter}
    num_param::Int64
    num_obs::Int64
    resp::R
    feat::F
    family::G
end

# Logistic Regression 
include("partition_functions/bernoulli.jl")


function obj(
    x::Vector{T}; 
    problem::GeneralizedLinearModel, 
    store::Dict{Symbol, Array},
    batch::AbstractVector{Int64}=Base.OneTo(problem.num_obs)
) where T
    
    # Increment Objective Counter  
    increment!(problem.counter[:obj])  
    
    # Compute Objective 
    o = 0
    for i in Base.OneTo(problem.num_obs)
        o += likelihood(problem.family, x=x, resp=problem.resp[i],
            feat=view(problem.feat, i, :))
    end

    return o
end

function grad!(
    x::Vector{T};
    problem::GeneralizedLinearModel,
    store::Dict{Symbol, Array},
    params::AbstractVector{Int64}=eachindex(x)
) where T
    
    # Increment Gradient Counter(s)
    params == Base.OneTo(problem.num_param) ?
        increment!(problem.counter[:grad]) :
        increment!(problem.counter[:grad_inc_param], size=length(params))
    
    # Compute Gradient 
    fill!(store[:grad], T(0.0))
    for i in Base.OneTo(problem.num_obs)
        score!(problem.family, gradient=store[:grad], x=x, 
            resp=problem.resp[i], feat=view(problem.feat, i, :),
            params=params)
    end

    return 
end

#TODO: objgrad!
#TODO: hess!
#TODO: jacobian!
#TODO: residual?