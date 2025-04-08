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
struct Binomial <: GLMFamily end 
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

# Specific GLM Families 
include("partition_functions/bernoulli.jl") # Bernoulli 
include("partition_functions/binomial.jl") # Binomial 


# Preallocation
"""
    allocate(
        problem::GeneralizedLinearModel;
        type::DataType=Float64,
        obj::Bool=true, 
        grad::Bool=true,
        hess::Bool=false,
        weights::Bool=false,
        residual::Bool=false,
        jacobian::Bool=false,
    )

Allocates storage necessary in a `Dict` for calculating objective information to reduce 
    allocations during runtime. 

# Arguments 
- `problem::GeneralizedLinearModel`, specifies the problem that is being considered
- `type::DataType`, specifies the target `DataType` for all calculations. Defaults to  
    `Float64`.
- `obj::Bool`, if `true` adds a key `:obj` to the storage dictionary with value `type(0.0)`.
    Otherwise, does nothing to the storage dictionary. 
- `grad::Bool`, if `true` adds a key `:grad` to the storage dictionary with value 
    `zeros(type, problem.num_param)`. Otherwise, does nothing to storage dictionary. 
- `hess::Bool`, if `true` adds a key `:hess` to the storage dictionary with value 
    `zeros(type, problem.num_param, problem.num_param)`. Otherwise, does nothing to 
    storage dictionary. 
- `weights::Bool`, if `true` adds a key `:weights` to the storage dictionary with value 
    `zeros(type, problem.num_obs)`. Otherwise, does nothing to storage dictionary. 
- `residual::Bool`, if `true` adds a key `:residual` to the storage dictionary with value 
    `zeros(type, problem.num_obs)`. Otherwise, does nothing to storage dictionary.
- `jacobian::Bool`, if `true` adds a key `:jacobian` to the storage dictionary with value 
    `zeros(type, problem.num_obs, num_param)`. Otherwise, does nothign to storage
    dictionary. 

# Returns 
- Object of type `Dict{Symbol, Any}`.
"""
function allocate(
    problem::GeneralizedLinearModel;
    type::DataType=Float64,
    obj::Bool=true, 
    grad::Bool=true,
    hess::Bool=false,
    weights::Bool=false,
    residual::Bool=false,
    jacobian::Bool=false,
)

    # Initialize Empty Store 
    store = Dict{Symbol,Any}()

    # For each object to be stored, add to store 
    obj && push!(store, :obj=> type(0.0))
    grad && push!(store, :grad=> zeros(type, problem.num_param))
    hess && push!(store, :hess=> zeros(type, problem.num_param, problem.num_param))
    weights && push!(store, :weights=> zeros(type, problem.num_obs))
    residual && push!(store, :residual=> zeros(type, problem.num_obs))
    jacobian && push!(store, :jacobian=> zeros(type, problem.num_obs, problem.num_param))

    return store 
end

"""
    obj!(problem::GeneralizedLinearModel; store::Dict{Symbol, Any},
        x::Vector{T}, reset::Bool=true, batch::AbstractVector{Int64}=
        Base.OneTo(problem.num_obs)
    ) where T

Updates the value of the objective function stored in `store[:obj]` for 
    the problem specified in `problem` using the observations specified in 
    `batch`. If `reset` is `true`, then the value of the objective is 
    first set to zero. Returns `nothing`.
"""
function obj!(
    problem::GeneralizedLinearModel;
    store::Dict{Symbol, Any},
    x::Vector{T},
    reset::Bool=true, 
    batch::AbstractVector{Int64}=Base.OneTo(problem.num_obs)
) where T
    
    # Increment Objective Counter  
    increment_batch!(problem.counters[:obj], size=length(batch)) 
    increment_block!(problem.counters[:obj], size=problem.num_param) 
    
    # Compute Objective 
    reset && (store[:obj] = T(0.0))
    for i in batch
        store[:obj] += likelihood(problem.family, x=x, resp=problem.resp[i],
            feat=view(problem.feat, i, :))
    end

    return nothing 
end

"""
    grad!(problem::GeneralizedLinearModel; store::Dict{Symbol, Any},
        x::Vector{T}, reset::Bool=true, batch::AbstractVector{Int64}=
        Base.OneTo(problem.num_obs), block::AbstractVector{Int64}=eachindex(x)
    ) where T

Updates the value of the gradient function stored in `store[:grad]` for the 
    problem specified in `problem` using the observation specified in 
    `batch` and only updates the entries in `block`. If `reset==true`,
    sets `store[:grad][block]` to zero. Returns `nothing`.
"""
function grad!(
    problem::GeneralizedLinearModel;
    store::Dict{Symbol, Any},
    x::Vector{T},
    reset::Bool=true,
    batch::AbstractVector{Int64}=Base.OneTo(problem.num_obs),
    block::AbstractVector{Int64}=eachindex(x)
) where T
    
    # Increment Gradient Counter(s)
    increment_batch!(problem.counters[:grad], size=length(batch))
    increment_block!(problem.counters[:grad], size=length(block))
    
    # Compute Gradient 
    reset && fill!(view(store[:grad], block), T(0.0))
    for i in batch
        score!(problem.family, gradient=store[:grad], x=x, 
            resp=problem.resp[i], feat=view(problem.feat, i, :),
            params=block)
    end

    return nothing
end

"""
    objgrad!(problem::GeneralizedLinearModel; store::Dict{Symbol, Any},
        x::Vector{T}, reset::Bool=true, batch::AbstractVector{Int64}=
        Base.OneTo(problem.num_obs), block::AbstractVector{Int64}=eachindex(x)
    ) where T

Updates the objective value and gradient value in `store` for the problem 
    specified in `problem` using the observations specified in `batch`.
    If `block` is specified, then, for the gradient update, 
    only the values in `store[:grad][block]`. `block` does not impact the 
    objective update. If `reset==true`, then the objective and 
    `store[:grad][block]` are set to zeros. Returns `nothing`.
"""
function objgrad!(
    problem::GeneralizedLinearModel;
    store::Dict{Symbol, Any},
    x::Vector{T},
    reset::Bool=true,
    batch::AbstractVector{Int64}=Base.OneTo(problem.num_obs),
    block::AbstractVector{Int64}=eachindex(x)
) where T

    # Increment Objective and Gradient Counters 
    increment_batch!(problem.counters[:obj], size=length(batch))
    increment_block!(problem.counters[:obj], size=problem.num_param)
    increment_batch!(problem.counters[:grad], size=length(batch))
    increment_block!(problem.counters[:grad], size=length(block))

    # Reset 
    if reset
        store[:obj] = T(0.0)
        fill!(view(store[:grad], block), T(0.0))
    end

    # Compute objective and gradient
    for i in batch
        store[:obj] += likelihoodscore!(problem.family, gradient=store[:grad],
            x=x, resp=problem.resp[i], feat=view(problem.feat, i, :), 
            params=block)
    end

    return nothing
end


"""
    hess!(problem::GeneralizedLinearModel, store::Dict{Symbol, Any},
        x::Vector{T}, reset::Bool=true, batch::AbstractVector{Int64}=
        Base.OneTo(problem.num_obs), block::AbstractVector{Int64}=eachindex(x)
    ) where T

Updates the hessian in `store[:hess]` for the problem specified in `problem`
    using the observations specified in `batch`. If `block` is specified 
    then only the entries of the sub-array, `store[:hess][block, block]`,
    are updated. If `reset==true`, then the entries of `store[:hess][block,block]`
    are first set to zero. Returns `nothing`.
"""
function hess!(
    problem::GeneralizedLinearModel;
    store::Dict{Symbol, Any},
    x::Vector{T},
    reset::Bool=true,
    batch::AbstractVector{Int64}=Base.OneTo(problem.num_obs),
    block::AbstractVector{Int64}=eachindex(x)
) where T

    # Increment Hessian Counters 
    increment_batch!(problem.counters[:hess], size=length(batch))
    increment_block!(problem.counters[:hess], size=length(block))

    # Compute hessian 
    reset && fill!(view(store[:hess], block, block), T(0.0))
    for i in batch 
        information!(problem.family, hessian=store[:hess], x=x,
            resp=problem.resp[i], feat=view(problem.feat, i, :),
            params=block 
        )
    end

    return nothing 
end

#TODO: jacobian!
#TODO: residual?