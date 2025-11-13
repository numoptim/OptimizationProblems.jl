######################################
# Link Model 
######################################
"""
    LinkModel

An abstract type for link models for a Wedderburn Quasi-likelihood Model.
"""
abstract type LinkModel end


struct Identity <: LinkModel end 
struct InvCompLogLog <: LinkModel end 
struct Logistic <: LinkModel end 

######################################
# Variance Model  
######################################
"""
    VarianceModel 

An abstract type for variance models for a Wedderburn Quasi-likelihood Model.
"""
abstract type VarianceModel end


struct ShiftedMonmial{T<:Real} <: VarianceModel 
    p::T #Power 
    c::T #Constant 
end
struct Sinusoidal{T<:Real} <: VarianceModel end


######################################
# Wedderburn Model  
######################################
"""
    WedderburnModel{R, F, L<:LinkFunction, V<:VarianceModel} <: 
    OptimizationProblem

Data for specifying the optimization problem for estimating Wedderburn's 
    Quasi-likelihood family of models. 

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
- `link::L`, the link function for the model 
- `variance::V`, the variance function for the model 
- `integrator::Function`, the numerical integrator for the model
"""
struct WedderburnModel{R, F, L<:LinkFunction, V<:VarianceModel} <: 
    OptimizationProblem

    name::String
    counters::Dict{Symbol, Counter}
    num_param::Int64
    num_obs::Int64
    resp::R
    feat::F
    link::L
    variance::V
    integrator::Function 
end

###############################################
# Import Link and Variance Model Functionality
###############################################
include("link_models/identity.jl")
include("link_models/invcomploglog.jl")
include("link_models/logistic.jl")

include("variance_models/shiftedmonomial.jl")
include("variance_models/sinusoidal.jl")

######################################
# Preallocation 
######################################

function allocate(
    problem::WedderburnModel;
    type::DataType=Float64,
    obj::Bool=true,
    grad::Bool=true,
    hess::Bool=false,
    weights::Boool=false,
    residual::Bool=false,
    jacobian::Bool=false,
)

    # Initialize 
    store = Dict{Symbol,Any}()

    # For each object to be stored, add to store 
    obj && push!(store, :obj => type(0.0))
    grad && push!(store, :grad => zeros(type, problem.num_param))
    hess && push!(store, :hess => zeros(type, problem.num_param, problem.num_param))
    weights && push!(store, :weights=> zeros(type, problem.num_obs))
    residual && push!(store, :residual=> zeros(type, problem.num_obs))
    jacobian && push!(store, :jacobian=> zeros(type, problem.num_obs, problem.num_param))

    return store  
end

######################################
# Evaluations  
######################################

"""
#TODO: Docstrings and tests
"""
function obj!(
    problem::WedderburnModel;
    store::Dict{Symbol, Any},
    x::Vector{T},
    reset::Bool=true,
    batch::AbstractVector{Int64}=Base.OneTo(problem.num_obs)
) where T

    # Increment Objective Counters 
    increment_batch!(problem.counters[:obj], size=length(batch))
    increment_block!(problem.counters[:obj], size=problem.num_param)

    # Compute Objective 
    reset && (store[:obj] = T(0.0))
    for i in batch 
        η = dot(x, view(problem.feat, i, :))
        μ = link(problem.link, η)
        y = problem.response[i]
        store[:obj] -= problem.integrator(
            m -> (y - m)/ var(problem.variance, m),
            y,
            μ
        )
    end

    return nothing
end

"""
#TODO: Docstrings and tests 
"""
function grad!(
    problem::WedderburnModel;
    store::Dict{Symbol, Any},
    x::Vector{T},
    reset::Bool=true,
    batch::AbstractVector{Int64}=Base.OneTo(problem.num_obs),
    block::AbstractVector{Int64}=eachindex(x)
) where T 

    # Increment Gradient Counters
    increment_batch!(problem.counters[:grad], size=length(batch))
    increment_block!(problem.counters[:grad], size=length(block))

    # Compute Gradient 
    reset && fill!(view(store[:grad], block), T(0.0))
    for i in batch 
        η = dot(x, view(problem.feat, i, :))
        μ = link(problem.link, η)
        ∂μ = der_link(problem.link, η)
        v = var(problem.variance, μ)
        view(store[:grad], block) .-= ((problem.resp[i] - μ) * ∂μ / v) * 
            view(problem.feat, i, block)
    end

    return nothing
end