######################################
# Link Model 
######################################
"""
    LinkModel

An abstract type for link models for a Wedderburn Quasi-likelihood Model.
"""
abstract type LinkModel end


struct Identity <: LinkModel end 
struct InvCompLogLog <: LinkFunction end 
struct Logistic <: LinkModel end 

######################################
# Variance Model  
######################################
"""
    VarianceModel 

An abstract type for variance models for a Wedderburn Quasi-likelihood Model.
"""
abstract type VarianceModel end

#struct CenteredExp <: VarianceModel end
#struct ShiftedCenteredLog <: VarianceModel end
struct ShiftedMonmial{T<:Real} <: VarianceModel 
    p::T
    c::T
end
#struct ShiftedPlusSine <: VarianceModel end


######################################
# Wedderburn Model  
######################################
"""
    WedderburnModel{R, F, G<:GLMFamily, L<:LinkFunction, V<:VarianceModel} <: 
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
struct WedderburnModel{R, F, G<:GLMFamily, L<:LinkFunction, V<:VarianceModel} <: 
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

######################################
# Link and Variance Model  
######################################

######################################
# Preallocation 
######################################

######################################
# Evaluations  
######################################
