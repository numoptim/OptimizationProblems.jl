"""
    LogisticRegression(::Type{T}; num_param::Int64, num_obs::Int64, 
        name::String="Logistic Regression") where T <: Real 

Constructs a Logistic Regression Problem where the given number of parameters 
    is `num_param` and the number of observations is `num_obs`. The feature 
    matrix is a matrix of type `T`.
    It has a column of ones followed by `num_param-1` columns of 
    independent random normal vectors with mean zero and variance 
    `1/num_param`. The response vector is a randomly generated `BitVector`
    corresponding to a logistic regression model. 
    Returns a `GeneralizedLinearModel{BitVector, Matrix{T}, Bernoulli}`.
"""
function LogisticRegression(::Type{T}; num_param::Int64, num_obs::Int64,
        name::String="Logistic Regression") where T <: Real 

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))

    feat = num_param > 1 ? hcat(
        ones(T, num_obs), 
        randn(T, num_obs, num_param-1) ./ T(sqrt(num_param-1))
    ) : ones(T, num_obs, 1)
    β = rand(num_param) .- 0.5
    resp = rand(num_obs) .>= (1 ./ (1 .+ exp.(-feat*β)))

    return GeneralizedLinearModel(
        name,
        Dict{Symbol, Counter}(
            :obj => Counter(block_total=num_param, batch_total=num_obs),
            :grad => Counter(block_total=num_param, batch_total=num_obs),
            :hess => Counter(block_total=num_param, batch_total=num_obs),
            :residual => Counter(block_total=num_param, batch_total=num_obs),
            :jacobian => Counter(block_total=num_param, batch_total=num_obs)
        ),
        num_param,
        num_obs, 
        resp, 
        feat, 
        Bernoulli()
    )
end

"""
    LogisticRegression(;resp::BitVector, feat::Matrix{T},
        name::String="Logistic Regression") where T <: Real 

Constructs a Logistic Regression Problem with the user-supplied response 
    vector, `resp`, and feature matrix, `feat`.  
    Returns a `GeneralizedLinearModel{BitVector, Matrix{T}, Bernoulli}`.
"""
function LogisticRegression(;resp::BitVector, feat::Matrix{T},
    name::String="Logistic Regression") where T <: Real
    
    num_obs, num_param = size(feat)

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    num_obs != length(resp) && throw(DimensionMismatch("length of `resp` must\
    equal the number of rows in `feat`."))

    return GeneralizedLinearModel(
        name,
        Dict{Symbol, Counter}(
            :obj => Counter(block_total=num_param, batch_total=num_obs),
            :grad => Counter(block_total=num_param, batch_total=num_obs),
            :hess => Counter(block_total=num_param, batch_total=num_obs),
            :residual => Counter(block_total=num_param, batch_total=num_obs),
            :jacobian => Counter(block_total=num_param, batch_total=num_obs)
        ),
        num_param,
        num_obs,
        resp,
        feat,
        Bernoulli()
    )
end

function likelihood(
    family::Bernoulli;
    x::Vector{T}, 
    resp::Bool,
    feat::S where S<:AbstractVector
) where T<:Real
    η = dot(x, feat)
    return T(-resp*η + log(1 + exp(η)))
end

function score!(
    family::Bernoulli;
    gradient::Vector{T},
    x::Vector{T},
    resp::Bool,
    feat::S where S<:AbstractVector, 
    params::AbstractVector{Int64}=eachindex(x)
) where T
    η = dot(x, feat)
    view(gradient, params)  .-= (resp - 1/(1+exp(-η))) * view(feat, params)
    return
end

function likelihoodscore!(
    family::Bernoulli;
    gradient::Vector{T},
    x::Vector{T},
    resp::Bool,
    feat::S where S<:AbstractVector, 
    params::AbstractVector{Int64}=eachindex(x)
) where T
    η = dot(x, feat)
    view(gradient, params) .-= (resp - 1/(1+exp(-η))) * view(feat, params)
    return T(-resp*η + log(1+exp(η)))
end

function information!(
    family::Bernoulli;
    hessian::Matrix{T},
    x::Vector{T},
    resp::Bool,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T
    μ = 1/(1 + exp(dot(x, feat)))
    view(hessian, params, params) .+= μ * (1 - μ) * view(feat, params) * 
        transpose(view(feat, params))
    return
end

function gnn_weight(
    family::Bernoulli;
    x::Vector{T},
    resp::Bool,
    feat::S where S<:AbstractVector
) where T
    μ = 1/(1 + exp(dot(x, feat)))
    return sqrt(μ*(1-μ))
end

function gnn_constant(
    family::Bernoulli;
    x::Vector{T},
    resp::Bool,
    feat::S where S<:AbstractVector,
    weight::T
) where T
    μ = 1/(1 + exp(-dot(x, feat)))
    return (resp - μ) / weight
end


function gnn_coefficient!(
    family::Bernoulli;
    weighted_feat::S,
    x::Vector{T},
    feat::S,
    weight::T,
    params::AbstractVector{Int64}=eachindex(x)
) where {S<:AbstractVector,T}
    view(weighted_feat, params) .= weight .* view(feat, params)
    return nothing 
end