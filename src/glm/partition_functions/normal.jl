"""
    LinearRegression(::Type{T}; num_param::Int64, num_obs::Int64,
        σ::T=T(1), name::String="Linear Regression") where T<:Real

Constructs a Linear Regression problem where the given number of parameters
    `num_params` and the number of observations is `num_obs`.
    The feature matrix has elements of type `T`.
    The feature matrix is a column of ones followed by `num_param-1` columns 
    of independent normal vectors with mean zero and variance `1/(num_param-1)`.
    The response is a vector of type `T`. 
    The value of `σ` specifies the standard deviation of the responses. 
    Returns a `GeneralizedLinearModel{Vector{T}, Matrix{T},Normal}`.
"""
function LinearRegression(::Type{T}; num_param::Int64, num_obs::Int64, 
    σ::T=T(1), name::String="Linear Regression") where T<:Real 

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))

    # Construct Feature 
    feat = num_param > 1 ? hcat(
        ones(T, num_obs),
        randn(T, num_obs, num_param-1) ./ T(sqrt(num_param-1))
    ) : ones(T, num_obs, 1)

    # Generate Oracle Parameter 
    β = rand(T, num_param) .- T(0.5) 

    # Generate Responses 
    resp = feat*β + σ*randn(T, num_obs)

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
        Normal()
    )
end

"""
    LinearRegression(;resp::Vector{T}, feat::Matrix{T}, 
        name::String="Linear Regression") where T<:Real

Constructs a Linear Regression problem with a user-supplied response vector, 
    `resp`, and feature matrix, `feat`. 
    Returns a `GeneralizedLinearModel{Vector{T}, Matrix{T}, Normal}`.
"""
function LinearRegression(;resp::Vector{T}, feat::Matrix{T}, 
    name::String="Linear Regression") where T<:Real

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
        Normal()
    )
end

function likelihood(
    family::Normal;
    x::Vector{T},
    resp::R where R<:Real,
    feat::S where S<:AbstractVector
) where T<:Real

    η = dot(x, feat)
    return T((resp - η)^2 / 2)
end 

function score!(
    family::Normal;
    gradient::Vector{T},
    x::Vector{T},
    resp::R where R<:Real,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    η = dot(x, feat)
    # -(y - x'feat) * feat 
    view(gradient, params) .-= (resp - η) * view(feat, params)
    return nothing 
end

function likelihoodscore!(
    family::Normal;
    gradient::Vector{T},
    x::Vector{T},
    resp::R where R<:Real,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    η = dot(x, feat)
    view(gradient, params) .-= (resp - η) * view(feat, params)
    return T((resp - η)^2 / 2)
end

function information!(
    family::Normal;
    hessian::Matrix{T},
    x::Vector{T},
    resp::R where R<:Real,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    view(hessian, params, params) .+= view(feat, params) * transpose(view(feat, params))
    return nothing 
end