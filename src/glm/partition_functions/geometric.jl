"""
    GeometricRegression(::Type{T}; num_param::Int64, num_obs::Int64,
        name::String="Geometric Regression") where T<:Real

Constructs a Geometric Regression problem with `num_param` parameters and
    `num_obs` observations.
    The feature matrix has elements of type `T`.
    The feature matrix is a column of ones followed by `num_param-1` columns of
    independent uniform vectors on `(0, 1/(num_param-1))`.
    The response is a vector of type `Int64` with non-negative values.
    Returns a `GeneralizedLinearModel{Vector{Int64}, Matrix{T}, Geometric}`.

!!! warn
    Under the GLM family, for a feature vector `feat` and parameter vector
    `x`, `dot(feat, x)` must be strictly negative.
"""
function GeometricRegression(::Type{T}; num_param::Int64, num_obs::Int64,
    name::String="Geometric Regression") where T<:Real

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))

    # Construct Feature (non-negative entries)
    feat = num_param > 1 ? hcat(
        ones(T, num_obs),
        rand(T, num_obs, num_param-1) ./ T(num_param-1)
    ) : ones(T, num_obs, 1)

    # Generate Oracle Parameter (negative → η = feat*β < 0)
    β = -rand(T, num_param) ./ T(num_param)

    # Compute linear effects
    η = feat * β

    # Sample via inverse CDF: Y = floor(log(1-U) / η)
    resp = floor.(Int64, log.(one(T) .- rand(T, num_obs)) ./ η)

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
        Geometric()
    )
end

"""
    GeometricRegression(; resp::Vector{Int64}, feat::Matrix{T},
        name::String="Geometric Regression") where T<:Real

Constructs a Geometric Regression problem with a user-supplied response vector,
    `resp`, and feature matrix, `feat`.
    Returns a `GeneralizedLinearModel{Vector{Int64}, Matrix{T}, Geometric}`.
"""
function GeometricRegression(; resp::Vector{Int64}, feat::Matrix{T},
    name::String="Geometric Regression") where T<:Real

    num_obs, num_param = size(feat)
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    length(resp) != num_obs && throw(
        DimensionMismatch(
            "`resp` must have the same number of observations as `feat`."
        )
    )

    # Check response vector is non-negative
    sum(resp .< 0) > 0 && throw(DomainError("`resp` must be non-negative."))

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
        Geometric()
    )
end

function likelihood(
    family::Geometric;
    x::Vector{T},
    resp::Int64,
    feat::S where S<:AbstractVector
) where T<:Real
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    return T(-resp*η - log(one(T) - exp(η)))
end

function score!(
    family::Geometric;
    gradient::Vector{T},
    x::Vector{T},
    resp::Int64,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    eη = exp(η)
    view(gradient, params) .-= (resp - eη/(one(T) - eη)) * view(feat, params)
    return nothing
end

function likelihoodscore!(
    family::Geometric;
    gradient::Vector{T},
    x::Vector{T},
    resp::Int64,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    eη = exp(η)
    view(gradient, params) .-= (resp - eη/(one(T) - eη)) * view(feat, params)
    return T(-resp*η - log(one(T) - eη))
end

function information!(
    family::Geometric;
    hessian::Matrix{T},
    x::Vector{T},
    resp::Int64,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    eη = exp(η)
    view(hessian, params, params) .+= eη/(one(T) - eη)^2 *
        view(feat, params) * transpose(view(feat, params))
    return nothing
end