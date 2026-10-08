"""
    NegativeBinomialRegression(
        ::Type{T};
        num_param::Int64,
        num_obs::Int64,
        r::Int64=1,
        name::String="Negative Binomial Regression"
) where T<:Real

Constructs a Negative Binomial Regression problem with `num_param` parameters,
    `num_obs` observations, and number of successes until we stop counting failures `r`.
    Returns a `GeneralizedLinearModel{Vector{Int64}, Matrix{T}, NegativeBinomial}`.
"""
function NegativeBinomialRegression(
    ::Type{T};
    num_param::Int64,
    num_obs::Int64,
    r::Int64=1, # number of successes until we stop counting failures
    name::String="Negative Binomial Regression"
) where T<:Real

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    r < 1 && throw(ArgumentError("`r` must be at least one."))

    # Construct Feature (non-negative entries)
    feat = num_param > 1 ? hcat(
        ones(T, num_obs),
        rand(T, num_obs, num_param-1) ./ T(num_param-1)
    ) : ones(T, num_obs, 1)

    # Generate Oracle Parameter (negative → η = feat*β < 0)
    β = -rand(T, num_param) ./ T(num_param)

    # Compute linear effects
    η = feat * β

    # Generate response
    resp = zeros(Int64, num_obs)
    for i in 1:r
        # Add one Geometric draw via inverse CDF: Y = floor(log(1-U) / η)
        resp .+= floor.(Int64, log.(one(T) .- rand(T, num_obs)) ./ η)
    end

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
        NegativeBinomial(r)
    )
end

"""
    NegativeBinomialRegression(
        ; resp::Vector{Int64},
        feat::Matrix{T},
        r::Int64=1,
        name::String="Negative Binomial Regression"
    ) where T<:Real

Constructs a Negative Binomial Regression problem with a user-supplied response vector,
    `resp`, feature matrix, `feat`, and number of successes until we stop counting
    failures `r`.
    Returns a `GeneralizedLinearModel{Vector{Int64}, Matrix{T}, NegativeBinomial}`.
"""
function NegativeBinomialRegression(
    ; resp::Vector{Int64},
    feat::Matrix{T},
    r::Int64=1,
    name::String="Negative Binomial Regression"
) where T<:Real

    num_obs, num_param = size(feat)
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    r < 1 && throw(ArgumentError("`r` must be at least one."))
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
        NegativeBinomial(r)
    )
end

function likelihood(
    family::NegativeBinomial;
    x::Vector{T},
    resp::Int64,
    feat::S where S<:AbstractVector
) where T<:Real
    r = family.r
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    return T(-resp*η - r * log(one(T) - exp(η)))
end

function score!(
    family::NegativeBinomial;
    gradient::Vector{T},
    x::Vector{T},
    resp::Int64,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    r = family.r
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    eη = exp(η)
    view(gradient, params) .-= (resp - r * eη/(one(T) - eη)) * view(feat, params)
    return nothing
end

function likelihoodscore!(
    family::NegativeBinomial;
    gradient::Vector{T},
    x::Vector{T},
    resp::Int64,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    r = family.r
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    eη = exp(η)
    view(gradient, params) .-= (resp - r * eη/(one(T) - eη)) * view(feat, params)
    return T(-resp*η - r * log(one(T) - eη))
end

function information!(
    family::NegativeBinomial;
    hessian::Matrix{T},
    x::Vector{T},
    resp::Int64,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    r = family.r
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    eη = exp(η)
    view(hessian, params, params) .+= r * eη/(one(T) - eη)^2 *
        view(feat, params) * transpose(view(feat, params))
    return nothing
end