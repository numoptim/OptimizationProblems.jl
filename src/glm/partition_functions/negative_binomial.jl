"""
    NegativeBinomialRegression(
        ::Type{T};
        num_param::Int64,
        num_obs::Int64,
        max_successes::Int64=100,
        name::String="Negative Binomial Regression"
    ) where T<:Real

Constructs a Negative Binomial Regression problem with `num_param` parameters and
    `num_obs` observations.
    The feature matrix has elements of type `T`.
    The feature matrix is a column of ones followed by `num_param-1` columns of
    independent uniform vectors on `(0, 1/(num_param-1))`.
    Each response is a pair of integers, `(failures, successes)`. Think of
    repeating a trial that either fails or succeeds, and stopping once a set
    number of successes is reached. The second integer is that number of
    successes, picked at random from 1 to `max_successes`. The first integer
    is how many failures happened before stopping.
    Returns a `GeneralizedLinearModel{Vector{Tuple{Int64, Int64}}, Matrix{T},
    NegativeBinomial}`.

!!! warning
    Under the GLM family, for a feature vector `feat` and parameter vector
    `x`, `dot(feat, x)` must be strictly negative.
"""
function NegativeBinomialRegression(
    ::Type{T};
    num_param::Int64,
    num_obs::Int64,
    max_successes::Int64=100,
    name::String="Negative Binomial Regression"
) where T<:Real

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    max_successes < 1 && throw(ArgumentError("`max_successes` must be at least one."))

    # Construct Feature (non-negative entries)
    feat = num_param > 1 ? hcat(
        ones(T, num_obs),
        rand(T, num_obs, num_param-1) ./ T(num_param-1)
    ) : ones(T, num_obs, 1)

    # Generate Oracle Parameter (negative → η = feat*β < 0)
    β = -rand(T, num_param) ./ T(num_param)

    # Compute linear effects
    η = feat * β

    # Allocate response pairs (failures, successes)
    resp = Vector{Tuple{Int64, Int64}}(undef, num_obs)
    # Sample: r successes, then failures via Distributions.jl
    for i in Base.OneTo(num_obs)
        r = rand(1:max_successes)
        y = rand(Distributions.NegativeBinomial(r, one(T) - exp(η[i])))
        resp[i] = (y, r)
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
        NegativeBinomial()
    )
end

"""
    NegativeBinomialRegression(
        ; resp::Vector{Tuple{Int64, Int64}},
        feat::Matrix{T},
        name::String="Negative Binomial Regression"
    ) where T<:Real

Constructs a Negative Binomial Regression problem from a user-supplied response vector,
    `resp`, and feature matrix, `feat`.
    Each entry of `resp` is a pair of integers, `(failures, successes)`. The second
    integer is the number of successes to reach, and the first is how many failures
    happened before reaching them. Failures must be non-negative and successes must
    be at least one.
    Returns a `GeneralizedLinearModel{Vector{Tuple{Int64, Int64}}, Matrix{T},
    NegativeBinomial}`.
"""
function NegativeBinomialRegression(;
    resp::Vector{Tuple{Int64, Int64}},
    feat::Matrix{T},
    name::String="Negative Binomial Regression"
) where T<:Real

    num_obs, num_param = size(feat)
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    length(resp) != num_obs && throw(
        DimensionMismatch(
            "`resp` must have the same number of observations as `feat`."
        )
    )
    # Validate response pairs
    for pair in resp
        pair[1] < 0 && throw(DomainError("Number of failures must be non-negative."))
        pair[2] < 1 && throw(DomainError("Number of successes must be at least one."))
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
        NegativeBinomial()
    )
end

function likelihood(
    family::NegativeBinomial;
    x::Vector{T},
    resp::Tuple{Int64, Int64},
    feat::S where S<:AbstractVector
) where T<:Real
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    return T(-resp[1]*η - resp[2] * log(one(T) - exp(η)))
end

function score!(
    family::NegativeBinomial;
    gradient::Vector{T},
    x::Vector{T},
    resp::Tuple{Int64, Int64},
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    eη = exp(η)
    view(gradient, params) .-= (resp[1] - resp[2] * eη/(one(T) - eη)) * view(feat, params)
    return nothing
end

function likelihoodscore!(
    family::NegativeBinomial;
    gradient::Vector{T},
    x::Vector{T},
    resp::Tuple{Int64, Int64},
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    eη = exp(η)
    view(gradient, params) .-= (resp[1] - resp[2] * eη/(one(T) - eη)) * view(feat, params)
    return T(-resp[1]*η - resp[2] * log(one(T) - eη))
end

function information!(
    family::NegativeBinomial;
    hessian::Matrix{T},
    x::Vector{T},
    resp::Tuple{Int64, Int64},
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    eη = exp(η)
    view(hessian, params, params) .+= resp[2] * eη/(one(T) - eη)^2 *
        view(feat, params) * transpose(view(feat, params))
    return nothing
end