"""
    InverseGaussianRegression(::Type{T}; num_param::Int64, num_obs::Int64,
        name::String="Inverse Gaussian Regression") where T<:Real

Constructs an Inverse Gaussian Regression problem where the given number of
    parameters is `num_param` and the number of observations is `num_obs`.
    The shape parameter `lambda` is fixed at `1`. The feature matrix is a matrix
    of type `T`. It has a column of ones followed by `num_param-1` columns of
    independent random vectors whose entries have uniform distribution
    `(0,1/num_param)`. The response vector is a randomly generated vector of
    type `T` corresponding to the inverse Gaussian regression model.
    Returns a `GeneralizedLinearModel{Vector{T}, Matrix{T}, InverseGaussian}`.

!!! warning
    Under the GLM family, for a feature vector `feat` and parameter vector
    `x`, `dot(feat, x)` must be strictly negative.
"""
function InverseGaussianRegression(
    ::Type{T};
    num_param::Int64,
    num_obs::Int64,
    name::String="Inverse Gaussian Regression"
) where T<:Real

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))

    feat = num_param > 1 ? hcat(
        ones(T, num_obs),
        rand(T, num_obs, num_param-1) ./ T(num_param-1)
    ) : ones(T, num_obs, 1)

    # Linear Effect = feat * x, where feat >= 0 and each weight x lies in
    # (-2/num_param, -1/num_param], so η is strictly negative and never 0
    η = feat * (-(one(T) .+ rand(T, num_param)) ./ num_param)

    # Mean of the inverse Gaussian model, μ = 1 / sqrt(-2η)
    μ = 1 ./ sqrt.(-2 .* η)

    # Sample responses from the inverse Gaussian with shape λ = 1
    resp = Vector{T}(undef, num_obs)
    for i in Base.OneTo(num_obs)
        resp[i] = T(rand(Distributions.InverseGaussian(μ[i], 1.0)))
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
        InverseGaussian()
    )
end


"""
    InverseGaussianRegression(; resp::Vector{T}, feat::Matrix{T},
        name::String="Inverse Gaussian Regression") where T<:Real

Constructs an Inverse Gaussian Regression problem with a user-supplied response
    vector, `resp`, and a feature matrix, `feat`. The shape parameter `lambda`
    is fixed at `1`.
    Returns a `GeneralizedLinearModel{Vector{T}, Matrix{T}, InverseGaussian}`.

!!! warning
    Under the GLM family, for a feature vector `feat` and parameter vector
    `x`, `dot(feat, x)` must be strictly negative.
"""
function InverseGaussianRegression(;
    resp::Vector{T},
    feat::Matrix{T},
    name::String="Inverse Gaussian Regression"
) where T<:Real

    num_obs, num_param = size(feat)

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    num_obs != length(resp) && throw(DimensionMismatch("length of `resp` must \
        equal the number of rows in `feat`."))
    sum(resp .<= 0) > 0 && throw(DomainError("`resp` must be positive valued."))

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
        InverseGaussian()
    )
end

function likelihood(
    family::InverseGaussian;
    x::Vector{T},
    resp::R,
    feat::S where S<:AbstractVector
) where {T<:Real, R<:Real}
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    return T(-resp*η - sqrt(-2*η))
end

function score!(
    family::InverseGaussian;
    gradient::Vector{T},
    x::Vector{T},
    resp::R,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where {T<:Real, R<:Real}
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    μ = 1/sqrt(-2*η)
    view(gradient, params) .-= (resp - μ) * view(feat, params)
    return
end

function likelihoodscore!(
    family::InverseGaussian;
    gradient::Vector{T},
    x::Vector{T},
    resp::R,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where {T<:Real, R<:Real}
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    μ = 1/sqrt(-2*η)
    view(gradient, params) .-= (resp - μ) * view(feat, params)
    return T(-resp*η - sqrt(-2*η))
end

function information!(
    family::InverseGaussian;
    hessian::Matrix{T},
    x::Vector{T},
    resp::R,
    feat::S where S<:AbstractVector,
    params::AbstractVector{Int64}=eachindex(x)
) where {T<:Real, R<:Real}
    η = dot(x, feat)
    η >= 0 && throw(DomainError("Linear effect must be negative"))
    μ = 1/sqrt(-2*η)
    view(hessian, params, params) .+= μ^3 * view(feat, params) *
        transpose(view(feat, params))
    return
end

#TODO: Implement GNN weight, constant and coefficient
