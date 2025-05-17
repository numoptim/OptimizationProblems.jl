"""
    ExponentialRegression(::Type{T}; num_param::Int64, num_obs::Int64,
        name::String="Exponential Regression") where T<:Real 

Constructs an Exponential Regression problem where the given number of parameters 
    is `num_param` and the number of observations is `num_obs`.
    The feature matrix is a matrix of type `T`.
    It has a column of ones followed by `num_param-1` columns of independent 
    random vectors whose entries have uniform distirbution `(0,1/num_param)`.
    The response vector is a randomly generated vector of type `T`
    corresponding to the exponential regression model. 
    Returns a `GeneralizedLinearModel{Vector{T}, Matrix{T}, Exponential}`.

!!! warn
    Under the GLM family, for a feature vector `feat` and parameter vector 
    `x`, `dot(feat, x)` must be non-positive.
"""
function ExponentialRegression(
    ::Type{T};
    num_param::Int64, 
    num_obs::Int64,
    name::String="Exponential Regression"
) where T<:Real 

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))

    feat = num_param > 1 ? hcat(
        ones(T, num_obs), 
        rand(T, num_obs, num_param-1) ./ T(num_param-1)
    ) : ones(T, num_obs, 1)

    # Linear Effect =  feat * x, where feat >= 0 and x <= 0
    η = feat * (-rand(T, num_param) ./ num_param)
    
    # Sample using inversion of CDF 
    resp = log.(1 .- rand(T, num_obs)) ./ η

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
        Exponential()
    )
end


"""
    ExponentialRegression(; resp::Vector{T}, feat::Matrix{T},
        name::String="Exponential Regression") where T<:Real

Constructs an Exponential Regression problem witha  user-supplied response vector, 
    `resp`, and a feature matrix, `feat`.
    Returns a `GeneralizedLinearModel{Vector{T}, Matrix{T}, Exponential}`.
"""
function ExponentialRegression(;
    resp::Vector{T},
    feat::Matrix{T},
    name::String="Exponential Regression"
) where T<:Real

    num_obs, num_param = size(feat)

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    num_obs != length(resp) && throw(DimensionMismatch("length of `resp` must \
        equal the number of rows in `feat`."))
    sum(resp .< 0) > 0 && throw(DomainError("`resp` must be non-negative valued."))

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
        Exponential()
    )
end

#TODO: Test Constructors 
#TODO: Partition Function Implementations
#TODO: Test Partition Function