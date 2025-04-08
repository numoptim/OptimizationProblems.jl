"""
    BinomialRegression(::Type{T}; num_param::Int64, num_obs::Int64,
        max_trials::Int64=100, name::String="Binomial Regression") where T<:Real 

Constructs a Binomial Regression problem where the given number of parameters is 
    `num_param` and the number of observations is `num_obs`. The feature 
    matrix is of type `T`.
    It has a column of ones followed by `num_param-1` columns of independent 
    random normal vectors wih mean zero and variance `1/num_param`. 
    The response is a vector of integer pairs. The first integer in the 
    pair is a non-negative integer representing the number of successes.
    The second integer in the pair is positive integer representing the 
    number of trials which is randomly selected from 1 to `max_trials`.
    Returns a `GeneralizedLinearModel{Vector{Tuple{Int64, Int64}, Matrix{T}, 
    Binomial}`.
"""
function BinomialRegression(::Type{T}; num_param::Int64, num_obs::Int64,
    max_trials::Int64=100, name::String="LogisticRegression") where T<:Real

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))

    # Construct Feature 
    feat = num_param > 1 ? hcat(
        ones(T, num_obs),
        randn(T, num_obs, num_param-1) ./ T(sqrt(num_param-1))
    ) : ones(T, num_obs, 1)

    # Allocate Response 
    resp = Vector{Tuple{Int64, Int64}}(undef, num_obs)

    #   Generate Oracle Parameter and Probabilities 
    β = rand(num_param) .- 0.5
    π = 1 ./ (1 .+ exp.(-feat * β))

    #   Sample Binomial to construct response vector 
    for i in Base.OneTo(num_obs)
        n = rand(1:max_trials)
        s = rand(Distributions.Binomial(n, π[i]))
        resp[i] = (s, n)
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
        Binomial()
    )
end

"""
    BinomialRegression(;resp::Vector{Tuple{Int64, Int64}}, feat::Matrix{T},
        name::String="Binomial Regression") where T<:Real

Constructors a Binomial Regression problem with the user-supplied response 
    vector, `resp`, and feature matrix, `feat`.
    Each entry of `resp` should be a pair with the first number indicating 
    the number of successes and the second number indicating the number of 
    trials. 
    Returns a `GeneralizedLinearModel{Vector{Tuple{Int64, Int64}, Matrix{T}, 
    Binomial}`.
"""
function BinomialRegression(;resp::Vector{Tuple{Int64, Int64}}, feat::Matrix{T},
    name::String="Binomial Regression") where T<:Real

    num_obs, num_param = size(feat)

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    num_obs != length(resp) && throw(DimensionMismatch("length of `resp` must\
    equal the number of rows in `feat`."))

    # Validate response data 
    for pair in resp
        # Check non-negative # of successes 
        pair[1] < 0 && throw(DomainError("The first entry of a pair in `resp` must be \
        be a non-negative integer."))

        # Check positive # of trials 
        pair[2] < 1 && throw(DomainError("The second entry of a pair in `resp` must be \
        a positive integer."))

        # Check that number of successes is no greater than number of trials 
        pair[1] > pair[2] && throw(DomainError("The first entry of a pair in `resp` must \
        not exceed the second entry of the pair"))
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
        Binomial()
    )

end