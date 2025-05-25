"""
    PoissonRegression(::Type{T}; num_param::Int64, num_obs::Int64, 
        name::String="Poisson Regression") where T<:Real
    
Constructs a Poisson Regression problem with the `num_param` parameters and
    `num_obs` observations.
    The feature matrix has elements of type `T`.
    The feature matrix is a column of ones followed by `num_param-1` columns of 
    independent normal vectors with mean zero and variance `1/(num_param-1)`.
    The response is a vector of type `Int64` with non-negative values.
    Returns a `GeneralizedLinearModel{Vector{Int64}, Matrix{T}, Poisson}`.
"""
function PoissonRegression(::Type{T}; num_param::Int64, num_obs::Int64, 
    name::String="Poisson Regression") where T<:Real

    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))

    # Construct Feature 
    feat = num_param > 1 ? hcat(
        ones(T, num_obs),
        randn(T, num_obs, num_param-1) ./ T(sqrt(num_param-1))
    ) : ones(T, num_obs, 1)

    # Generate Oracle Parameter 
    β = rand(T, num_param) .- T(0.5) 

    # Generate Responses via Inverse CDF Sampling Method
    λ = exp.(feat*β)
    resp = zeros(Int64, num_obs)
    
    # For each uniform RV, find the index for which the CDF exceeds the 
    # uniform RV 
    for i in 1:num_obs
        ceiling = min(20, round(Int64, 2 * λ[i]))
        ps = cumsum(exp(-λ[i]) .* [λ[i]^k / factorial(k) for k in 0:ceiling])
        var"Y+1" = findfirst(x -> x > rand(), ps)
        resp[i] = isnothing(var"Y+1") ? ceiling : (var"Y+1" - 1)
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
        Poisson()
    )
end

"""
    PoissonRegression(;resp::Vector{Int64}, feat::Matrix{T}, 
        name::String="Poisson Regression") where T<:Real
    
Constructs a Poisson Regression problem with a user-supplied response vector,
    `resp`, and feature matrix, `feat`. 
    Returns a `GeneralizedLinearModel{Vector{Int64}, Matrix{T}, Poisson}`.
"""
function PoissonRegression(;resp::Vector{Int64}, feat::Matrix{T}, 
    name::String="Poisson Regression") where T<:Real

    num_obs, num_param = size(feat)
    num_obs < 1 && throw(ArgumentError("`num_obs` must be at least one."))
    num_param < 1 && throw(ArgumentError("`num_param` must be at least one."))

    # Check response vector is non-negative
    sum(resp .< 0) > 0 && throw(ArgumentError("`resp` must be non-negative."))

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
        Poisson()
    )
end