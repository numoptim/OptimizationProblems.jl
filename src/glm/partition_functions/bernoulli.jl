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

# function allocate(
#     ::T,
#     data::GeneralizedLinearModel{R, F, Bernoulli};
#     gradient::Bool=true, 
#     hessian::Bool=false,
#     weights::Bool=false,
#     residual::Bool=false,
#     weighted_features::Bool=false,
# ) where {T, R, F}

#     num_param = data.num_param
#     num_obs = data.num_obs

#     store = Dict{Symbol, Array}()
#     gradient && setindex!(store, :grad, zeros(T, num_param))
#     hessian && setindex!(store, :hess, zeros(T, num_param, num_param))
#     weights && setindex!(store, :weights, zeros(T, num_obs, num_param))
#     residual && setindex!(store, :residual, zeros(T, num_obs))
#     weighted_features && setindex!(store, :jacobian, zeros(T, num_obs, num_param))

#     return store 
# end

# function allocate(
#     ::T,
#     data::GeneralizedLinearModel{R, F, Bernoulli};
#     num_param::Int64,
#     num_obs::Int64,
#     gradient::Bool=true, 
#     hessian::Bool=false,
#     weights::Bool=false,
#     residual::Bool=false,
#     weighted_features::Bool=false,
# ) where {T, R, F}

#     store = Dict{Symbol, Array}()
#     gradient && setindex!(store, :grad, zeros(T, num_param))
#     hessian && setindex!(store, :hess, zeros(T, num_param, num_param))
#     weights && setindex!(store, :weights, zeros(T, num_obs, num_param))
#     residual && setindex!(store, :residual, zeros(T, num_obs))
#     weighted_features && setindex!(store, :jacobian, zeros(T, num_obs, num_param))

#     return store
# end

# function likelihood(
#     family::Bernoulli;
#     x::Vector{T} where T, 
#     resp::Bool,
#     feat::S where S<:AbstractVector
# )
#     η = dot(x, feat)
#     return T(-resp*η + log(1 + exp(η)))
# end

# function score!(
#     family::Bernoulli;
#     gradient::Vector{T},
#     x::Vector{T},
#     resp::Bool,
#     feat::S where S<:AbstractVector, 
#     params::AbstractVector{Int64}=eachindex(x)
# ) where T
#     η = dot(x, feat)
#     gradient -= (resp - 1/(1+exp(η))) * view(feat, params)
#     return
# end

# function likelihoodscore!(
#     family::Bernoulli;
#     gradient::Vector{T},
#     x::Vector{T},
#     resp::Bool,
#     feat::S where S<:AbstractVector, 
#     params::AbstractVector{Int64}=eachindex(x)
# ) where T
#     η = dot(x, feat)
#     d = 1 + exp(η)
#     gradient -= (resp - 1/d) * view(feat, params)
#     return T(-resp*η + log(d))
# end


# function information!(
#     family::Bernoulli;
#     hessian::Matrix{T},
#     x::Vector{T},
#     resp:Bool,
#     feat::S where S<:AbstractVector,
#     params::AbstractVector{Int64}=eachindex(x)
# ) where T
#     μ = 1/(1 + exp(dot(x, view(feat, i, :))))
#     hessian .+= μ * (1 - μ) * view(feat, params) * transpose(view(feat, params))
#     return
# end

# function partition_sqrt_weights!(
#     family::Bernoulli;
#     sqrt_weights::Vector{T},
#     x::Vector{T},
#     resp::Bool,
#     feat::F where F<:AbstractMatrix
# ) where T
#     for i in eachindex(sqrt_weights)
#         μ = 1/(1 + exp(dot(x, view(feat, i, :))))
#         sqrt_weights[i] = sqrt(μ*(1-μ))
#     end

#     return
# end

# function constant_vector!(
#     resid::T,
#     family::Bernoulli;
#     x::T,
#     resp::R where R<:AbstractVector, 
#     feat::F where F<:AbstractMatrix
# ) where {T<:AbstractVector}
    
#     for i in eachindex(resid)
#         μ = 1/(1 + exp(dot(x, view(feat, i, :))))
#         resid[i] .= (resp[i] - μ) / (sqrt(μ*(1-μ)))
#     end

#     return
# end

# function coefficient_matrix!(
#     w_feat::W where W<:AbstractMatrix,
#     family::Bernoulli;
#     x::T,
#     feat::F where F<:AbstractMatrix,
#     params::Union{Base.OneTo{Int64},Vector{Int64}}=eachindex(x)
# )
#     for i in axes(w_feat, 1)
#         μ = 1/(1 + exp(dot(x, view(feat, i, :))))
#         sqrt_weight = sqrt(μ*(1-μ))
#         w_feat[i,:] .= sqrt_weight*view(feat, i, params)
#     end

#     return
# end