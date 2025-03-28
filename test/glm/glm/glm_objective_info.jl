module TestGLMObjectiveInformation

using Test, OptimizationProblems, Random, LinearAlgebra
using OptimizationProblems.OptimizationModels
import OptimizationProblems: likelihood, score!, likelihoodscore!, information!

######################################################
# Test Context
######################################################
struct TestFamily <: OptimizationProblems.GLMFamily end 

# likelihood returns 1
function likelihood(family::TestFamily;
    x::Vector{T},
    resp::Float64,
    feat::S where S<:AbstractVector
) where T<:Real
    return T(1)
end

# gradient[params] should be increased by value of their index 
function score!(family::TestFamily;
    gradient::Vector{T},
    x::Vector{T},
    resp::Float64, 
    feat::S where S<:AbstractVector, 
    params::AbstractVector{Int64}=eachindex(x)    
) where T<:Real
    view(gradient, params) .+= view(eachindex(x), params)

    return
end

# likelihood returns 1, gradient[params] should be increased by the value of 
# their index 
function likelihoodscore!(family::TestFamily;
    gradient::Vector{T},
    x::Vector{T},
    resp::Float64, 
    feat::S where S<:AbstractVector, 
    params::AbstractVector{Int64}=eachindex(x)    
) where T<:Real
    view(gradient, params) .+= view(eachindex(x), params)

    return T(1)
end

# hess[i,j] should be incremented by i*j 
function information!(family::TestFamily; 
    hessian::Matrix{T},
    x::Vector{T},
    resp::Float64,
    feat::S where S<:AbstractVector, 
    params::AbstractVector{Int64}=eachindex(x)
) where T<:Real 

    val = view(eachindex(x), params)
    view(hessian, params, params) .+= val * transpose(val)
    return 
end

# Problem 
num_param = 10 
num_obs = 20
problem = OptimizationProblems.GeneralizedLinearModel(
    "Test Problem",
    Dict{Symbol, Counter}(
        :obj => Counter(block_total=num_param, batch_total=num_obs),
        :grad => Counter(block_total=num_param, batch_total=num_obs),
        :hess => Counter(block_total=num_param, batch_total=num_obs),
        :residual => Counter(block_total=num_param, batch_total=num_obs),
        :jacobian => Counter(block_total=num_param, batch_total=num_obs)
    ),
    num_param,
    num_obs,
    randn(Float64, num_obs),
    randn(Float64, num_obs, num_param),
    TestFamily()
)

# Type and Pre-allocation 
type = Float16
x = randn(type, num_param)
store = allocate(problem, type=type, hess=true)


@testset "GLM Objective Information" begin 

    ######################################################
    # Objective Function 
    ######################################################

    # Compute full objective function 
    let
        obj!(problem, store=store, x=x)
        @test store[:obj] == Float64(num_obs)
        @test problem.counters[:obj].block_equivalent == 1
        @test problem.counters[:obj].batch_equivalent == 1

        reset!(problem.counters[:obj])
    end

    # Compute partial objective function information
    let batch_size=5, batch=randperm(num_obs)[1:batch_size]
        
        obj!(problem, store=store, x=x, batch=batch)
        @test store[:obj] == Float64(batch_size)
        @test problem.counters[:obj].block_equivalent == 1
        @test problem.counters[:obj].batch_equivalent == batch_size/num_obs

        reset!(problem.counters[:obj])
    end

    # Compute full objective function in two parts reset 
    let batch_size=5, batch_1=1:batch_size, batch_2=(batch_size+1):num_obs
        obj!(problem, store=store, x=x, batch=batch_1)
        obj!(problem, store=store, x=x, reset=false, batch=batch_2)

        @test store[:obj] == Float64(num_obs)
        @test problem.counters[:obj].block_equivalent == 2
        @test problem.counters[:obj].batch_equivalent == 1

        reset!(problem.counters[:obj])
    end

    ######################################################
    # Gradient Function  
    ######################################################

    # Compute full gradient function
    let 
        grad!(problem, store=store, x=x)

        @test store[:grad] == collect(1:num_param) .* num_obs
        @test problem.counters[:grad].block_equivalent == 1
        @test problem.counters[:grad].batch_equivalent == 1

        reset!(problem.counters[:grad])
    end

    # Compute batch subset gradient function 
    let batch_size=5, batch=randperm(num_obs)[1:batch_size]

        grad!(problem, store=store, x=x, batch=batch)

        @test store[:grad] == collect(1:num_param) .* batch_size 
        @test problem.counters[:grad].block_equivalent == 1 
        @test problem.counters[:grad].batch_equivalent == batch_size/num_obs

        reset!(problem.counters[:grad])
    end

    # Compute block subset gradient function 
    let block_size=5, block=randperm(num_param)[1:block_size]

        fill!(store[:grad], type(1))

        grad!(problem, store=store, x=x, block=block)

        @test store[:grad][block] == block .* num_obs
        @test store[:grad][setdiff(1:num_param,block)] == ones(type, num_param - 
            block_size)
        @test problem.counters[:grad].block_equivalent == block_size/num_param
        @test problem.counters[:grad].batch_equivalent == 1 

        reset!(problem.counters[:grad])
    end

    # Compute batch and block subset gradient function 
    let block_size=5, block=randperm(num_param)[1:block_size],
        batch_size=10, batch=randperm(num_obs)[1:batch_size]

        fill!(store[:grad], type(1))

        grad!(problem, store=store, x=x, batch=batch, block=block)

        @test store[:grad][block] == block .* batch_size 
        @test store[:grad][setdiff(1:num_param,block)] == ones(type, num_param -
            block_size)
        @test problem.counters[:grad].block_equivalent == block_size/num_param 
        @test problem.counters[:grad].batch_equivalent == batch_size/num_obs

        reset!(problem.counters[:grad])
    end

    # Compute full gradient in pieces 
    let block_size=5, batch_size=10,
        block_1=randperm(num_param)[1:block_size], 
        block_2=setdiff(1:num_param, block_1),  
        batch_1=randperm(num_obs)[1:batch_size], 
        batch_2=setdiff(1:num_obs,batch_1)

        fill!(store[:grad], type(0))

        grad!(problem, store=store, x=x, batch=batch_1, block=block_1)
        grad!(problem, store=store, x=x, reset=false, batch=batch_2, block=block_1)
        grad!(problem, store=store, x=x, reset=false, batch=batch_1, block=block_2)
        grad!(problem, store=store, x=x, reset=false, batch=batch_2, block=block_2)

        @test store[:grad] == (1:num_param) .* num_obs
        @test problem.counters[:grad].block_equivalent == 2
        @test problem.counters[:grad].batch_equivalent == 2

        reset!(problem.counters[:grad])
    end

    ######################################################
    # Objective-Gradient Function  
    ######################################################

    # Objective and Gradient calculation using full data, all parameters
    let
        store[:obj] = type(1) #Ensure values get overwritten
        fill!(store[:grad], type(1)) #Ensure values get overwritten 

        objgrad!(problem, store=store, x=x)

        @test store[:obj] == num_obs 
        @test store[:grad] == eachindex(x) .* num_obs
        @test problem.counters[:obj].block_equivalent == 1
        @test problem.counters[:obj].batch_equivalent == 1
        @test problem.counters[:grad].block_equivalent == 1
        @test problem.counters[:grad].batch_equivalent == 1

        reset!(problem.counters[:obj])
        reset!(problem.counters[:grad])
    end

    # Objective and Gradient calculation using subset of data, subset of parameters 
    let block_size=5, batch_size=10,
        block=randperm(num_param)[1:block_size],
        batch=randperm(num_obs)[1:batch_size]

        store[:obj] = type(1) #Ensure values get overwritten 
        fill!(store[:grad], type(1)) #Ensure values get overwritten 

        objgrad!(problem, store=store, x=x, batch=batch, block=block)

        @test store[:obj] == batch_size 
        @test store[:grad][block] == block .* batch_size
        @test store[:grad][setdiff(eachindex(x), block)] ==
            ones(type, num_param-block_size)
        @test problem.counters[:obj].block_equivalent == 1
        @test problem.counters[:obj].batch_equivalent == batch_size/num_obs 
        @test problem.counters[:grad].block_equivalent == block_size/num_param 
        @test problem.counters[:grad].batch_equivalent == batch_size/num_obs

        reset!(problem.counters[:obj])
        reset!(problem.counters[:grad])
    end

    # Full Objective and Gradient calculation in pieces 
    let block_size=5, batch_size=10,
        block_1=randperm(num_param)[1:block_size],
        block_2=setdiff(1:num_param, block_1),
        batch_1=randperm(num_obs)[1:batch_size],
        batch_2=setdiff(1:num_obs,batch_1)

        store[:obj] = type(0)
        fill!(store[:grad], type(0)) 

        objgrad!(problem, store=store, x=x, batch=batch_1, block=block_1)
        objgrad!(problem, store=store, x=x, batch=batch_2, block=block_1,
            reset=false)
        objgrad!(problem, store=store, x=x, batch=batch_1, block=block_2,
            reset=false)
        objgrad!(problem, store=store, x=x, batch=batch_2, block=block_2,
            reset=false)

        @test store[:obj] == num_obs * 2 #Full objective computed twice as blocks
        # do not matter   
        @test store[:grad] == (1:num_param) .* num_obs 
        @test problem.counters[:obj].block_equivalent == 4
        @test problem.counters[:obj].batch_equivalent == 2
        @test problem.counters[:grad].block_equivalent == 2
        @test problem.counters[:grad].batch_equivalent == 2
    end

    ######################################################
    # Hessian Function  
    ######################################################
    
    # Full Hessian calculation
    let v=eachindex(x)
        fill!(store[:hess], type(1)) #Should be overwritten 

        hess!(problem, store=store, x=x)

        @test store[:hess] == (v * v') .* num_obs 
        @test problem.counters[:hess].batch_equivalent == 1
        @test problem.counters[:hess].block_equivalent == 1

        reset!(problem.counters[:hess])
    end

    # Partial Hessian Calculation 
    let block_size=5, batch_size=10, v=eachindex(x),
        block_1=randperm(num_param)[1:block_size],
        block_2=setdiff(v, block_1),
        batch_1=randperm(num_param)[1:batch_size],
        batch_2=setdiff(1:num_obs,batch_1)

        fill!(store[:hess], type(1))

        hess!(problem, store=store, x=x, batch=batch_1, block=block_1)

        @test store[:hess][block_1,block_1] == (block_1 * block_1') .* batch_size
        @test store[:hess][block_2,block_2] == ones(num_param-block_size, num_param-
            block_size)
        @test problem.counters[:hess].batch_equivalent == batch_size/num_obs 
        @test problem.counters[:hess].block_equivalent == block_size/num_param

        reset!(problem.counters[:hess])
    end

    # Sub-Matrix of Hessian calculation in pieces 
    let block_size=5, batch_size=10, v=eachindex(x),
        block_1=randperm(num_param)[1:block_size],
        block_2=setdiff(v, block_1),
        batch_1=randperm(num_param)[1:batch_size],
        batch_2=setdiff(1:num_obs,batch_1)

        fill!(store[:hess], type(1)) #Ensures values are reset 

        hess!(problem, store=store, x=x, batch=batch_1, block=block_1)
        hess!(problem, store=store, x=x, batch=batch_2, block=block_1, reset=false)
        hess!(problem, store=store, x=x, batch=batch_1, block=block_2)
        hess!(problem, store=store, x=x, batch=batch_2, block=block_2, reset=false)

        @test store[:hess][block_1,block_1] == (block_1 * block_1') .* num_obs 
        @test store[:hess][block_2,block_2] == (block_2 * block_2') .* num_obs 
        @test store[:hess][block_1,block_2] == ones(type, length(block_1), length(block_2))
        @test store[:hess][block_2,block_1] == ones(type, length(block_2), length(block_1))
    end
end
end