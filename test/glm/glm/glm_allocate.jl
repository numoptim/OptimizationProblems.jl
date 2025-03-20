module TestGLMAllocate

using Test, OptimizationProblems

# Test Parameters 
num_param = 10
num_obs = 20
struct TestFamily <: OptimizationProblems.GLMFamily end 
problem = OptimizationProblems.GeneralizedLinearModel(
    "Test Problem",
    Dict{Symbol, OptimizationProblems.OptimizationModels.Counter}(),
    num_param,
    num_obs,
    BitVector(undef, num_obs),
    Matrix{Int64}(undef,num_obs,num_param),
    TestFamily()
)
types = [Float16, Float32, Float64]


@testset "GLM Allocate" begin 

    for type in types, (obj, grad, hess, weights, residual, jacobian) in 
        Iterators.product(repeat([[true, false]], 6)...)

        store = allocate(problem, type=type, obj=obj, grad=grad, hess=hess, 
            weights=weights, residual=residual, jacobian=jacobian)

            
        # Check if keys are correctly defined and types are correctly allocated 
        for (key, state, val, dim) in Iterators.zip(
            (:obj, :grad, :hess, :weights, :residual, :jacobian),
            (obj, grad, hess, weights, residual, jacobian),
            (type, Vector{type}, Matrix{type}, Vector{type}, Vector{type}, Matrix{type}),
            ((), (num_param,), (num_param, num_param), (num_obs,), (num_obs,),
            (num_obs, num_param))
        )
            # Test if key exists in store, should match state 
            # E.g., if obj==true, then haskey(store, :obj) == true
            @test haskey(store, key) == state

            # If key exists in store, check its value type 
            state && (@test typeof(store[key]) == val)

            # If key exists in store, check its dimensions
            state && (@test size(store[key]) == dim)
        end 
        

    end
end
end