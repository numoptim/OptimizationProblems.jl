module TestGLMConstructor

using Test, OptimizationProblems

struct TestFamily <: OptimizationProblems.GLMFamily end 
resp_types = [BitVector, Vector{Tuple{Int,Int}}, Vector{Number}]
feat_types = [Matrix{Float16}, Matrix{Float32}, Matrix{Float64}]
fnames_types(R, F, G) = [
    (:name,String), 
    (:counters,Dict{Symbol,OptimizationProblems.OptimizationModels.Counter}), 
    (:num_param,Int64), 
    (:num_obs,Int64), 
    (:resp,R),
    (:feat,F), 
    (:family,G)
]


@testset "GLM Constructor" begin

    #Definitions and Supertypes 
    @test isdefined(OptimizationProblems, :GeneralizedLinearModel)
    @test supertype(OptimizationProblems.GeneralizedLinearModel) ==
        OptimizationProblems.OptimizationProblem

    #Check Field Names 
    for pair_name_type in fnames_types(Any, Any, TestFamily)
        @test pair_name_type[1] in fieldnames(OptimizationProblems.GeneralizedLinearModel)
    end

    #Constructors & Field Types 
    for resp_type in resp_types, feat_type in feat_types
        problem =  OptimizationProblems.GeneralizedLinearModel(
            "Test Problem",
            Dict{Symbol, OptimizationProblems.OptimizationModels.Counter}(),
            10,
            20,
            resp_type(undef, 10),
            feat_type(undef,20,10),
            TestFamily()
        )

        # Test Field Types 
        for pair_name_type in fnames_types(resp_type, feat_type, TestFamily)
            @test typeof(getfield(problem, pair_name_type[1])) == pair_name_type[2]
        end
    end

    

end 
end