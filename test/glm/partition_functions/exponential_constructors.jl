module TestExponentialConstructor

using Test, OptimizationProblems

family = :Exponential
types = [Float16, Float32, Float64]
constructor = ExponentialRegression
response_type(type) = Vector{type}
feature_type(type) = Matrix{type}

@testset "Exponential Regression Constructors" begin

    # Definitions and Supertypes 
    @test isdefined(OptimizationProblems, family)
    @test supertype(eval(Meta.parse("OptimizationProblems.$family"))) ==
        OptimizationProblems.GLMFamily

    # Constructors for randomly generated problems 
    for type in types 
        # Number of parameters should be at least one 
        let num_param=0, num_obs=10
            @test_throws ArgumentError constructor(type,
                num_param=num_param, num_obs=num_obs)
        end

        #Number of observations should be at least one 
        let num_param=1, num_obs=0
            @test_throws ArgumentError constructor(type,
                num_param=num_param, num_obs=num_obs)
        end

        #Verify unique fields of the problem and values
        let num_param=1, num_obs=1
            problem = constructor(type, num_param=num_param, num_obs=
                num_obs)

            @test length(problem.resp) == num_obs 
            @test typeof(problem.resp) == response_type(type) 
            @test size(problem.feat) == (num_obs, num_param)
            @test typeof(problem.feat) == feature_type(type)
            @test problem.feat[:,1] == ones(type, num_obs)
        end

        # Verify same conditions for different number of parameters 
        let num_param=5, num_obs=1
            problem = constructor(type, num_param=num_param, num_obs=
                num_obs)

            @test length(problem.resp) == num_obs 
            @test typeof(problem.resp) == response_type(type) 
            @test size(problem.feat) == (num_obs, num_param)
            @test typeof(problem.feat) == feature_type(type)
            @test problem.feat[:,1] == ones(type, num_obs)
        end

        # Verify same conditions for different number of parameters and observations
        let num_param=5, num_obs=10
            problem = constructor(type, num_param=num_param, num_obs=
                num_obs)

            @test length(problem.resp) == num_obs 
            @test typeof(problem.resp) == response_type(type) 
            @test size(problem.feat) == (num_obs, num_param)
            @test typeof(problem.feat) == feature_type(type)
            @test problem.feat[:,1] == ones(type, num_obs)
        end
    end

    # Constructors for user-supplied problems
    for type in types
        
        #Number of parameters should be at least one 
        let resp=rand(type, 10), feat=zeros(type, 10, 0) 
            @test_throws ArgumentError constructor(resp=resp, feat=feat)
        end

        #Number of observations should be at least one 
        let resp=rand(type, 0), feat=zeros(type, 0, 3)
            @test_throws ArgumentError constructor(resp=resp, feat=feat)
        end

        #Number of responses and number of observations in feature must match
        let resp=rand(type, 0), feat=zeros(type, 5, 3)
            @test_throws DimensionMismatch constructor(resp=resp, feat=feat)
        end

        #Observations must be non-negative 
        let resp=-rand(type, 10), feat=zeros(type, 10, 3)
            @test_throws DomainError constructor(resp=resp, feat=feat)
        end

        #Verify sizes and types
        let resp=zeros(type, 1), feat=zeros(type, 1, 1)

            problem = constructor(resp=resp, feat=feat)

            @test length(problem.resp) == 1 
            @test typeof(problem.resp) == response_type(type) 
            @test size(problem.feat) == (1, 1)
            @test typeof(problem.feat) == feature_type(type)
        end

        #Verify sizes and types 
        let resp=zeros(type, 1), feat=zeros(type, 1, 3)
            problem = constructor(resp=resp, feat=feat)

            @test length(problem.resp) == 1 
            @test typeof(problem.resp) == response_type(type) 
            @test size(problem.feat) == (1, 3)
            @test typeof(problem.feat) == feature_type(type)
        end

        #Verify sizes and types 
        let resp=zeros(type, 10), feat=zeros(type, 10, 3)
            problem = constructor(resp=resp, feat=feat)

            @test length(problem.resp) == 10
            @test typeof(problem.resp) == response_type(type) 
            @test size(problem.feat) == (10, 3)
            @test typeof(problem.feat) == feature_type(type)
        end
    end
end

end