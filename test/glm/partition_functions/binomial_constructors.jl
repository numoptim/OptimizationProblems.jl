module TestBinomialConstructor

using Test, OptimizationProblems

family = :Binomial
types = [Float16, Float32, Float64]
constructor = BinomialRegression 
response_type(type) = Vector{Tuple{Int64, Int64}}
feature_type(type) = Matrix{type}

@testset "Binomial Regression Constructors" begin 

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
        let resp=Vector{Tuple{Int64, Int64}}(undef, 10), feat=zeros(type, 10, 0) 
            @test_throws ArgumentError constructor(resp=resp, feat=feat)
        end

        #Number of observations should be at least one 
        let resp=Vector{Tuple{Int64, Int64}}(undef, 0), feat=zeros(type, 0, 3)
            @test_throws ArgumentError constructor(resp=resp, feat=feat)
        end

        #Number of responses and number of observations in feature must match
        let resp=Vector{Tuple{Int64, Int64}}(undef, 0), feat=zeros(type, 5, 3)
            @test_throws DimensionMismatch constructor(resp=resp, feat=feat)
        end

        #Verify sizes and types
        let resp=Tuple{Int64, Int64}[(0,1)], feat=zeros(type, 1, 1)
            problem = constructor(resp=resp, feat=feat)

            @test length(problem.resp) == 1 
            @test typeof(problem.resp) == response_type(type) 
            @test size(problem.feat) == (1, 1)
            @test typeof(problem.feat) == feature_type(type)
        end

        #Verify sizes and types 
        let resp=Tuple{Int64, Int64}[(0,1)], feat=zeros(type, 1, 3)
            problem = constructor(resp=resp, feat=feat)

            @test length(problem.resp) == 1 
            @test typeof(problem.resp) == response_type(type) 
            @test size(problem.feat) == (1, 3)
            @test typeof(problem.feat) == feature_type(type)
        end

        #Verify sizes and types 
        let resp=repeat([(0,1)], 10), feat=zeros(type, 10, 3)
            problem = constructor(resp=resp, feat=feat)

            @test length(problem.resp) == 10
            @test typeof(problem.resp) == response_type(type) 
            @test size(problem.feat) == (10, 3)
            @test typeof(problem.feat) == feature_type(type)
        end

        #Verify response validation of successes being non-negative 
        let resp=[(-1,1)], feat=zeros(type, 1, 1)
            @test_throws DomainError("The first entry of a pair in `resp` must be \
            be a non-negative integer.") constructor(resp=resp, feat=feat)
        end

        #Verify response validation of trials being positive 
        let resp=[(0,0)], feat=zeros(type, 1, 1)
            @test_throws DomainError("The second entry of a pair in `resp` must be \
            a positive integer.") constructor(resp=resp, feat=feat)
        end

        #Verify response validation of trials being greater than successes 
        let resp=[(0,1), (2,1)], feat=zeros(type, 2, 1)
            @test_throws DomainError("The first entry of a pair in `resp` must \
            not exceed the second entry of the pair") constructor(resp=resp, feat=feat)
        end
    end
end
end