module TestBernoulliGGN

using Test, OptimizationProblems, LinearAlgebra, Random

family = :Bernoulli 
types = [Float16, Float32, Float64]
responses = [true, false]
xs(type) = [randn(type, 5), randn(type, 10), randn(type, 15)]
feats(type) = [randn(type, 5), randn(type, 10), randn(type, 15)]
logpartition(η) = log(1 + exp(η))
derivative_lp(η) = 1/(1+exp(-η))
second_der_lp(η) = 1/(1+exp(-η))*(1/(1+exp(η)))

@testset "Bernoulli Generalized Gauss-Newton Functions" begin 

    for resp in responses, type in types 
        for (x, feat) in Iterators.zip(xs(type), feats(type))
            # Helper: Linear Effect 
            η = dot(x, feat)
            
            # Helper: Number of parameters 
            num_param = length(feat) 

            ######################################################
            # Test Weights Calculation
            ######################################################
            let resp=resp, x=x, feat=feat
                
                # Calculate Square Root of Weight 
                val = OptimizationProblems.gnn_weight(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    x=x,
                    resp=resp,
                    feat=feat
                )

                @test typeof(val) == type 
                @test isapprox(val, sqrt(second_der_lp(η)), atol=
                    eps(type)*num_param)
            end

            ######################################################
            # Test Constant Calculation
            ######################################################
            let resp=resp, x=x, feat=feat
                
                # Calculate Constant Term  
                val = OptimizationProblems.gnn_constant(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    x=x,
                    resp=resp,
                    feat=feat,
                    weight=type(3.0)
                )

                @test typeof(val) == type 
                @test isapprox(val, (resp - derivative_lp(η))/type(3.0), 
                    atol=sqrt(eps(type)*num_param))
            end

            ######################################################
            # Test Weighted Feature Calculation 
            ######################################################
            let resp=resp, x=x, feat=feat, weighted_feat=zeros(type, num_param)
                
                # Calculate Weighted Feature 
                OptimizationProblems.gnn_coefficient!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    weighted_feat=weighted_feat,
                    x=x,
                    feat=feat,
                    weight=type(3.0)
                )

                @test isapprox(weighted_feat, type(3.0).*feat, 
                    atol=eps(type)*num_param)
            end

            ######################################################
            # Test Partial Weighted Feature Calculation 
            ######################################################
            let resp=resp, x=x, feat=feat, weighted_feat=zeros(type, num_param),
                params=sort(randperm(num_param)[1:3])
                
                # Calculate Partial Weighted Feature 
                OptimizationProblems.gnn_coefficient!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    weighted_feat=weighted_feat,
                    x=x,
                    feat=feat,
                    weight=type(3.0),
                    params=params
                )

                @test isapprox(weighted_feat[params], type(3.0).*feat[params], 
                    atol=eps(type)*num_param)

                @test isapprox(norm(weighted_feat) - norm(weighted_feat[params]), 
                    type(0.0), atol=eps(type)*num_param)
            end
        end
    end
end

end