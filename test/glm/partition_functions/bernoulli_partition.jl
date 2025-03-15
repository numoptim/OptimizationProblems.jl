module TestBernoulliPartition 

using Test, OptimizationProblems, LinearAlgebra, Random 

family = :Bernoulli 
types = [Float16, Float32, Float64]
responses = [true, false]
xs(type) = [randn(type, 5), randn(type, 10), randn(type, 15)]
feats(type) = [randn(type, 5), randn(type, 10), randn(type, 15)]
logpartition(η) = log(1 + exp(η))
derivative_lp(η) = 1/(1+exp(-η))
second_der_lp(η) = 1/(1+exp(-η))*(1/(1+exp(η)))

@testset "Bernoulli Log Partition Function" begin 

    for resp in responses, type in types
        for (x, feat) in Iterators.zip(xs(type), feats(type))
            # Helper: Linear Effect 
            η = dot(x, feat)
            
            # Helper: Number of parameters 
            num_param = length(feat) 

            ##############################
            # Test likelihood calculation 
            ##############################
            let resp=resp, x=x, feat=feat
                
                # Calculate the likelihood using code base
                val = OptimizationProblems.likelihood(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    x=x,
                    resp=resp,
                    feat=feat
                )

                # Test likelihood calculation
                @test typeof(val) == type
                @test val == -resp*η + logpartition(η)
            end

            ##############################
            # Test Full Score Calculation 
            ##############################
            let gradient=zeros(type, num_param), resp=resp, x=x, feat=feat

                # Calculate score function using code base
                # Answer is stored in gradient. 
                OptimizationProblems.score!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    gradient=gradient,
                    x=x,
                    resp=resp,
                    feat=feat
                )

                # Test score calculation 
                @test typeof(gradient) == Vector{type}
                @test isapprox(gradient, -resp*feat + derivative_lp(η)*feat, 
                    atol=eps(type)*num_param)
            end

            ################################################################
            # Test Partial Score Calculation at 3 randomly selected indices
            ################################################################
            let gradient=zeros(type, num_param), resp=resp, x=x, feat=feat,
                params=sort(randperm(num_param)[1:3])

                # Calculate score function using code base for specified 
                # parameters. Answer is stored in gradient[params], 
                # other terms should remain fixed. 
                OptimizationProblems.score!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    gradient=gradient,
                    x=x,
                    resp=resp,
                    feat=feat,
                    params=params
                )

                # Test score calculation 
                @test typeof(gradient) == Vector{type}
                @test isapprox(gradient[params], -resp*feat[params] + 
                    derivative_lp(η)*feat[params], atol=eps(type)*num_param)
                @test isapprox(norm(gradient) - norm(gradient[params]), 0, 
                    atol=eps(type))
            end

            ##########################################################
            # Test Simultaneous Likelihood and Score Calculations
            ##########################################################
            let gradient=zeros(type, num_param), resp=resp, x=x, feat=feat

                # Calculate likelihood and score function using code base 
                # Likelihood is returned, score is stored in gradient. 
                val = OptimizationProblems.likelihoodscore!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    gradient=gradient,
                    x=x,
                    resp=resp,
                    feat=feat
                )

                # Test likelihood calculation
                @test typeof(val) == type
                @test val == -resp*η + logpartition(η)

                # Test score calculation  
                @test typeof(gradient) == Vector{type}
                @test isapprox(gradient, -resp*feat + derivative_lp(η)*feat, 
                    atol=eps(type)*num_param)
            end

            ##############################################################
            # Test Simultaneous Likelihood and Partial Score Calculations
            ##############################################################
            let gradient=zeros(type, num_param), resp=resp, x=x, feat=feat,
                params=sort(randperm(num_param)[1:3])

                # Calculate likelihood and score function using code base 
                # Likelihood is returned, score is stored in gradient.
                # Only indices in params are updated in gradient.  
                val = OptimizationProblems.likelihoodscore!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    gradient=gradient,
                    x=x,
                    resp=resp,
                    feat=feat,
                    params=params
                )

                # Test likelihood calculation
                @test typeof(val) == type
                @test val == -resp*η + logpartition(η)

                # Test score calculation 
                @test typeof(gradient) == Vector{type}
                @test isapprox(gradient[params], -resp*feat[params] + 
                    derivative_lp(η)*feat[params], atol=eps(type)*num_param)
                @test isapprox(norm(gradient) - norm(gradient[params]), 0, 
                    atol=eps(type))
            end

            ##############################################################
            # Test Information Calculations
            ##############################################################
            let hessian=zeros(type, num_param, num_param), resp=resp, x=x, 
                feat=feat

                # Calculation the information matrix
                # Solution should updated to hessian
                OptimizationProblems.information!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    hessian=hessian,
                    x=x,
                    resp=resp,
                    feat=feat,
                )

                # Test hessian calculation 
                @test typeof(hessian) == Matrix{type}
                @test isapprox(hessian, 
                    second_der_lp(η)*feat*transpose(feat),
                    atol=eps(type)*num_param
                )
            end

            ##############################################################
            # Test Partial Information Calculations
            ##############################################################
            let hessian=zeros(type, num_param, num_param), resp=resp, x=x, 
                feat=feat, params=sort(randperm(num_param)[1:3])

                # Calculation the information matrix
                # Solution should updated to hessian
                OptimizationProblems.information!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    hessian=hessian,
                    x=x,
                    resp=resp,
                    feat=feat,
                    params=params
                )

                # Test hessian calculation 
                @test typeof(hessian) == Matrix{type}
                @test isapprox(hessian[params, params], 
                    second_der_lp(η)*feat[params]*transpose(feat[params]),
                    atol=eps(type)*num_param
                )
                @test isapprox(norm(hessian) - norm(hessian[params, params]), 
                    0, atol=eps(type))
            end
        end
    end
end

end