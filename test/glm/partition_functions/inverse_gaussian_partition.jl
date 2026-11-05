module TestInverseGaussianPartition

using Test, OptimizationProblems, LinearAlgebra, Random

family = :InverseGaussian
types = [Float16, Float32, Float64]
responses(type) = rand(type, 3)
xs(type) = [-rand(type, 5), -rand(type, 10), -rand(type, 15)]
feats(type) = [rand(type, 5), rand(type, 10), rand(type, 15)]
logpartition(η) = -sqrt(-2*η)
derivative_lp(η) = 1/sqrt(-2*η)
second_der_lp(η) = (-2*η)^(-3/2)

@testset "Inverse Gaussian Log Partition Function" begin

    for type in types
        for (x, resp, feat) in Iterators.zip(xs(type), responses(type),
            feats(type))

            # Helper: Linear Effect
            η = min(dot(x, feat), type(0))

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

            # η = dot(x, feat) = ||x||^2 > 0
            let resp=resp, x=x, feat=x
                @test_throws DomainError OptimizationProblems.likelihood(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    x=x,
                    resp=resp,
                    feat=feat
                )
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

            # η = dot(x, feat) = ||x||^2 > 0
            let gradient=zeros(type, num_param), resp=resp, x=x, feat=x
                @test_throws DomainError OptimizationProblems.score!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    gradient=gradient,
                    x=x,
                    resp=resp,
                    feat=feat
                )
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

            # η = dot(x, feat) = ||x||^2 > 0
            let gradient=zeros(type, num_param), resp=resp, x=x, feat=x
                @test_throws DomainError OptimizationProblems.likelihoodscore!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    gradient=gradient,
                    x=x,
                    resp=resp,
                    feat=feat
                )
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
                    atol=sqrt(eps(type)*num_param), rtol=sqrt(eps(type))
                )
            end

            # η = dot(x, feat) = ||x||^2 > 0
            let hessian=zeros(type, num_param, num_param), resp=resp, x=x,
                feat=x
                @test_throws DomainError OptimizationProblems.information!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    hessian=hessian,
                    x=x,
                    resp=resp,
                    feat=feat
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
                    atol=sqrt(eps(type)*num_param), rtol=sqrt(eps(type))
                )
                @test isapprox(norm(hessian) - norm(hessian[params, params]),
                    0, atol=eps(type))
            end

            ##############################################################
            # Test DomainError when η = dot(x, feat) is exactly zero
            ##############################################################
            let resp=resp, x0=zeros(type, num_param), feat0=zeros(type,
                num_param), gradient=zeros(type, num_param), hessian=
                zeros(type, num_param, num_param)

                # Construct x0 and feat0 so that dot(x0, feat0) == 0 exactly.
                # Inverse Gaussian requires η strictly negative because the
                # mean μ = 1/sqrt(-2η) is undefined at η = 0.
                feat0[1] = one(type)
                x0[2] = -one(type)

                @test_throws DomainError OptimizationProblems.likelihood(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    x=x0,
                    resp=resp,
                    feat=feat0
                )

                @test_throws DomainError OptimizationProblems.score!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    gradient=gradient,
                    x=x0,
                    resp=resp,
                    feat=feat0
                )

                @test_throws DomainError OptimizationProblems.likelihoodscore!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    gradient=gradient,
                    x=x0,
                    resp=resp,
                    feat=feat0
                )

                @test_throws DomainError OptimizationProblems.information!(
                    eval(Meta.parse("OptimizationProblems.$family()")),
                    hessian=hessian,
                    x=x0,
                    resp=resp,
                    feat=feat0
                )
            end
        end
    end
end
end
