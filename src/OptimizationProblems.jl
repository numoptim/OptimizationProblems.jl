module OptimizationProblems

using OptimizationModels, LinearAlgebra, Random, Distributions

include("glm/generalized_linear_models.jl")

export allocate, obj!, grad!, objgrad!, hess!
export LogisticRegression, BinomialRegression, ExponentialRegression,
    LinearRegression, PoissonRegression, GeometricRegression, NegativeBinomialRegression

end # module OptimizationProblems
