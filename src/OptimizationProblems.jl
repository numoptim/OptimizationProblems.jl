module OptimizationProblems

using OptimizationModels, LinearAlgebra

include("glm/generalized_linear_models.jl")

export allocate, obj!, grad!, objgrad!, hess!
export LogisticRegression

end # module OptimizationProblems
