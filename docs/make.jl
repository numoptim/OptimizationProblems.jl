using Documenter, DocumenterCitations
using OptimizationProblems

makedocs(
    sitename="OptimizationProblems Documentation",
    pages = [
        "Overview" => "index.md",
        "Manual" => [
            "Quick Start" => "manual/quick_start.md",
            "Generalized Linear Models" => "manual/glm.md",
            "Quasi-likelihood Models" => "manual/ql.md",
        ],
        "API" => [
            "Optimization Interface" => "api/interface.md",
            "Genearlized Linear Models" => "api/glm.md",
            "Quasi-Likelihood Models" => "api/ql.md",
        ]
    ]
)

#When repo is public and github workflow is created
deploydocs(
    repo = "github.com/numoptim/OptimizationProblems.jl",
)