using Documenter, DocumenterCitations
using OptimizationProblems

makedocs(
    sitename="OptimizationProblems Documentation",
    pages = [
        "Overview" => "index.md",
    ]
)

#When repo is public and github workflow is created
#deploydocs(
#    repo = "github.com/numoptim/OptimizationProblems.jl",
#)