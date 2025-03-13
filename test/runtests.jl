using Test
using OptimizationProblems

@testset verbose=true "OptimizationProblems.jl" begin
    for path in readlines(joinpath(@__DIR__, "test.txt"))
        string(path[1]) != "%" && include(path)
    end
end