using Test
using OptimizationProblems

@testset verbose=true "OptimizationProblems.jl" begin
    for file in readlines(joinpath(@__DIR__, "test.txt"))
        include(file)
    end
end