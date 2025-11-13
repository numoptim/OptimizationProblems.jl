function trapezoidal(
    f::Function, 
    a::T, 
    b::T;
    rtol=sqrt(eps(T)),
    maxevals::Int64=10^7
) where T<:Real

    # Number of evaluations 
    N = min(ceil(Int64, abs(b-a) / sqrt(12 * rtol)), maxevals)
    Δ = (b - a) / N

    # Approximate integral 
    I = (f(a)+f(b))/2
    for i in 1:(N-1) 
        I += f(a + i*Δ)
    end
    
    return Δ*I
end

#TODO: docstrings and tests 