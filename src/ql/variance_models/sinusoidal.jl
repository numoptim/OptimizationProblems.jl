function var(
    model::Sinusoidal{T},
    μ::T 
) where T<:Real 
    return T(1 + μ + sin(2*π*μ))
end

function der_var(
    model::Sinusoidal{T},
    μ::T
) where T<:Real 
    return T(1 + 2*π*cos(2*π*μ))
end