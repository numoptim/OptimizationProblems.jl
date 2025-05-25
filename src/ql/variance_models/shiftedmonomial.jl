function var(
    model::ShiftedMonomial{T},
    μ::T
) where T<:Real
    return T(μ^(model.p*2) + model.c)
end

function der_var(
    model::ShiftedMonomial{T},
    μ::T
) where T<:Real
    return T(2*model.p*μ^(model.p*2-1))
end