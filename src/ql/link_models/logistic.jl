function link(
    model::Logistic;
    η::T 
) where T<:Real
    return T(1/(1 + exp(-η)))
end

function der_link(
    model::Logistic;
    η::T 
) where T<:Real
    μ = link(model, η)
    return μ * (1 - μ)
end

function der2_link(
    model::Logistic;
    η::T 
) where T<:Real
    μ = link(model, η)
    return μ * (1 - μ) * (1 - 2*μ)
end