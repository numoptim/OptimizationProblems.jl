function link(
    model::Identity;
    η::T
) where T<:Real
    return η
end

function der_lin(
    model::Identity;
    η::T
) where T<:Real
    return T(1)
end

function der2_lin(
    model::Identity;
    η::T
) where T<:Real
    return T(0)
end

