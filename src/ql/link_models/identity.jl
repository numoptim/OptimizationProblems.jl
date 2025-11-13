function link(
    model::Identity;
    η::T
) where T<:Real
    return η
end

function der_link(
    model::Identity;
    η::T
) where T<:Real
    return T(1)
end

function der2_link(
    model::Identity;
    η::T
) where T<:Real
    return T(0)
end

