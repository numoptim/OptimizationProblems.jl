function link(
    model::InvCompLogLog;
    η::T
) where T<:Real 
    return T(1 - exp(-exp(η)))
end

function der_link(
    model::InvCompLogLog;
    η::T
) where T<:Real
    return T(exp(η - exp(η)))
end

function der2_link(
    model::InvCompLogLog;
    η::T
) where T<:Real
    return T(exp(η - exp(η)) * (1 - exp(η)))
end