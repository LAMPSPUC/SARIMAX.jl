module Parameters

export sarimaParameters

mutable struct SarimaParameters
    p::Int64
    d::Int64
    q::Int64
    P::Int64
    D::Int64
    Q::Int64
    m::Int64
end

function sarimaParameters(p,d,q)
    if (p < 0 || d < 0 || q < 0)
        error("Negative values not allowed")
    end
    return SarimaParameters(p,d,q,0,0,0,0)
end

function sarimaParameters(p,d,q,P,D,Q,m)
    if (p < 0 || d < 0 || q < 0 || P < 0 || D < 0 || Q < 0 || m < 0)
        error("Negative values not allowed")
    end
    return SarimaParameters(p,d,q,P,D,Q,m)
end

end # module Parameters