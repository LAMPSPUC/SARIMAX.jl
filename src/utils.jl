"""
    differentiate(
        series::TimeArray,
        d::Int=0, 
        D::Int=0, 
        s::Int=1
    )

    Differentiates a TimeArray `series` `d` times and `D` times with a seasonal difference of `s` periods.

    This method only works with d,D ∈ {0,1}.

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> stationaryAirPassengers = differentiate(airPassengers, d=1, D=1, s=12)
```
"""
function differentiate(series::TimeArray,d::Int=0, D::Int=0, s::Int=1)
    series = TimeArray(timestamp(series),values(series))
    if D > 0
        @info("Seasonal difference")
        diffValues = []
        originalValues = values(series)
        T = length(originalValues)
        for i=1:D
            # Δyₜ = yₜ - y_t-s
            for j=i*s+1:T
                push!(diffValues, originalValues[j] - originalValues[j-i*s])
            end
        end
        series = TimeArray(copy(timestamp(series))[(D*s)+1:end],diffValues)
    end
    # non seasonal diff y
    @info("Non seasonal difference")
    for _ in 1:d
        diffValues = []
        originalValues = values(series)
        T = length(originalValues)
        # Δyₜ = yₜ - y_t-1
        for j=2:T
            push!(diffValues,originalValues[j] - originalValues[j-1])
        end
        series = TimeArray(copy(timestamp(series))[2:end],diffValues)
    end
    return series
end

"""
    integrate(
        series::TimeArray, 
        diffSeries::Vector{Fl}, 
        d::Int = 0, 
        D::Int = 0, 
        s::Int = 1
    ) where Fl<:Real

    Integrates a vector of a differentiated series `diffSeries` `d` times and `D` times with a seasonal difference of `s` periods assuming the original previous observations are in `series`.

    This method only works with d,D ∈ {0,1}.
"""
function integrate(series::TimeArray, diffSeries::Vector{Fl}, d::Int=0, D::Int=0, s::Int=1) where Fl<:Real
    series = TimeArray(timestamp(series),values(series))
    stepsAhead = length(diffSeries)
    y = deepcopy(values(series))
    T = length(y)
    y = vcat(y,diffSeries)
    for i=T+1:T+stepsAhead
        # @info("Non seasonal integration")
        recoveredValue = y[i]
        # Δyt = y[t] - y[t-1] - y[t-12] + y[t-12-1]
        for _ in 1:d
            # Δyₜ = yₜ - y_t-1 ⇒ yₜ = Δyₜ + y_t-1
            recoveredValue += y[i-1]
        end
        # @info("Seasonal integration")
        for _ in 1:D
            # Δyₜ = yₜ - y_t-s ⇒ yₜ = Δyₜ + y_t-s
            recoveredValue += y[i-s]
        end
        # @info("Correction for seasonal integration")
        if D > 0 && d > 0
            # Δyₜ = yₜ - y_t-s ⇒ yₜ = Δyₜ + y_t-s
            recoveredValue -= y[i-s-1]
        end
        y[i] = recoveredValue
    end
    
    return y[T+1:end]
end