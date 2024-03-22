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
        # @info("Seasonal difference")
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
    # @info("Non seasonal difference")
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

function selectSeasonalIntegrationOrder(
    y::Vector{Float64},
    seasonality::Int64,
    test::String
)
    if test == "seas"
        return StateSpaceModels.seasonal_strength_test(y,seasonality)
    elseif test == "ch"
        return StateSpaceModels.canova_hansen_test(y,seasonality)
    end

    throw(ArgumentError("The test $test is not supported"))
end

function selectIntegrationOrder(
    y::Vector{Float64},
    maxd::Int64,
    D::Int64,
    seasonality::Int64,
    test::String
)
    if test == "kpss"
        return StateSpaceModels.repeated_kpss_test(y,maxd,D,seasonality)
    end

    throw(ArgumentError("The test $test is not supported"))
end

"""

"""
function automaticDifferentiation(
    series::TimeArray;
    seasonalPeriod::Int=1,
    seasonalIntegrationTest::String="seas",
    integrationTest::String="kpss",
    maxd::Int=2
)
    @assert integrationTest ∈ ["kpss"]
    @assert seasonalIntegrationTest ∈ ["seas","ch"]
    @assert seasonalPeriod >= 1 

    # Indentify seasonal integration order
    seasonalIntegrationOrder = 0
    if seasonalPeriod != 1
        seasonalIntegrationOrder = selectSeasonalIntegrationOrder(values(series),s,"seas")
    end

    # Indentify integration order
    integrationOrder = selectIntegrationOrder(values(series),maxd,seasonalIntegrationOrder,seasonalPeriod,integrationTest)
    
    # Apply the integration orders to differentiate the time series
    diffSeries = differentiate(series,integrationOrder,seasonalIntegrationOrder,seasonalPeriod)
    
    automaticDifferentiationResult::NamedTuple = (diffSeries=diffSeries, d=integrationOrder, D=seasonalIntegrationOrder, s=seasonalPeriod)

    return automaticDifferentiationResult
end

"""
    loglikelihood(
        model::SARIMAModel
    )

    Calculates the loglikelihood of a SARIMA model based on the following formula:
    -0.5 * (T * log(2π) + T * log(σ²) + sum(ϵ.^2 ./ σ²))
"""
function loglikelihood(model::SarimaxModel)
    !hasFitMethods(typeof(model)) && throw(MissingMethodImplementation("fit!"))
    !isFitted(model) && throw(ModelNotFitted())
    T = length(model.ϵ)
    return -0.5 * (T * log(2π) + T * log(model.σ²) + sum(model.ϵ.^2 ./ model.σ²))
end

"""
    loglike(
        model::SARIMAModel
    )

    For a set of independent data points {x_1, x_2, ..., x_n} assumed to be drawn from a probability distribution with a PDF f(x; θ), where θ represents the model parameters, the log-likelihood LL(θ) is calculated as the sum of the log PDFs:

    LL(θ) = Σ[log(f(x_i; θ))]

    Here:

    LL(θ) is the log-likelihood.
    Σ denotes the summation over all data points from i = 1 to n.
    log() represents the natural logarithm.
    f(x_i; θ) is the probability density function of the model evaluated at data point x_i with parameter θ.
"""
function loglike(model::SarimaxModel)
    !hasFitMethods(typeof(model)) && throw(MissingMethodImplementation("fit!"))
    !isFitted(model) && throw(ModelNotFitted())
    return sum(logpdf.(Normal(0, sqrt(model.σ²)), model.ϵ))
end