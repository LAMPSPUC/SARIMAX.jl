"""
    differentiate(
        series::TimeArray,
        d::Int=0, 
        D::Int=0, 
        s::Int=1
    )

Differentiates a `TimeArray` `series` `d` times and `D` times with a seasonal difference of `s` periods.

# Arguments
- `series::TimeArray`: The time series data to differentiate.
- `d::Int=0`: The number of non-seasonal differences to take.
- `D::Int=0`: The number of seasonal differences to take.
- `s::Int=1`: The seasonal period for the differences.

# Returns
A differentiated `TimeArray`.

# Errors
- This method only works with `d` and `D` in the set {0,1}.

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> stationaryAirPassengers = differentiate(airPassengers, d=1, D=1, s=12)
```
"""
function differentiate(series::TimeArray,d::Int=0, D::Int=0, s::Int=1)
    series = TimeArray(timestamp(series),values(series), colnames(series))
    seriesName = colnames(series)[1]
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
        series = TimeArray(copy(timestamp(series))[(D*s)+1:end],diffValues,[seriesName])
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
        series = TimeArray(copy(timestamp(series))[2:end],diffValues,[seriesName])
    end
    return series
end

"""
    differentiatedCoefficients(d::Int, D::Int, s::Int)

Compute the coefficients for differentiating a time series.

# Arguments
- `d::Int`: Order of non-seasonal differencing.
- `D::Int`: Order of seasonal differencing.
- `s::Int`: Seasonal period.

# Returns
- `coeffs::Vector{Float64}`: Coefficients for differentiation.
"""
function differentiatedCoefficients(d::Int, D::Int, s::Int)
    # Calculate the length of the resulting coefficients array
    lenCoeffs = d + D * s + 1

    # Initialize an array to store the coefficients
    coeffs = zeros(Float64, lenCoeffs)

    # Calculate the binomial coefficients
    binomialCoeffsd = [binomial(d, i) for i in 0:d]
    binomialCoeffsD = [binomial(D, j) for j in 0:D]

    # Calculate the coefficients
    for i in 0:d
        for j in 0:D
            coeffs[i + j * s + 1] = (-1)^i * binomialCoeffsd[i + 1] * (-1)^j * binomialCoeffsD[j + 1]
        end
    end

    return coeffs
end


"""
    integrate(initialValues::Vector{T}, diffSeries::Vector{T}, d::Int, D::Int, s::Int) where T

Converts a differentiated time series back to its original scale.

# Arguments
- `initialValues::Vector{T}`: Initial values of the original time series.
- `diffSeries::Vector{T}`: Differentiated time series.
- `d::Int`: Order of non-seasonal differencing.
- `D::Int`: Order of seasonal differencing.
- `s::Int`: Seasonal period.

# Returns
- `origSeries::Vector{Float64}`: Time series in the original scale.
"""
function integrate(initialValues::Vector{T}, diffSeries::Vector{T}, d::Int, D::Int, s::Int) where T
    # Get the coefficients for differentiation
    coeffs = differentiatedCoefficients(d, D, s)
    lenCoeffs = length(coeffs)
    
    # Calculate the length of the original series
    lenSeries = length(diffSeries) + d + D*s
    
    # Initialize an array to store the original series
    origSeries = zeros(Float64, lenSeries)
    
    # Copy the initial values to the original series
    origSeries[1:length(initialValues)] .= initialValues
    initialOffset = length(initialValues)
    
    # Iterate through the differentiated series and compute the original series
    for i in 1:length(diffSeries)
        # Calculate the value at the current index
        origSeries[initialOffset+i] = diffSeries[i]
        
        # Add contributions from past observations
        origSeries[initialOffset+i] -= coeffs[2:end]'origSeries[initialOffset+i-1:-1:initialOffset+i-lenCoeffs+1]
    end
    
    return origSeries
end


"""
    selectSeasonalIntegrationOrder(y, seasonality, test)

Selects the seasonal integration order for a time series based on the specified test.

# Arguments
- `y::Vector{Float64}`: The time series data.
- `seasonality::Int64`: The seasonal period of the time series.
- `test::String`: The name of the test to use for selecting the seasonal integration order.

# Returns
The selected seasonal integration order.

# Errors
Throws an ArgumentError if the specified test is not supported.

"""
function selectSeasonalIntegrationOrder(
    y::Vector{Float64},
    seasonality::Int64,
    test::String
)
    if test == "seas"
        return StateSpaceModels.seasonal_strength_test(y, seasonality)
    elseif test == "ch"
        return StateSpaceModels.canova_hansen_test(y, seasonality)
    end

    throw(ArgumentError("The test $test is not supported"))
end

"""
    selectIntegrationOrder(y, maxd, D, seasonality, test)

Selects the integration order for a time series based on the specified test.

# Arguments
- `y::Vector{Float64}`: The time series data.
- `maxd::Int64`: The maximum order of differencing to consider.
- `D::Int64`: The maximum seasonal order of differencing to consider.
- `seasonality::Int64`: The seasonal period of the time series.
- `test::String`: The name of the test to use for selecting the integration order.

# Returns
The selected integration order.

# Errors
Throws an ArgumentError if the specified test is not supported.

"""
function selectIntegrationOrder(
    y::Vector{Float64},
    maxd::Int64,
    D::Int64,
    seasonality::Int64,
    test::String
)
    if test == "kpss"
        return StateSpaceModels.repeated_kpss_test(y, maxd, D, seasonality)
    end

    throw(ArgumentError("The test $test is not supported"))
end

"""
    automaticDifferentiation(series; seasonalPeriod=1, seasonalIntegrationTest="seas", integrationTest="kpss", maxd=2)

Automatically applies differentiation to each series in a TimeArray.

# Arguments
- `series::TimeArray`: The input TimeArray containing the time series data.
- `seasonalPeriod::Int=1`: The seasonal period of the time series.
- `seasonalIntegrationTest::String="seas"`: The test used to select the seasonal integration order.
- `integrationTest::String="kpss"`: The test used to select the integration order.
- `maxd::Int=2`: The maximum order of differencing to consider.

# Returns
A tuple `(diffSeries, diffSeriesMetadata)` containing:
- `diffSeries::Vector{TimeArray}`: The differentiated time series.
- `diffSeriesMetadata::Vector{Dict{Symbol, Any}}`: Metadata containing the integration orders used for differentiation.

# Errors
Throws an AssertionError if invalid test options or seasonal period are provided.

"""
function automaticDifferentiation(
    series::TimeArray;
    seasonalPeriod::Int=1,
    seasonalIntegrationTest::String="seas",
    integrationTest::String="kpss",
    maxd::Int=2
)
    @assert integrationTest ∈ ["kpss"]
    @assert seasonalIntegrationTest ∈ ["seas", "ch"]
    @assert seasonalPeriod ≥ 1 

    diffSeriesVector::Array{TimeArray} = []
    diffSeriesMetadata = Dict{Symbol, Any}()
    
    for col in colnames(series)
        # Identify seasonal integration order
        y = series[col]
        seasonalIntegrationOrder = 0
        if seasonalPeriod ≠ 1
            seasonalIntegrationOrder = selectSeasonalIntegrationOrder(values(y), seasonalPeriod, seasonalIntegrationTest)
        end

        # Identify integration order
        integrationOrder = Sarimax.selectIntegrationOrder(values(y), maxd, seasonalIntegrationOrder, seasonalPeriod, integrationTest)
        
        # Apply the integration orders to differentiate the time series
        diffSeriesAux = differentiate(y, integrationOrder, seasonalIntegrationOrder, seasonalPeriod)
        push!(diffSeriesVector, diffSeriesAux)
        diffSeriesMetadata[col] = Dict(:d => integrationOrder, :D => seasonalIntegrationOrder)
    end

    diffSeries = merge(diffSeriesVector)
    return diffSeries, diffSeriesMetadata
end


"""
    loglikelihood(model::SarimaxModel)

Calculate the log-likelihood of a SARIMAModel.
The log-likelihood is calculated based on the formula
`-0.5 * (T * log(2π) + T * log(σ²) + sum(ϵ.^2 ./ σ²))`
where:
- T is the length of the residuals vector (ϵ).
- σ² is the estimated variance of the model.

# Arguments
- `model::SarimaxModel`: A SARIMAModel object.

# Returns
- The log-likelihood of the SARIMAModel.

# Errors
- `MissingMethodImplementation("fit!")`: Thrown if the `fit!` method is not implemented for the given model type.
- `ModelNotFitted()`: Thrown if the model has not been fitted.

"""
function loglikelihood(model::SarimaxModel)
    !hasFitMethods(typeof(model)) && throw(MissingMethodImplementation("fit!"))
    !isFitted(model) && throw(ModelNotFitted())
    T = length(model.ϵ)
    return -0.5 * (T * log(2π) + T * log(model.σ²) + sum(model.ϵ.^2 ./ model.σ²))
end

"""
    loglike(model::SarimaxModel)

Calculate the log-likelihood of a SARIMAModel using the normal probability density function.
The log-likelihood is calculated by summing the logarithm of the probability density function (PDF) of each data point under the assumption of a normal distribution with mean 0 and standard deviation equal to the square root of the estimated variance (σ²) of the model.

# Arguments
- `model::SarimaxModel`: A SARIMAModel object.

# Returns
- The log-likelihood of the SARIMAModel.

# Errors
- `MissingMethodImplementation("fit!")`: Thrown if the `fit!` method is not implemented for the given model type.
- `ModelNotFitted()`: Thrown if the model has not been fitted.

"""
function loglike(model::SarimaxModel)
    !hasFitMethods(typeof(model)) && throw(MissingMethodImplementation("fit!"))
    !isFitted(model) && throw(ModelNotFitted())
    return sum(logpdf.(Normal(0, sqrt(model.σ²)), model.ϵ))
end
