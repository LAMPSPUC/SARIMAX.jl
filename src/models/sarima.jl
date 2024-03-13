mutable struct SARIMAModel <: SarimaxModel
    y::TimeArray
    p::Int64
    d::Int64
    q::Int64
    seasonality::Int64
    P::Int64
    D::Int64
    Q::Int64
    c::Union{Float64,Nothing}
    trend::Union{Float64,Nothing}
    ϕ::Union{Vector{Float64},Nothing}
    θ::Union{Vector{Float64},Nothing}
    Φ::Union{Vector{Float64},Nothing}
    Θ::Union{Vector{Float64},Nothing}
    ϵ::Union{Vector{Float64},Nothing}
    σ²::Float64
    fitInSample::Union{TimeArray,Nothing}
    forecast::Union{Array{Float64},Nothing}
    silent::Bool
    allowMean::Bool
    allowDrift::Bool
    function SARIMAModel(y::TimeArray,
                        p::Int64,
                        d::Int64,
                        q::Int64;
                        seasonality::Int64=1,
                        P::Int64 = 0,
                        D::Int64 = 0,
                        Q::Int64 = 0,
                        c::Union{Float64,Nothing}=nothing,
                        trend::Union{Float64,Nothing}=nothing,
                        ϕ::Union{Vector{Float64},Nothing}=nothing,
                        θ::Union{Vector{Float64},Nothing}=nothing,
                        Φ::Union{Vector{Float64},Nothing}=nothing,
                        Θ::Union{Vector{Float64},Nothing}=nothing,
                        ϵ::Union{Vector{Float64},Nothing}=nothing,
                        σ²::Float64=0.0,
                        fitInSample::Union{TimeArray,Nothing}=nothing,
                        forecast::Union{TimeArray,Nothing}=nothing,
                        silent::Bool=true,
                        allowMean::Bool=true,
                        allowDrift::Bool=false)
        @assert p >= 0
        @assert d >= 0
        @assert q >= 0
        @assert P >= 0
        @assert D >= 0
        @assert Q >= 0
        @assert seasonality >= 1
        return new(y,p,d,q,seasonality,P,D,Q,c,trend,ϕ,θ,Φ,Θ,ϵ,σ²,fitInSample,forecast,silent,allowMean,allowDrift)
    end
end

function print(model::SARIMAModel)
    println("=================MODEL===============")
    println("SARIMA ($(model.p), $(model.d) ,$(model.q))($(model.P), $(model.D) ,$(model.Q) s=$(model.seasonality))")
    model.allowMean  && println("Estimated c       : ",model.c)
    model.allowDrift && println("Estimated trend   : ",model.trend)
    model.p != 0     && println("Estimated ϕ       : ", model.ϕ)
    model.q != 0     && println("Estimated θ       : ",model.θ)
    model.P != 0     && println("Estimated Φ       : ", model.Φ)
    model.Q != 0     && println("Estimated θ       : ",model.Θ)
    println("Residuals σ²      : ",model.σ²)
end

function SARIMA(y::TimeArray,
                p::Int64,
                d::Int64,
                q::Int64;
                seasonality::Int64=1,
                P::Int64 = 0,
                D::Int64 = 0,
                Q::Int64 = 0,
                silent::Bool=true,
                allowMean::Bool=true,
                allowDrift::Bool=false)
    return SARIMAModel(y,p,d,q;seasonality=seasonality,P=P,D=D,Q=Q,silent=silent,allowMean=allowMean,allowDrift=allowDrift)
end

function fillFitValues!(model::SARIMAModel,
                        c::Float64,
                        trend::Float64,
                        ϕ::Vector{Float64},
                        θ::Vector{Float64},
                        ϵ::Vector{Float64},
                        σ²::Float64,
                        fitInSample::TimeArray;
                        Φ::Union{Vector{Float64},Nothing}=nothing,
                        Θ::Union{Vector{Float64},Nothing}=nothing)
    model.c = c
    model.trend = trend
    model.ϕ = ϕ
    model.θ = θ
    model.ϵ = ϵ
    model.σ²= σ²
    model.Φ = Φ
    model.Θ = Θ
    model.fitInSample = fitInSample
end

function Base.copy(y::TimeArray)
    return TimeArray(copy(timestamp(y)),copy(values(y)))
end

function Base.deepcopy(y::TimeArray)
    return TimeArray(deepcopy(timestamp(y)),deepcopy(values(y)))
end


"""
    isFitted(
        model::SARIMAModel
    )

    Returns true if the SARIMA model has been fitted.

"""
function isFitted(model::SARIMAModel)
    hasResiduals = !isnothing(model.ϵ)
    hasFitInSample = !isnothing(model.fitInSample)
    estimatedAR = (model.p == 0) || !isnothing(model.ϕ)
    estimatedMA = (model.q == 0) || !isnothing(model.θ)
    estimatedSeasonalAR = (model.P == 0) || !isnothing(model.Φ)
    estimatedSeasonalMA = (model.Q == 0) || !isnothing(model.Θ)
    estimatedIntercept =  !model.allowMean || !isnothing(model.c)
    return hasResiduals && hasFitInSample && estimatedAR && estimatedMA && estimatedSeasonalAR && estimatedSeasonalMA && estimatedIntercept
end


"""
    getHyperparametersNumber(
        model::SARIMAModel
    )

    Returns the number of hyperparameters of a SARIMA model.

"""
function getHyperparametersNumber(model::SARIMAModel)
    k = (model.allowMean) ? 1 : 0
    k = (model.allowDrift) ? k + 1 : k
    return model.p + model.q + model.P + model.Q + k
end

"""
    fit!(
        model::SARIMAModel;
        silent::Bool=true,
        optimizer::DataType=Ipopt.Optimizer,
        objectiveFunction::String="mse"
    )

Estimate the Sarima model parameters via non linear least squares. The resulting optimal
parameters as well as the resisuals and the model σ² are stored within the model.
The default objective function used to estimate the parameters is the mean squared error (MSE)
but it can be changed to the maximum likelihood (ML) by setting the `objectiveFunction` parameter to "ml". 

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers,0,1,1;seasonality=12,P=0,D=1,Q=1)

julia> fit!(model)
```
"""
function fit!(model::SARIMAModel;silent::Bool=true,optimizer::DataType=Ipopt.Optimizer, objectiveFunction::String="mse")
    isFitted(model) && @info("The model has already been fitted. Overwriting the previous results")
    @assert objectiveFunction ∈ ["mse","ml","bilevel"] "The objective function $objectiveFunction is not supported. Please use 'mse', 'ml' or 'bilevel'"
    @assert model.d <= 1 "The estimation only works with d <= 1. Soon this will be fixed"
    @assert model.D <= 1 "The estimation only works with D <= 1. Soon this will be fixed"
    
    diffY = differentiate(model.y,model.d,model.D, model.seasonality)

    T = length(diffY)

    yValues = values(diffY)

    mod = Model(optimizer)
    if silent 
        set_silent(mod)
    end

    if (model.allowMean)
        @variable(mod,c)
    else
        @variable(mod,c in Parameter(1.0))
        set_parameter_value(mod[:c], 0.0)
    end

    if (model.allowDrift)
        @variable(mod,trend)
    else
        @variable(mod,trend in Parameter(1.0))
        set_parameter_value(mod[:trend], 0.0)
    end

    @variable(mod,-1 <= ϕ[1:model.p] <= 1)
    @variable(mod,-1 <= Φ[1:model.P] <= 1)
    @variable(mod,ϵ[1:T])
    
    if MACoefficientsAreModelParameters(objectiveFunction)
        @variable(mod,θ[i=1:model.q] in Parameter(i))
        @variable(mod,Θ[i=1:model.Q] in Parameter(i))
    else
        @variable(mod,-1 <= θ[1:model.q] <= 1)
        @variable(mod,-1 <= Θ[1:model.Q] <= 1)
        for i in 1:model.q 
            set_start_value(mod[:θ][i], 0.0) 
        end
        
        for i in 1:model.Q 
            set_start_value(mod[:Θ][i], 0.0) 
        end
    end

    includeSolverParameters!(mod)
    
    lb = max(model.p,model.q,model.P*model.seasonality,model.Q*model.seasonality) + 1
    fix.(ϵ[1:lb-1],0.0)

    objectiveFunctionDefinition!(mod, objectiveFunction, T, lb)

    if model.seasonality > 1
        @expression(mod, ŷ[t=lb:T], c + trend*t + sum(ϕ[i]*yValues[t - i] for i=1:model.p) + sum(θ[j]*ϵ[t - j] for j=1:model.q) + sum(Φ[k]*yValues[t - (model.seasonality*k)] for k=1:model.P) + sum(Θ[w]*ϵ[t - (model.seasonality*w)] for w=1:model.Q))
    else
        @expression(mod, ŷ[t=lb:T], c + trend*t +sum(ϕ[i]*yValues[t - i] for i=1:model.p) + sum(θ[j]*ϵ[t - j] for j=1:model.q))
    end
    @constraint(mod, [t=lb:T], yValues[t] == ŷ[t] + ϵ[t])

    optimizeModel!(mod, model, objectiveFunction)
    
    # TODO: - The reconciliation works for just d,D <= 1
    fitInSample::TimeArray = TimeArray(timestamp(diffY)[lb:end], OffsetArrays.no_offset_view(value.(ŷ)))
    
    if model.d > 0 # We differenciated the timeseries
        # Δyₜ = yₜ - y_t-1 => yₜ = Δyₜ + y_t-1
        fittedValues = values(fitInSample)
        yOriginal = values(model.y)
        for _=1:model.d
            originalIndex = findfirst(ts -> ts == timestamp(fitInSample)[1], timestamp(model.y))
            for j=1:length(fitInSample)
                fittedValues[j] += yOriginal[originalIndex+(j-1)-1]
            end
        end
        fitInSample = TimeArray(timestamp(fitInSample), fittedValues)
    end

    if model.D > 0 # We differenciated the timeseries
        # Δyₜ = yₜ - y_t-12 => yₜ = Δyₜ + y_t-12
        fittedValues = values(fitInSample)
        yOriginal = values(model.y)
        for i=1:model.D
            originalIndex = findfirst(ts -> ts == timestamp(fitInSample)[1], timestamp(model.y))
            for j=1:length(fitInSample)
                fittedValues[j] += yOriginal[originalIndex+(j-1)-model.seasonality*i]
            end
        end
        fitInSample = TimeArray(timestamp(fitInSample), fittedValues)
    end

    if model.D > 0 && model.d > 0 # We differenciated the timeseries
        fittedValues = values(fitInSample)
        yOriginal = values(model.y)
        for j=1:length(fitInSample)
            originalIndex = findfirst(ts -> ts == timestamp(fitInSample)[1], timestamp(model.y))
            fittedValues[j] -= yOriginal[originalIndex+(j-1)-(model.seasonality+1)]
        end
        fitInSample = TimeArray(timestamp(fitInSample), fittedValues)
    end

    residualsVariance = computeSARIMAModelVariance(mod, lb, objectiveFunction)

    c = is_valid(mod, c) ? value(c) : 0.0
    trend = is_valid(mod, trend) ? value(trend) : 0.0

    fillFitValues!(model,c,trend,value.(ϕ),value.(θ),value.(ϵ)[lb:end],residualsVariance,fitInSample;Φ=value.(Φ),Θ=value.(Θ))
end

function MACoefficientsAreModelParameters(objectiveFunction::String)
    return objectiveFunction == "bilevel"
end

function includeSolverParameters!(model::Model)
    if solver_name(model) == "Gurobi"
        set_optimizer_attribute(mod, "NonConvex", 2)
    end
end
    
"""
    objectiveFunctionDefinition!(
        model::Model,
        objectiveFunction::String,
        T::Int,
        lb::Int
    )
"""
function objectiveFunctionDefinition!(model::Model, objectiveFunction::String, T::Int, lb::Int)
    if objectiveFunction == "mse"
        @objective(model, Min, mean(model[:ϵ].^2))
    elseif objectiveFunction == "bilevel"
        @objective(model, Min, mean(model[:ϵ].^2))
        set_time_limit_sec(model, 1.0)
    elseif objectiveFunction == "ml"
        # llk(ϵ,μ,σ) = logpdf(Normal(μ,abs(σ)),ϵ)
        # register(model, :llk, 3, llk, autodiff=true)
        # @NLobjective( model, Max, sum(llk(ϵ[t],μ,σ) for t=lb:T))
        @variable(model, μ, start = 0.0)
        @variable(model, σ >= 0.0, start = 1.0)
        @constraint(model,0 <= μ <= 0.0) 
        @NLobjective( model, Max,((T-lb)/2) * log(1 / (2*π*σ*σ)) - sum((model[:ϵ][t] - μ)^2 for t in lb:T) / (2*σ*σ))
    end
end

function optimizeModel!(jumpModel::Model, model::SARIMAModel, objectiveFunction::String)
    JuMP.optimize!(jumpModel)

    if objectiveFunction == "bilevel"
        
        function optimizeMA(coefficients)
            maCoefficients = coefficients[1:model.q]
            smaCoefficients = coefficients[model.q+1:end]
            set_parameter_value.(jumpModel[:θ],maCoefficients)
            set_parameter_value.(jumpModel[:Θ],smaCoefficients)
            JuMP.optimize!(jumpModel)
            return objective_value(jumpModel)
        end
    
        if model.q + model.Q > 0
            ma_lower_bound = -1 .* ones(model.q+model.Q)
            ma_upper_bound = ones(model.q+model.Q)
            initialCoefficients = zeros(model.q+model.Q)# vcat(parameter_value.(θ),parameter_value.(Θ))# 
            results = Optim.optimize(optimizeMA, ma_lower_bound, ma_upper_bound, initialCoefficients)
            #results = Optim.optimize(optimizeMA,initialCoefficients,LBFGS(),Optim.Options(time_limit=60))
            if !Optim.converged(results)
                @warn("The optimization did not converge")
                @warn("Trying another method")
                results = Optim.optimize(optimizeMA, initialCoefficients, Optim.NelderMead())
                println(Optim.converged(results))
                Optim.converged(results) || @warn("The optimization did not converge")
            end
        end
    end
end

function computeSARIMAModelVariance(model::Model, lb::Int, objectiveFunction::String)
    if objectiveFunction == "ml"
        return value(model[:σ])^2
    end

    return var(value.(model[:ϵ])[lb:end])
end

"""
    predict!(
        model::SARIMAModel,
        stepsAhead::Int64=1
    )

Predicts the SARIMA model for the next `stepsAhead` periods.
The resulting forecast is stored within the model in the `forecast` field.

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers,0,1,1;seasonality=12,P=0,D=1,Q=1)

julia> fit!(model)

julia> predict!(model; stepsAhead=12)
```
"""
function predict!(model::SARIMAModel, stepsAhead::Int64=1)
    !isFitted(model) && throw(ModelNotFitted())

    diffY = differentiate(model.y,model.d,model.D,model.seasonality)
    yValues::Vector{Float64} = deepcopy(values(diffY))
    errors = deepcopy(model.ϵ)
    t = length(yValues)
    for _= 1:stepsAhead
        forecastedValue = model.c + model.trend*(t+stepsAhead)
        if model.p > 0
            # ∑ϕᵢyₜ -i
            forecastedValue += sum(model.ϕ[i]*yValues[end-i+1] for i=1:model.p)
        end
        if model.q > 0
            # ∑θᵢϵₜ-i
            forecastedValue += sum(model.θ[j]*errors[end-j+1] for j=1:model.q)
        end
        if model.P > 0
            # ∑Φₖyₜ-(s*k)
            forecastedValue += sum(model.Φ[k]*yValues[end-(model.seasonality*k)+1] for k=1:model.P)
        end
        if model.Q > 0
            # ∑Θₖϵₜ-(s*k)
            forecastedValue += sum(model.Θ[w]*errors[end-(model.seasonality*w)+1] for w=1:model.Q)
        end
        push!(errors, 0)
        push!(yValues, forecastedValue)
    end
    forecast_values = integrate(model.y, yValues[end-stepsAhead+1:end], model.d, model.D, model.seasonality)
    model.forecast = forecast_values
end


"""
    predict(
        model::SARIMAModel, 
        stepsAhead::Int64 = 1, 
    )

Predicts the SARIMA model for the next `stepsAhead` periods assuming that the model`s estimated σ².
Returns the forecasted values.

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers,0,1,1;seasonality=12,P=0,D=1,Q=1)

julia> fit!(model)

julia> forecastedValues = predict(model, stepsAhead=12)
```
"""
function predict(model::SARIMAModel, stepsAhead::Int64=1)
    !isFitted(model) && throw(ModelNotFitted())

    diffY = differentiate(model.y,model.d,model.D,model.seasonality)
    yValues::Vector{Float64} = deepcopy(values(diffY))
    errors = deepcopy(model.ϵ)
    t = length(yValues)
    for _= 1:stepsAhead
        forecastedValue = model.c + model.trend*(t+stepsAhead)
        if model.p > 0
            # ∑ϕᵢyₜ -i
            forecastedValue += sum(model.ϕ[i]*yValues[end-i+1] for i=1:model.p)
        end
        if model.q > 0
            # ∑θᵢϵₜ-i
            forecastedValue += sum(model.θ[j]*errors[end-j+1] for j=1:model.q)
        end
        if model.P > 0
            # ∑Φₖyₜ-(s*k)
            forecastedValue += sum(model.Φ[k]*yValues[end-(model.seasonality*k)+1] for k=1:model.P)
        end
        if model.Q > 0
            # ∑Θₖϵₜ-(s*k)
            forecastedValue += sum(model.Θ[w]*errors[end-(model.seasonality*w)+1] for w=1:model.Q)
        end
        ϵₜ = rand(Normal(0,sqrt(model.σ²)))
        forecastedValue += ϵₜ
        push!(errors, ϵₜ)
        push!(yValues, forecastedValue)
    end
    forecast_values = integrate(model.y, yValues[end-stepsAhead+1:end], model.d, model.D, model.seasonality)
    return forecast_values
end


"""
    simulate(
        model::SARIMAModel, 
        stepsAhead::Int64 = 1, 
        numScenarios::Int64 = 200
    )

Simulates the SARIMA model for the next `stepsAhead` periods assuming that the model`s estimated σ².
Returns a vector of `numScenarios` scenarios of the forecasted values.

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers,0,1,1;seasonality=12,P=0,D=1,Q=1)

julia> fit!(model)

julia> scenarios = simulate(model, stepsAhead=12, numScenarios=1000)
```
"""
function simulate(model::SARIMAModel, stepsAhead::Int64=1, numScenarios::Int64=200)
    !isFitted(model) && throw(ModelNotFitted())
    scenarios::Vector{Vector{Float64}} = []
    for _=1:numScenarios
        push!(scenarios, predict(model, stepsAhead))
    end
    return scenarios
end



"""
    auto(
        y::TimeArray;
        seasonality::Int64=1,
        d::Int64 = -1,
        D::Int64 = -1,
        maxp::Int64 = 5,
        maxd::Int64 = 2,
        maxq::Int64 = 5,
        maxP::Int64 = 2,
        maxD::Int64 = 1,
        maxQ::Int64 = 2,
        informationCriteria::String = "aicc",
        allowMean::Bool = true,
        integrationTest::String = "kpss",
        seasonalIntegrationTest::String = "seas",
        objectiveFunction::String = "mse"
    )

Automatically fits the best [`SARIMA`](@ref) model according to the best information criteria 


# References
* Hyndman, RJ and Khandakar. 
Automatic time series forecasting: The forecast package for R.
Journal of Statistical Software, 26(3), 2008.

"""
function auto(
    y::TimeArray;
    seasonality::Int64=1,
    d::Int64 = -1,
    D::Int64 = -1,
    maxp::Int64 = 5,
    maxd::Int64 = 2,
    maxq::Int64 = 5,
    maxP::Int64 = 2,
    maxD::Int64 = 1,
    maxQ::Int64 = 2,
    informationCriteria::String = "aicc",
    allowMean::Bool = true,
    allowDrift::Bool = true,
    integrationTest::String = "kpss",
    seasonalIntegrationTest::String = "seas",
    objectiveFunction::String = "mse",
    assertStationarity::Bool = false,
    assertInvertibility::Bool = false
)
    @assert seasonality >= 1 "seasonality must be greater than 1. Use 1 for non seasonal models"
    @assert d >= -1 
    @assert d <= maxd
    @assert D >= -1
    @assert D <= maxD
    @assert maxp >= 0
    @assert maxd >= 0
    @assert maxq >= 0
    @assert maxP >= 0
    @assert maxD >= 0
    @assert maxQ >= 0
    @assert informationCriteria ∈ ["aic","aicc","bic"]
    @assert integrationTest ∈ ["kpss"]
    @assert seasonalIntegrationTest ∈ ["seas","ch"]
    @assert objectiveFunction ∈ ["mse","ml","bilevel"] 

    informationCriteriaFunction = getInformationCriteriaFunction(informationCriteria)

    if seasonality == 1
        D = 0
    end

    if D < 0
        D = selectSeasonalIntegrationOrder(deepcopy(values(y)) ,seasonality,seasonalIntegrationTest)
    end

    if d < 0 
        d = selectIntegrationOrder(deepcopy(values(y)), maxd, D, seasonality, integrationTest)
    end

    allowMean = allowMean && (d+D == 0)
    allowDrift = allowDrift && (d+D == 1)

    # Include intial models
    candidateModels = Vector{SARIMAModel}()
    visitedModels = Dict{String,Dict{String,Any}}()

    if seasonality == 1
        initialNonSeasonalModels!(candidateModels, y, maxp, d, maxq, allowMean, allowDrift)
    else
        initialSeasonalModels!(candidateModels, y, maxp, d, maxq, maxP, D, maxQ, seasonality, allowMean, allowDrift)
    end

    # Fit models
    bestCriteria, bestModel = localSearch!(candidateModels, visitedModels, informationCriteriaFunction, objectiveFunction, assertStationarity, assertInvertibility)
    
    ITERATION_LIMIT = 100
    iterations = 1
    while iterations <= ITERATION_LIMIT

        addNonSeasonalModels!(bestModel, candidateModels, visitedModels, maxp, maxq, allowMean, allowDrift)
        (seasonality > 1) && addSeasonalModels!(bestModel, candidateModels, visitedModels, maxP, maxQ, allowMean, allowDrift)
        (d+D == 0) && addChangedConstantModel!(bestModel, candidateModels, visitedModels)
        (d+D == 1) && addChangedConstantModel!(bestModel, candidateModels, visitedModels,true)

        itBestCriteria, itBestModel = localSearch!(candidateModels, visitedModels, informationCriteriaFunction, objectiveFunction, assertStationarity, assertInvertibility)
        
        (itBestCriteria > bestCriteria) && break
        bestCriteria = itBestCriteria
        bestModel = itBestModel

        iterations += 1
    end
    @info("The best model found is $(getId(bestModel)) with $(iterations) iterations")

    return bestModel
end

function getInformationCriteriaFunction(informationCriteria)
    if informationCriteria == "aic"
        return aic
    elseif informationCriteria == "aicc"
        return aicc
    elseif informationCriteria == "bic"
        return bic
    end
    throw(ArgumentError("The information criteria $informationCriteria is not supported"))
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

function initialNonSeasonalModels!(
    models::Vector{SARIMAModel}, 
    y::TimeArray, 
    maxp::Int64, 
    d::Int64, 
    maxq::Int64, 
    allowMean::Bool,
    allowDrift::Bool
)
    push!(models, SARIMA(y,0,d,0;allowMean=allowMean,allowDrift=allowDrift))
    (maxp >= 1) && push!(models, SARIMA(y,1,d,0;allowMean=allowMean,allowDrift=allowDrift))
    (maxq >= 1) && push!(models, SARIMA(y,0,d,1;allowMean=allowMean,allowDrift=allowDrift))
    (maxp >= 2 && maxq >= 2) && push!(models, SARIMA(y,2,d,2;allowMean=allowMean,allowDrift=allowDrift))
end

function initialSeasonalModels!(
    models::Vector{SARIMAModel}, 
    y::TimeArray, 
    maxp::Int64, 
    d::Int64, 
    maxq::Int64, 
    maxP::Int64, 
    D::Int64, 
    maxQ::Int64, 
    seasonality::Int64, 
    allowMean::Bool,
    allowDrift::Bool
)
    push!(models, SARIMA(y,0,d,0;seasonality=seasonality,P=0,D=D,Q=0,allowMean=allowMean,allowDrift=allowDrift))
    (maxp >= 1 && maxP >= 1) && push!(models, SARIMA(y,1,d,0;seasonality=seasonality,P=1,D=D,Q=0, allowMean=allowMean,allowDrift=allowDrift))
    (maxq >= 1 && maxQ >= 1) && push!(models, SARIMA(y,0,d,1;seasonality=seasonality,P=0,D=D,Q=1,allowMean=allowMean,allowDrift=allowDrift))
    (maxp >= 2 && maxq >= 2 && maxP >= 1 && maxQ >= 1) && push!(models, SARIMA(y,2,d,2;seasonality=seasonality,P=1,D=D,Q=1,allowMean=allowMean,allowDrift=allowDrift))
end

function getId(
    model::SARIMAModel
)
    return "SARIMA($(model.p),$(model.d),$(model.q))($(model.P),$(model.D),$(model.Q) s=$(model.seasonality), c=$(model.allowMean), drift=$(model.allowDrift))"
end

function isVisited(model::SARIMAModel, visitedModels::Dict{String,Dict{String,Any}})
    id = getId(model)
    return haskey(visitedModels,id)
end

function localSearch!(
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},
    informationCriteriaFunction::Function,
    objectiveFunction::String = "mse",
    assertStationarity::Bool = false,
    assertInvertibility::Bool = false
)   
    localBestCriteria = Inf
    localBestModel = nothing
    foreach(model ->
        if !isFitted(model) 
            fit!(model;objectiveFunction=objectiveFunction)
            criteria = informationCriteriaFunction(model)
            @info("Fitted $(getId(model)) with $(criteria)")
            visitedModels[getId(model)] = Dict(
                "criteria" => criteria
            )

            if criteria < localBestCriteria 
                invertible = !assertInvertibility || StateSpaceModels.assert_invertibility(model.θ)
                invertible || @info("The model $(getId(model)) is not invertible")
                stationarity = !assertStationarity || StateSpaceModels.assert_stationarity(model.ϕ)
                stationarity || @info("The model $(getId(model)) is not stationary")
                (!invertible || !stationarity) && @info("The model will not be considered")
                if invertible && stationarity
                    localBestCriteria = criteria
                    localBestModel = model
                end
            end
        end
    , candidateModels)
    return localBestCriteria, localBestModel
end

function addNonSeasonalModels!(
    bestModel::SARIMAModel, 
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},  
    maxp::Int64, 
    maxq::Int64, 
    allowMean::Bool,
    allowDrift::Bool
)
    for p in -1:1, q in -1:1
        newp = bestModel.p + p
        newq = bestModel.q + q
        if newp < 0 || newq < 0 || newp > maxp || newq > maxq
            continue
        end

        newModel = SARIMA(
                    deepcopy(bestModel.y),
                    newp,
                    bestModel.d,
                    newq;
                    seasonality=bestModel.seasonality, 
                    P=bestModel.P,
                    D=bestModel.D,
                    Q=bestModel.Q,
                    allowMean=allowMean,
                    allowDrift=allowDrift
                )
        if !isVisited(newModel,visitedModels)
            push!(candidateModels, newModel)
        end
    end
end

function addSeasonalModels!(
    bestModel::SARIMAModel, 
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}}, 
    maxP::Int64, 
    maxQ::Int64, 
    allowMean::Bool,
    allowDrift::Bool
)
    for P in -1:1, Q in -1:1
        newP = bestModel.P + P
        newQ = bestModel.Q + Q
        if newP < 0 || newQ < 0 || newP > maxP || newQ > maxQ
            continue
        end

        newModel = SARIMA(
                    deepcopy(bestModel.y),
                    bestModel.p,
                    bestModel.d,
                    bestModel.q;
                    seasonality=bestModel.seasonality,
                    P=newP,
                    D=bestModel.D,
                    Q=newQ,
                    allowMean=allowMean,
                    allowDrift=allowDrift
                )
        if !isVisited(newModel,visitedModels)
            push!(candidateModels, newModel)
        end
    end
end

function addChangedConstantModel!(
    bestModel::SARIMAModel,
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},
    drift::Bool = false
)   
    allowDrift = drift && !bestModel.allowDrift
    allowMean = !drift && !bestModel.allowMean
    newModel = SARIMA(deepcopy(bestModel.y),bestModel.p,bestModel.d,bestModel.q;seasonality=bestModel.seasonality,P=bestModel.P,D=bestModel.D,Q=bestModel.Q,allowMean=allowMean,allowDrift=allowDrift)
    if !isVisited(newModel,visitedModels)
        push!(candidateModels, newModel)
    end
end
