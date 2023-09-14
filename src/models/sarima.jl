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
    ϕ::Union{Vector{Float64},Nothing}
    θ::Union{Vector{Float64},Nothing}
    Φ::Union{Vector{Float64},Nothing}
    Θ::Union{Vector{Float64},Nothing}
    ϵ::Union{Vector{Float64},Nothing}
    σ²::Float64
    fitInSample::Union{TimeArray,Nothing}
    forecast::Union{Array{Float64},Nothing}
    silent::Bool
    function SARIMAModel(y::TimeArray,
                        p::Int64,
                        d::Int64,
                        q::Int64;
                        seasonality::Int64=1,
                        P::Int64 = 0,
                        D::Int64 = 0,
                        Q::Int64 = 0,
                        c::Union{Float64,Nothing}=nothing,
                        ϕ::Union{Vector{Float64},Nothing}=nothing,
                        θ::Union{Vector{Float64},Nothing}=nothing,
                        Φ::Union{Vector{Float64},Nothing}=nothing,
                        Θ::Union{Vector{Float64},Nothing}=nothing,
                        ϵ::Union{Vector{Float64},Nothing}=nothing,
                        σ²::Float64=0.0,
                        fitInSample::Union{TimeArray,Nothing}=nothing,
                        forecast::Union{TimeArray,Nothing}=nothing,
                        silent::Bool=true)
        @assert p >= 0
        @assert d >= 0
        @assert q >= 0
        @assert P >= 0
        @assert D >= 0
        @assert Q >= 0
        @assert seasonality >= 1
        return new(y,p,d,q,seasonality,P,D,Q,c,ϕ,θ,Φ,Θ,ϵ,σ²,fitInSample,forecast,silent)
    end
end

function print(model::SARIMAModel)
    println("=================MODEL===============")
    println("SARIMA ($(model.p), $(model.d) ,$(model.q))($(model.P), $(model.D) ,$(model.Q) s=$(model.seasonality))")
    println("Estimated c       : ",model.c)
    println("Estimated ϕ       : ", model.ϕ)
    println("Estimated θ       : ",model.θ)
    println("Estimated Φ       : ", model.Φ)
    println("Estimated θ       : ",model.Θ)
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
                silent::Bool=true)
    return SARIMAModel(y,p,d,q;seasonality,P,D,Q,silent)
end

function fillFitValues!(model::SARIMAModel,
                        c::Float64,
                        ϕ::Vector{Float64},
                        θ::Vector{Float64},
                        ϵ::Vector{Float64},
                        σ²::Float64,
                        fitInSample::TimeArray;
                        Φ::Union{Vector{Float64},Nothing}=nothing,
                        Θ::Union{Vector{Float64},Nothing}=nothing)
    model.c = c
    model.ϕ = ϕ
    model.θ = θ
    model.ϵ = ϵ
    model.σ²= σ²
    model.Φ = Φ
    model.Θ = Θ
    model.fitInSample = fitInSample
end

function copy(y::TimeArray)
    return TimeArray(copy(timestamp(y)),copy(values(y)))
end

"""
    fit!(
        model::SARIMAModel;
        silent::Bool=true,
        optimizer::DataType=Ipopt.Optimizer,
        normalize::Bool=false
    )

Estimate the Sarima model parameters via non linear least squares. The resulting optimal
parameters as well as the resisuals and the model σ² are stored within the model. 

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers,0,1,1;seasonality=12,P=0,D=1,Q=1)

julia> fit!(model)
```
"""
function fit!(model::SARIMAModel;silent::Bool=true,optimizer::DataType=Ipopt.Optimizer, normalize::Bool=false)
    diffY = differentiate(model.y,model.d,model.D, model.seasonality)

    T = length(diffY)

    # Normalizing arrays 
    if normalize
        @info("Normalizing time series")
        diffY = (diffY .- mean(values(diffY)))./std(values(diffY)) 
    end

    yValues = values(diffY)

    mod = Model(optimizer)
    if silent 
        set_silent(mod)
    end
    
    @variable(mod,ϕ[1:model.p])
    @variable(mod,θ[1:model.q])
    @variable(mod,Φ[1:model.P])
    @variable(mod,Θ[1:model.Q])
    @variable(mod,ϵ[1:T])
    @variable(mod,c)
    
    for i in 1:model.q 
        set_start_value(mod[:θ][i], 0) 
    end
    
    for i in 1:model.Q 
        set_start_value(mod[:Θ][i], 0) 
    end

    @objective(mod, Min, mean(ϵ.^2))# + 0.1*(sum(θ.^2)+sum(Θ.^2)))
    lb = max(model.p,model.q,model.P*model.seasonality,model.Q*model.seasonality) + 1
    fix.(ϵ[1:lb-1],0.0)
    if model.seasonality > 1
        @expression(mod, ŷ[t=lb:T], c + sum(ϕ[i]*yValues[t - i] for i=1:model.p) + sum(θ[j]*ϵ[t - j] for j=1:model.q) + sum(Φ[k]*yValues[t - (model.seasonality*k)] for k=1:model.P) + sum(Θ[w]*ϵ[t - (model.seasonality*w)] for w=1:model.Q))
    else
        @expression(mod, ŷ[t=lb:T], c + sum(ϕ[i]*yValues[t - i] for i=1:model.p) + sum(θ[j]*ϵ[t - j] for j=1:model.q))
    end
    @constraint(mod, [t=lb:T], yValues[t] == ŷ[t] + ϵ[t])
    optimize!(mod)
    termination_status(mod)
    
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
    residualsVariance = var(value.(ϵ)[lb:end])
    fillFitValues!(model,value(c),value.(ϕ),value.(θ),value.(ϵ)[lb:end],residualsVariance,fitInSample;Φ=value.(Φ),Θ=value.(Θ))
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
    diffY = differentiate(model.y,model.d,model.D,model.seasonality)
    yValues::Vector{Float64} = deepcopy(values(diffY))
    errors = deepcopy(model.ϵ)
    for _= 1:stepsAhead
        forecastedValue = model.c
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
    diffY = differentiate(model.y,model.d,model.D,model.seasonality)
    yValues::Vector{Float64} = deepcopy(values(diffY))
    errors = deepcopy(model.ϵ)
    for _= 1:stepsAhead
        forecastedValue = model.c
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
        ϵₜ = rand(Normal(0,sqrt(σ²)))
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
    scenarios::Vector{Vector{Float64}} = []
    for _=1:numScenarios
        push!(scenarios, predict(model, stepsAhead))
    end
    return scenarios
end