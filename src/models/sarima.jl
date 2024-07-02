mutable struct SARIMAModel{Fl <: AbstractFloat} <: SarimaxModel
    y::TimeArray
    p::Int
    d::Int
    q::Int
    seasonality::Int
    P::Int
    D::Int
    Q::Int
    metadata::Dict{String,Any}
    exog::Union{TimeArray,Nothing}
    c::Union{Fl,Nothing}
    trend::Union{Fl,Nothing}
    ϕ::Union{Vector{Fl},Nothing}
    θ::Union{Vector{Fl},Nothing}
    Φ::Union{Vector{Fl},Nothing}
    Θ::Union{Vector{Fl},Nothing}
    ϵ::Union{Vector{Fl},Nothing}
    exogCoefficients::Union{Vector{Fl},Nothing}
    σ²::Fl
    fitInSample::Union{TimeArray,Nothing}
    forecast::Union{TimeArray,Nothing}
    silent::Bool
    allowMean::Bool
    allowDrift::Bool
    keepProvidedCoefficients::Bool
    function SARIMAModel{Fl}(
                        y::TimeArray,
                        p::Int,
                        d::Int,
                        q::Int;
                        seasonality::Int=1,
                        P::Int = 0,
                        D::Int = 0,
                        Q::Int = 0,
                        exog::Union{TimeArray,Nothing}=nothing,
                        c::Union{Fl,Nothing}=nothing,
                        trend::Union{Fl,Nothing}=nothing,
                        ϕ::Union{Vector{Fl},Nothing}=nothing,
                        θ::Union{Vector{Fl},Nothing}=nothing,
                        Φ::Union{Vector{Fl},Nothing}=nothing,
                        Θ::Union{Vector{Fl},Nothing}=nothing,
                        ϵ::Union{Vector{Fl},Nothing}=nothing,
                        exogCoefficients::Union{Vector{Fl},Nothing}=nothing,
                        σ²::Fl=0.0,
                        fitInSample::Union{TimeArray,Nothing}=nothing,
                        forecast::Union{TimeArray,Nothing}=nothing,
                        silent::Bool=true,
                        allowMean::Bool=true,
                        allowDrift::Bool=false,
                        keepProvidedCoefficients::Bool=false
    ) where Fl<:AbstractFloat
        @assert p >= 0
        @assert d >= 0
        @assert q >= 0
        @assert P >= 0
        @assert D >= 0
        @assert Q >= 0
        @assert seasonality >= 1
        yMetadata = Dict()
        granularityInfo = identifyGranularity(timestamp(y))
        yMetadata["granularity"] = granularityInfo.granularity
        yMetadata["frequency"] = granularityInfo.frequency
        yMetadata["weekDaysOnly"] = granularityInfo.weekdays
        yMetadata["startDatetime"] = timestamp(y)[1]
        yMetadata["endDatetime"] = timestamp(y)[end]
        if !isnothing(exog)
            @assert yMetadata["startDatetime"] == timestamp(exog)[1] "The endogenous and exogenous variables must start at the same timestamp"
            @assert yMetadata["endDatetime"] <= timestamp(exog)[end] "The exogenous variables must end after the endogenous variables"
            @assert granularityInfo == identifyGranularity(timestamp(exog)) "The endogenous and exogenous variables must have the same granularity, frequency and pattern"
        end
        return new{Fl}(y,p,d,q,seasonality,P,D,Q,yMetadata,exog,c,trend,ϕ,θ,Φ,Θ,ϵ,exogCoefficients,σ²,fitInSample,forecast,silent,allowMean,allowDrift,keepProvidedCoefficients)
    end
end

typeofModelElements(model::SARIMAModel) = eltype(values(model.y))

function print(model::SARIMAModel)
    println("=================MODEL===============")
    println("SARIMA ($(model.p), $(model.d) ,$(model.q))($(model.P), $(model.D) ,$(model.Q) s=$(model.seasonality))")
    model.allowMean       && println("Estimated c       : ",model.c)
    model.allowDrift      && println("Estimated trend   : ",model.trend)
    model.p != 0          && println("Estimated ϕ       : ", model.ϕ)
    model.q != 0          && println("Estimated θ       : ",model.θ)
    model.P != 0          && println("Estimated Φ       : ", model.Φ)
    model.Q != 0          && println("Estimated θ       : ",model.Θ)
    isnothing(model.exog) || println("Exogenous coefficients: ",model.exogCoefficients)
    println("Residuals σ²      : ",model.σ²)
    model.keepProvidedCoefficients && println("The model preserves the provided coefficients. To optimize the whole model, set keepProvidedCoefficients=false")
    println("======================================")
end

function Base.show(io::IO, model::SARIMAModel)
    zeroMean = model.allowMean ? "non zero mean" : "zero mean"
    zeroDrift = model.allowDrift ? "non zero drift" : "zero drift"
    print(io, "SARIMA ($(model.p), $(model.d) ,$(model.q))($(model.P), $(model.D) ,$(model.Q) s=$(model.seasonality)) with $(zeroMean) and $(zeroDrift)")
    return nothing
end

function SARIMA(y::TimeArray,
                p::Int,
                d::Int,
                q::Int;
                seasonality::Int=1,
                P::Int = 0,
                D::Int = 0,
                Q::Int = 0,
                silent::Bool=true,
                allowMean::Bool=true,
                allowDrift::Bool=false)
        modelFl = eltype(values(y))
    return SARIMAModel{modelFl}(y,p,d,q;seasonality=seasonality,P=P,D=D,Q=Q,silent=silent,allowMean=allowMean,allowDrift=allowDrift)
end

function SARIMA(y::TimeArray;
                exog::Union{TimeArray,Nothing}=nothing,
                arCoefficients::Union{Vector{Fl},Nothing}=nothing,
                maCoefficients::Union{Vector{Fl},Nothing}=nothing,
                seasonalARCoefficients::Union{Vector{Fl},Nothing}=nothing,
                seasonalMACoefficients::Union{Vector{Fl},Nothing}=nothing,
                mean::Union{Fl,Nothing}=nothing,
                trend::Union{Fl,Nothing}=nothing,
                exogCoefficients::Union{Vector{Fl},Nothing}=nothing,
                d::Int = 0,
                D::Int = 0,
                seasonality::Int=1,
                silent::Bool=true,
                allowMean::Bool=true,
                allowDrift::Bool=false
    ) where Fl<:AbstractFloat

    if isnothing(arCoefficients) && isnothing(maCoefficients) && isnothing(seasonalARCoefficients) && isnothing(seasonalMACoefficients)
        throw(InvalidParametersCombination("At least one of the AR, MA, seasonal AR or seasonal MA coefficients must be provided"))
    end

    if (!isnothing(seasonalARCoefficients) || !isnothing(seasonalMACoefficients)) && seasonality == 1
        throw(InvalidParametersCombination("The seasonality must be provided if seasonal AR and/or MA coefficients are provided"))
    end

    if isnothing(exog) && !isnothing(exogCoefficients)
        throw(InvalidParametersCombination("Exogenous coefficients were provided but no exogenous variable was passed"))
    end

    if !isnothing(exog) && islength(colnames(exog)) != length(exogCoefficients)
        throw(InvalidParametersCombination("The number of exogenous coefficients must match the number of exogenous variables"))
    end

    p = isnothing(arCoefficients) ? 0 : length(arCoefficients)
    q = isnothing(maCoefficients) ? 0 : length(maCoefficients)
    P = isnothing(seasonalARCoefficients) ? 0 : length(seasonalARCoefficients)
    Q = isnothing(seasonalMACoefficients) ? 0 : length(seasonalMACoefficients)
    c = isnothing(mean) ? nothing : mean
    trend = isnothing(trend) ? nothing : trend
    allowMean = !isnothing(mean) || allowMean
    allowDrift = !isnothing(trend) || allowDrift

    return SARIMAModel{Fl}(y,p,d,q;seasonality=seasonality,P=P,D=D,Q=Q,exog=exog,c=c,trend=trend,ϕ=arCoefficients,θ=maCoefficients,Φ=seasonalARCoefficients,Θ=seasonalMACoefficients,exogCoefficients=exogCoefficients,silent=silent,allowMean=allowMean,allowDrift=allowDrift,keepProvidedCoefficients=true)
end

function SARIMA(y::TimeArray,
                exog::Union{TimeArray,Nothing},
                p::Int,
                d::Int,
                q::Int;
                seasonality::Int=1,
                P::Int = 0,
                D::Int = 0,
                Q::Int = 0,
                silent::Bool=true,
                allowMean::Bool=true,
                allowDrift::Bool=false) 
    modelFl = eltype(values(y))
    return SARIMAModel{modelFl}(y,p,d,q;seasonality=seasonality,P=P,D=D,Q=Q,exog=exog,silent=silent,allowMean=allowMean,allowDrift=allowDrift)
end

"""
    fillFitValues!(
        model::SARIMAModel,
        c::Fl,
        trend::Fl,
        ϕ::Vector{Fl},
        θ::Vector{Fl},
        ϵ::Vector{Fl},
        σ²::Fl,
        fitInSample::TimeArray;
        Φ::Union{Vector{Fl}, Nothing}=nothing,
        Θ::Union{Vector{Fl}, Nothing}=nothing,
        exogCoefficients::Union{Vector{Fl}, Nothing}=nothing
    ) where Fl<:AbstractFloat

Fills the SARIMA model with fitted values.

# Arguments
- `model::SARIMAModel`: The SARIMA model to be filled.
- `c::Fl`: The intercept value.
- `trend::Fl`: The trend value.
- `ϕ::Vector{Fl}`: The autoregressive coefficients.
- `θ::Vector{Fl}`: The moving average coefficients.
- `ϵ::Vector{Fl}`: The residuals.
- `σ²::Fl`: The model's σ².
- `fitInSample::TimeArray`: The fitted values.
- `Φ::Union{Vector{Fl}, Nothing}`: The seasonal autoregressive coefficients. Default is `nothing`.
- `Θ::Union{Vector{Fl}, Nothing}`: The seasonal moving average coefficients. Default is `nothing`.
- `exogCoefficients::Union{Vector{Fl}, Nothing}`: The exogenous variable coefficients. Default is `nothing`.

"""
function fillFitValues!(model::SARIMAModel,
                        c::Fl,
                        trend::Fl,
                        ϕ::Vector{Fl},
                        θ::Vector{Fl},
                        ϵ::Vector{Fl},
                        σ²::Fl,
                        fitInSample::TimeArray;
                        Φ::Union{Vector{Fl},Nothing}=nothing,
                        Θ::Union{Vector{Fl},Nothing}=nothing,
                        exogCoefficients::Union{Vector{Fl},Nothing}=nothing) where Fl<:AbstractFloat
    model.c = c
    model.trend = trend
    model.ϕ = ϕ
    model.θ = θ
    model.ϵ = ϵ
    model.σ²= σ²
    model.Φ = Φ
    model.Θ = Θ
    model.fitInSample = fitInSample
    model.exogCoefficients = exogCoefficients
end

"""
    isFitted(model::SARIMAModel)

Returns `true` if the SARIMA model has been fitted.

# Arguments
- `model::SARIMAModel`: The SARIMA model.

# Returns
- `Bool`: `true` if the model has been fitted; otherwise, `false`.

"""
function isFitted(model::SARIMAModel)
    hasResiduals = !isnothing(model.ϵ)
    hasFitInSample = !isnothing(model.fitInSample)
    estimatedAR = (model.p == 0) || !isnothing(model.ϕ)
    estimatedMA = (model.q == 0) || !isnothing(model.θ)
    estimatedSeasonalAR = (model.P == 0) || !isnothing(model.Φ)
    estimatedSeasonalMA = (model.Q == 0) || !isnothing(model.Θ)
    estimatedIntercept =  !model.allowMean || !isnothing(model.c)
    estimatedExog = isnothing(model.exog) || !isnothing(model.exogCoefficients)
    return hasResiduals && hasFitInSample && estimatedAR && estimatedMA && estimatedSeasonalAR && estimatedSeasonalMA && estimatedIntercept && estimatedExog
end

"""
    getHyperparametersNumber(model::SARIMAModel)

Returns the number of hyperparameters of a SARIMA model.

# Arguments
- `model::SARIMAModel`: The SARIMA model.

# Returns
- `Int`: The number of hyperparameters.

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

Estimate the SARIMA model parameters via non-linear least squares. The resulting optimal
parameters as well as the residuals and the model's σ² are stored within the model.
The default objective function used to estimate the parameters is the mean squared error (MSE),
but it can be changed to the maximum likelihood (ML) by setting the `objectiveFunction` parameter to "ml".

# Arguments
- `model::SARIMAModel`: The SARIMA model to be fitted.
- `silent::Bool`: Whether to suppress solver output. Default is `true`.
- `optimizer::DataType`: The optimizer to be used for optimization. Default is `Ipopt.Optimizer`.
- `objectiveFunction::String`: The objective function used for estimation. Default is "mse".

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers,0,1,1;seasonality=12,P=0,D=1,Q=1)

julia> fit!(model)
```
"""
function fit!(model::SARIMAModel;silent::Bool=true,optimizer::DataType=Ipopt.Optimizer, objectiveFunction::String="mse")
    Fl = typeofModelElements(model)
    isFitted(model) && @info("The model has already been fitted. Overwriting the previous results")
    @assert objectiveFunction ∈ ["mae","mse","ml","bilevel"] "The objective function $objectiveFunction is not supported. Please use 'mae', 'mse', 'ml' or 'bilevel'"
    
    diffY = differentiate(model.y,model.d,model.D, model.seasonality)
    
    if !isnothing(model.exog)
        diffExog, exogMetadata = automaticDifferentiation(model.exog;seasonalPeriod=model.seasonality)
        model.metadata["exog"] = exogMetadata
        diffY = TimeSeries.merge(diffY, diffExog)
    end

    T = length(diffY)

    yValues = values(diffY)[:,1]
    nExog = isnothing(model.exog) ? 0 : size(values(diffY),2) - 1
    exogValues = isnothing(model.exog) ? [] : values(diffY)[:,2:end]

    mod = Model(optimizer)

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

    @variable(mod,-1 <= β[1:nExog] <= 1)
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

    model.keepProvidedCoefficients && setProvidedCoefficients!(mod, model)
    includeSolverParameters!(mod, silent)
    
    lb = max(model.p,model.P*model.seasonality,model.q,model.Q*model.seasonality) + 1
    fix.(ϵ[1:lb-1],0.0)

    if model.seasonality > 1
        @expression(mod, ŷ[t=lb:T], c + trend + sum(β[i]*exogValues[t,i] for i=1:nExog) + sum(ϕ[i]*yValues[t - i] for i=1:model.p) + sum(θ[j]*ϵ[t - j] for j=1:model.q if (t-j > 0)) + sum(Φ[k]*yValues[t - (model.seasonality*k)] for k=1:model.P) + sum(Θ[w]*ϵ[t - (model.seasonality*w)] for w=1:model.Q if (t - (model.seasonality*w) > 0)))
    else
        @expression(mod, ŷ[t=lb:T], c + trend + sum(β[i]*exogValues[t,i] for i=1:nExog) + sum(ϕ[i]*yValues[t - i] for i=1:model.p) + sum(θ[j]*ϵ[t - j] for j=1:model.q if (t-j > 0)))
    end
    
    includeModelConstraints!(mod, yValues, T, lb, objectiveFunction)

    objectiveFunctionDefinition!(mod, model, objectiveFunction, T, lb)

    optimizeModel!(mod, model, objectiveFunction)

    fittedValues::Vector{Fl} = Vector(OffsetArrays.no_offset_view(value.(ŷ)))
    fittedOriginalLengthDifference = length(values(model.y)) - length(fittedValues)
    initialValuesLength = model.d + model.D*model.seasonality
    initialValuesOffset = fittedOriginalLengthDifference > initialValuesLength ? fittedOriginalLengthDifference - initialValuesLength + 1 : 1
    initialValues::Vector{Fl} = values(model.y)[initialValuesOffset:fittedOriginalLengthDifference]

    integratedFit = integrate(deepcopy(initialValues), deepcopy(fittedValues), model.d, model.D, model.seasonality)
    lengthIntegratedFit = length(integratedFit)
    fitInSample::TimeArray = TimeArray(timestamp(model.y)[end-lengthIntegratedFit+1:end],integratedFit)

    residualsVariance = computeSARIMAModelVariance(mod, lb, objectiveFunction)

    c = is_valid(mod, c) ? value(c) : 0.0
    trend = is_valid(mod, trend) ? value(trend) : 0.0
    exogCoefficients = isnothing(model.exog) ? nothing : value.(β) 

    fillFitValues!(model,c,trend,value.(ϕ),value.(θ),value.(ϵ)[lb:end],residualsVariance,fitInSample;Φ=value.(Φ),Θ=value.(Θ),exogCoefficients=exogCoefficients)
end

"""
    MACoefficientsAreModelParameters(objectiveFunction::String)

Determines if the moving average coefficients are treated as model parameters based on the objective function.

# Arguments
- `objectiveFunction::String`: The objective function used.

# Returns
- `Bool`: `true` if the moving average coefficients are treated as model parameters; otherwise, `false`.
"""
function MACoefficientsAreModelParameters(objectiveFunction::String)
    return objectiveFunction == "bilevel"
end

function getParametersVector(model::SARIMAModel)
    parametersVector::Vector{Symbol} = Vector{Symbol}()
    model.allowMean && push!(parametersVector, :c)
    model.allowDrift && push!(parametersVector, :trend)
    model.p > 0 && push!(parametersVector, :ϕ)
    model.q > 0 && push!(parametersVector, :θ)
    model.P > 0 && push!(parametersVector, :Φ)
    model.Q > 0 && push!(parametersVector, :Θ)
    !isnothing(model.exog) && push!(parametersVector, :β)
    return parametersVector
end

"""
    setProvidedCoefficients!(jumpModel::Model, model::SARIMAModel)

Sets the provided coefficient values from a `SARIMAModel` to the corresponding parameters in a `jumpModel`.

# Arguments
- `jumpModel::Model`: The target model where the coefficients will be set.
- `model::SARIMAModel`: The source model containing the coefficients.

# Description
This function assigns the provided coefficients from the `model` to the corresponding parameters in the `jumpModel` if they are not `nothing`.

# Details
- If `model.c` is not `nothing`, it sets `jumpModel[:c]` to `model.c`.
- If `model.trend` is not `nothing`, it sets `jumpModel[:trend]` to `model.trend`.
- If `model.ϕ` is not `nothing`, it sets `jumpModel[:ϕ]` to `model.ϕ`.
- If `model.θ` is not `nothing`, it sets `jumpModel[:θ]` to `model.θ`.
- If `model.Φ` is not `nothing`, it sets `jumpModel[:Φ]` to `model.Φ`.
- If `model.Θ` is not `nothing`, it sets `jumpModel[:Θ]` to `model.Θ`.
- If `model.exogCoefficients` is not `nothing`, it sets `jumpModel[:β]` to `model.exogCoefficients`.

"""
function setProvidedCoefficients!(jumpModel::Model, model::SARIMAModel)
    !isnothing(model.c) && fix(jumpModel[:c],model.c)
    !isnothing(model.trend) && fix(jumpModel[:trend],model.trend)
    !isnothing(model.ϕ) && fix.(jumpModel[:ϕ],model.ϕ; force=true)
    !isnothing(model.θ) && fix.(jumpModel[:θ],model.θ; force=true)
    !isnothing(model.Φ) && fix.(jumpModel[:Φ],model.Φ; force=true)
    !isnothing(model.Θ) && fix.(jumpModel[:Θ],model.Θ; force=true)
    !isnothing(model.exogCoefficients) && fix.(jumpModel[:β],model.exogCoefficients; force=true)
end

"""
    includeSolverParameters!(model::Model)

Includes solver-specific parameters in the JuMP model.

# Arguments
- `model::Model`: The JuMP model to which solver parameters will be included.

"""
function includeSolverParameters!(model::Model, isSilent::Bool=true)
    isSilent && solver_name(model) != "Alpine" && set_silent(model)
    if solver_name(model) == "Gurobi"
        set_optimizer_attribute(model, "NonConvex", 2)
    elseif solver_name(model) == "Alpine"
        ipopt = optimizer_with_attributes(Ipopt.Optimizer)
        highs = optimizer_with_attributes(HiGHS.Optimizer)
        set_optimizer_attribute(model, "nlp_solver", ipopt)
        set_optimizer_attribute(model, "mip_solver", highs)
    end
end

"""
    includeModelConstraints!(jumpModel::Model, yValues::Fl, T::Int, lb::Int, objectiveFunction::String) where Fl<:AbstractFloat

Includes the constraints in the JuMP model for the SARIMA model.

# Arguments
- `jumpModel::Model`: The JuMP model to which constraints will be included.
- `yValues::Fl`: The values of the time series.
- `T::Int`: The total number of observations.
- `lb::Int`: The lag from which to start considering observations.
- `objectiveFunction::String`: The objective function used for optimization.
"""
function includeModelConstraints!(jumpModel::Model, yValues::Vector{Fl}, T::Int, lb::Int, objectiveFunction::String) where Fl<:AbstractFloat
    if objectiveFunction == "mae"
        @variable(jumpModel, ϵ_plus[lb:T] >= 0)
        @variable(jumpModel, ϵ_minus[lb:T] >= 0)
        @constraint(jumpModel, [t=lb:T], jumpModel[:ϵ][t] == ϵ_plus[t] + ϵ_minus[t])
        @constraint(jumpModel, [t=lb:T], yValues[t] - jumpModel[:ŷ][t] <= ϵ_plus[t])
        @constraint(jumpModel, [t=lb:T], jumpModel[:ŷ][t] - yValues[t] <= - ϵ_minus[t])
    else
        @constraint(jumpModel, [t=lb:T], yValues[t] == jumpModel[:ŷ][t] + jumpModel[:ϵ][t])
    end
end
    
"""
    objectiveFunctionDefinition!(
        jumpModel::Model,
        model::SARIMAModel,
        objectiveFunction::String,
        T::Int,
        lb::Int
    )

Defines the objective function for optimization in the SARIMA model.

# Arguments
- `jumpModel::Model`: The JuMP model.
- `model::SARIMAModel`: The SARIMA model to be optimized.
- `objectiveFunction::String`: The objective function to be defined.
- `T::Int`: The total number of observations.
- `lb::Int`: The lag from which to start considering observations.

"""
function objectiveFunctionDefinition!(jumpModel::Model, model::SARIMAModel, objectiveFunction::String, T::Int, lb::Int)
    if objectiveFunction == "mse"
        @objective(jumpModel, Min, mean(jumpModel[:ϵ][lb:T].^2))
    elseif objectiveFunction == "mae"
        @objective(jumpModel, Min, sum(jumpModel[:ϵ_plus][t] + jumpModel[:ϵ_minus][t] for t=lb:T))
    elseif objectiveFunction == "bilevel"
        @objective(jumpModel, Min, mean(jumpModel[:ϵ][lb:T].^2))
        set_time_limit_sec(jumpModel, 1.0)
    elseif objectiveFunction == "lasso"
        parametersVector::Vector{Symbol} = getParametersVector(model)
        auxVariables = @variable(jumpModel, [i=1:length(parametersVector)])
        @constraints(jumpModel, begin
            [i=1:length(parametersVector)], auxVariables[i] >= 0
            [i=1:length(parametersVector)], auxVariables[i] >= jumpModel[parametersVector[i]]
            [i=1:length(parametersVector)], auxVariables[i] >= -jumpModel[parametersVector[i]]
        end)
        λ = 1/sqrt(T)
        @objective(jumpModel, Min, mean(jumpModel[:ϵ][lb:T].^2) + λ * sum(auxVariables[i]))
    elseif objectiveFunction == "ml"
        # llk(ϵ,μ,σ) = logpdf(Normal(μ,abs(σ)),ϵ)
        # register(jumpModel, :llk, 3, llk, autodiff=true)
        # @NLobjective( jumpModel, Max, sum(llk(ϵ[t],μ,σ) for t=lb:T))
        @variable(jumpModel, μ, start = 0.0)
        @variable(jumpModel, σ >= 0.0, start = 1.0)
        @constraint(jumpModel,0 <= μ <= 0.0) 
        @objective( jumpModel, Max,((T-lb)/2) * log(1 / (2*π*σ*σ)) - sum((jumpModel[:ϵ][t] - μ)^2 for t in lb:T) / (2*σ*σ))
    end
end

"""
    optimizeModel!(jumpModel::Model, model::SARIMAModel, objectiveFunction::String)

Optimizes the SARIMA model using the specified objective function.

# Arguments
- `jumpModel::Model`: The JuMP model to be optimized.
- `model::SARIMAModel`: The SARIMA model to be optimized.
- `objectiveFunction::String`: The objective function used for optimization.

"""
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

"""
    computeSARIMAModelVariance(model::Model, lb::Int, objectiveFunction::String)

Computes the variance of the SARIMA model's errors.

# Arguments
- `model::Model`: The SARIMA model.
- `lb::Int`: The lag from which to compute the variance.
- `objectiveFunction::String`: The objective function used for fitting the model.

# Returns
- `AbstractFloat`: The computed variance.

"""
function computeSARIMAModelVariance(model::Model, lb::Int, objectiveFunction::String)
    if objectiveFunction == "ml"
        return value(model[:σ])^2
    end

    return var(value.(model[:ϵ])[lb:end])
end

"""
    completeCoefficientsVector(model::SARIMAModel)

Complete the coefficient vectors for AR and MA parts of a SARIMA model.

# Arguments
- `model::SARIMAModel`: The SARIMA model containing the AR and MA coefficients, seasonal orders, and other model parameters.

# Returns
- `arCoefficients`: A vector of AR coefficients, extended to include seasonal AR coefficients.
- `maCoefficients`: A vector of MA coefficients, extended to include seasonal MA coefficients.

The function handles the seasonal components by zero-padding the coefficient vectors and placing the seasonal coefficients at the appropriate positions.
"""
function completeCoefficientsVector(model::SARIMAModel)
    ModelFl = typeofModelElements(model)
    maCoefficients = model.θ
    if model.Q > 0
        maCoefficients = zeros(ModelFl, model.Q * model.seasonality)
        maCoefficients[1:model.q] = model.θ
        for i in 1:model.Q
            maCoefficients[model.seasonality * i] = model.Θ[i]
        end
    end 

    arCoefficients = model.ϕ
    if model.P > 0
        arCoefficients = zeros(ModelFl, model.P * model.seasonality)
        arCoefficients[1:model.p] = model.ϕ
        for i in 1:model.P
            arCoefficients[model.seasonality * i] = model.Φ[i]
        end
    end

    return arCoefficients, maCoefficients
end

"""
    toMA(model::SARIMAModel, maxLags::Int=12)

    Convert a SARIMA model to a Moving Average (MA) model.

    # Arguments
    - `model::SARIMAModel`: The SARIMA model to convert.
    - `maxLags::Int=12`: The maximum number of lags to include in the MA model.

    # Returns
    - `MAmodel::MAModel`: The coefficients of the lagged errors in the MA model.

    # References
    - Brockwell, P. J., & Davis, R. A. Time Series: Theory and Methods (page 92). Springer(2009)
"""
function toMA(model::SARIMAModel, maxLags::Int=12)
    arCoefficients, maCoefficients = completeCoefficientsVector(model)
    p = isnothing(arCoefficients) ? 0 : length(arCoefficients)
    q = isnothing(maCoefficients) ? 0 : length(maCoefficients)
    ψ = zeros(maxLags)

    for i in 1:maxLags
        tmp = (i <= q) ? maCoefficients[i] : 0.0
        for j in 1:min(i, p)
            tmp += arCoefficients[j] * ((i-j > 0) ? ψ[i-j] : 1.0)
        end
        ψ[i] = tmp
    end
    return ψ 
end


"""
    forecastErrors(model::SARIMAModel, maxLags::Int=12)

    The function computes the forecast errors for the SARIMA model using the estimated σ² and the MA coefficients.
    
    # Arguments
    - `model::SARIMAModel`: The SARIMA model.
    - `maxLags::Int=12`: The maximum number of lags to include in the forecast errors.

    # Returns
    - `computedForecastErrors::Vector{Fl}`: The computed forecast errors.

    # References
    - Brockwell, P. J., & Davis, R. A. Time Series: Theory and Methods (page 92). Springer(2009) 
"""
function forecastErrors(model::SARIMAModel, maxLags::Int=12)
    ψ = toMA(model, maxLags)
    computedForecastErrors = zeros(maxLags)
    computedForecastErrors[1] = model.σ²
    for lag=2:maxLags
        computedForecastErrors[lag] = model.σ² * (1 + sum(ψ[i]^2 for i=1:lag-1))
    end
    return computedForecastErrors
end

"""
    predict!(
        model::SARIMAModel;
        stepsAhead::Int = 1
        seed::Int = 1234,
        isSimulation::Bool = false,
        displayConfidenceIntervals::Bool = false,
        confidenceLevel::Fl = 0.95
    ) where Fl<:AbstractFloat

Predicts the SARIMA model for the next `stepsAhead` periods.
The resulting forecast is stored within the model in the `forecast` field.

# Arguments
- `model::SARIMAModel`: The SARIMA model to make predictions.
- `stepsAhead::Int`: The number of periods ahead to forecast (default: 1).
- `seed::Int`: Seed for random number generation when simulating forecasts (default: 1234).
- `isSimulation::Bool`: Whether to perform a simulation-based forecast (default: false).
- `displayConfidenceIntervals::Bool`: Whether to display confidence intervals (default: false).
- `confidenceLevel::Fl`: The confidence level for the confidence intervals (default: 0.95).

# Example
```julia
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)

julia> fit!(model)

julia> predict!(model; stepsAhead=12)
"""
function predict!(
            model::SARIMAModel;
            stepsAhead::Int = 1,
            seed::Int = 1234,
            isSimulation::Bool = false,
            displayConfidenceIntervals::Bool = false,
            confidenceLevel::Fl = 0.95
        )   where Fl<:AbstractFloat
    ModelFl = typeofModelElements(model)
    Random.seed!(seed)
    forecastValues::Vector{ModelFl} = predict(model, stepsAhead, isSimulation)
    forecastTimestamps::Vector{TimeType} = buildDatetimes(timestamp(model.y)[end], getproperty(Dates, model.metadata["granularity"])(model.metadata["frequency"]), model.metadata["weekDaysOnly"], stepsAhead)
    if displayConfidenceIntervals
        α::ModelFl = 1 - confidenceLevel
        computedForecastErrors::Vector{ModelFl} = forecastErrors(model, stepsAhead)
        zValue::ModelFl = quantile(Normal(0,1), 1 - α/2)
        lowerConfidenceInterval::Vector{ModelFl} = [forecastValues[i] - zValue*sqrt(computedForecastErrors[i]) for i=1:stepsAhead]
        upperConfidenceInterval::Vector{ModelFl} = [forecastValues[i] + zValue*sqrt(computedForecastErrors[i]) for i=1:stepsAhead]
        data = (datetime = forecastTimestamps, forecast = forecastValues, lower = lowerConfidenceInterval, upper = upperConfidenceInterval)
        model.forecast = TimeArray(data; timestamp=:datetime)
    else
        model.forecast = TimeArray(forecastTimestamps,forecastValues,["forecast"]) 
    end  
end


"""
    predict(
        model::SARIMAModel, 
        stepsAhead::Int = 1,
        isSimulation::Bool = true
    )

Predicts the SARIMA model for the next `stepsAhead` periods assuming the model's estimated σ² in case of a simulation.
Returns the forecasted values.

# Arguments
- `model::SARIMAModel`: The SARIMA model to make predictions.
- `stepsAhead::Int`: The number of periods ahead to forecast (default: 1).
- `isSimulation::Bool`: Whether to perform a simulation-based forecast (default: true).

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)

julia> fit!(model)

julia> forecastedValues = predict(model, stepsAhead=12)
````
"""
function predict(
            model::SARIMAModel, 
            stepsAhead::Int=1, 
            isSimulation::Bool=true
        )
    !isFitted(model) && throw(ModelNotFitted())
    ModelFl = typeofModelElements(model)

    diffY = differentiate(model.y,model.d,model.D,model.seasonality)
    valuesExog = []
    if !isnothing(model.exog)
        diffExog, _ = automaticDifferentiation(model.exog)
        # Adjust start points
        start_date = min(timestamp(diffY)[1],timestamp(diffExog)[1])
        diffY = from(diffY, start_date)
        diffExog = from(diffExog, start_date)

        valuesExog = values(diffExog)
    end

    T = size(diffY,1)
    exogT = isnothing(model.exog) ? 0 : size(diffExog,1)
    if !isnothing(model.exog) && T + stepsAhead > exogT
        throw(MissingExogenousData())
    end

    yValues::Vector{ModelFl} = deepcopy(values(diffY))
    errors = deepcopy(model.ϵ)

    for _= 1:stepsAhead
        forecastedValue::ModelFl = model.c + model.trend # *(T+stepsAhead)
        errorsLength = length(errors)
        if model.p > 0
            # ∑ϕᵢyₜ -i
            forecastedValue += sum(model.ϕ[i]*yValues[end-i+1] for i=1:model.p)
        end
        if model.q > 0
            # ∑θᵢϵₜ-i
            forecastedValue += sum(model.θ[j]*errors[end-j+1] for j=1:model.q if (errorsLength-j+1 > 0))
        end
        if model.P > 0
            # ∑Φₖyₜ-(s*k)
            forecastedValue += sum(model.Φ[k]*yValues[end-(model.seasonality*k)+1] for k=1:model.P)
        end
        if model.Q > 0
            # ∑Θₖϵₜ-(s*k)
            forecastedValue += sum(model.Θ[w]*errors[end-(model.seasonality*w)+1] for w=1:model.Q if (errorsLength-(model.seasonality*w)+1 > 0))
        end
        if !isnothing(model.exog)
            forecastedValue += valuesExog[T+stepsAhead,:]'model.exogCoefficients
        end

        ϵₜ = isSimulation ? rand(Normal(0,sqrt(model.σ²))) : 0
        forecastedValue += ϵₜ

        push!(errors, ϵₜ)
        push!(yValues, forecastedValue)
    end
    initialValuesLength = model.d + model.D*model.seasonality
    initialValuesOffset = length(values(model.y)) - initialValuesLength + 1
    initialValues::Vector{ModelFl} = values(model.y)[initialValuesOffset:end]
    forecast_values = integrate(initialValues, yValues[end-stepsAhead+1:end], model.d, model.D, model.seasonality)
    return forecast_values[initialValuesLength+1:end]
end


"""
    simulate(
        model::SARIMAModel, 
        stepsAhead::Int = 1, 
        numScenarios::Int = 200,
        seed::Int = 1234
    )

Simulates the SARIMA model for the next `stepsAhead` periods assuming that the model's estimated σ².
Returns a vector of `numScenarios` scenarios of the forecasted values.

# Arguments
- `model::SARIMAModel`: The SARIMA model to simulate.
- `stepsAhead::Int`: The number of periods ahead to simulate. Default is 1.
- `numScenarios::Int`: The number of simulation scenarios. Default is 200.
- `seed::Int`: The seed of the simulation. Default is 1234.

# Returns
- `Vector{Vector{AbstractFloat}}`: A vector of scenarios, each containing the forecasted values for the next `stepsAhead` periods.

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)

julia> model = SARIMA(airPassengers, 0, 1, 1; seasonality=12, P=0, D=1, Q=1)

julia> fit!(model)

julia> scenarios = simulate(model, stepsAhead=12, numScenarios=1000)
```
"""
function simulate(
            model::SARIMAModel, 
            stepsAhead::Int=1, 
            numScenarios::Int=200, 
            seed::Int=1234
        )
    !isFitted(model) && throw(ModelNotFitted())
    ModelFl = typeofModelElements(model)
    Random.seed!(seed)

    scenarios::Vector{Vector{ModelFl}} = []
    for _=1:numScenarios
        push!(scenarios, predict(model, stepsAhead, true))
    end
    return scenarios
end

"""
    auto(
        y::TimeArray;
        exog::Union{TimeArray,Nothing}=nothing,
        seasonality::Int=1,
        d::Int = -1,
        D::Int = -1,
        maxp::Int = 5,
        maxd::Int = 2,
        maxq::Int = 5,
        maxP::Int = 2,
        maxD::Int = 1,
        maxQ::Int = 2,
        maxOrder::Int = 5,
        informationCriteria::String = "aicc",
        allowMean:Union{Bool,Nothing} = nothing,
        allowDrift::Union{Bool,Nothing} = nothing,
        integrationTest::String = "kpss",
        seasonalIntegrationTest::String = "seas",
        objectiveFunction::String = "mse",
        assertStationarity::Bool = false,
        assertInvertibility::Bool = false,
        showLogs::Bool = false
    )

Automatically fits the best SARIMA model according to the specified parameters.

# Arguments
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `seasonality::Int`: The seasonality period. Default is 1 (non-seasonal).
- `d::Int`: The degree of differencing for the non-seasonal part. Default is -1 (auto-select).
- `D::Int`: The degree of differencing for the seasonal part. Default is -1 (auto-select).
- `maxp::Int`: The maximum autoregressive order for the non-seasonal part. Default is 5.
- `maxd::Int`: The maximum integration order for the non-seasonal part. Default is 2.
- `maxq::Int`: The maximum moving average order for the non-seasonal part. Default is 5.
- `maxP::Int`: The maximum autoregressive order for the seasonal part. Default is 2.
- `maxD::Int`: The maximum integration order for the seasonal part. Default is 1.
- `maxQ::Int`: The maximum moving average order for the seasonal part. Default is 2.
- `maxOrder::Int`: The maximum order for the non-seasonal part. Default is 5.
- `informationCriteria::String`: The information criteria to be used for model selection. Options are "aic", "aicc", or "bic". Default is "aicc".
- `allowMean::Union{Bool,Nothing}`: Whether to include a mean term in the model. Default is nothing.
- `allowDrift::Union{Bool,Nothing}`: Whether to include a drift term in the model. Default is nothing.
- `integrationTest::String`: The integration test to be used for determining the non-seasonal integration order. Default is "kpss".
- `seasonalIntegrationTest::String`: The integration test to be used for determining the seasonal integration order. Default is "seas".
- `objectiveFunction::String`: The objective function to be used for model selection. Options are "mse", "ml", or "bilevel". Default is "mse".
- `assertStationarity::Bool`: Whether to assert stationarity of the fitted model. Default is false.
- `assertInvertibility::Bool`: Whether to assert invertibility of the fitted model. Default is false.
- `showLogs::Bool`: Whether to suppress output. Default is false.

# References
- Hyndman, RJ and Khandakar. "Automatic time series forecasting: The forecast package for R." Journal of Statistical Software, 26(3), 2008.
"""
function auto(
    y::TimeArray;
    exog::Union{TimeArray,Nothing}=nothing,
    seasonality::Int=1,
    d::Int = -1,
    D::Int = -1,
    maxp::Int = 5,
    maxd::Int = 2,
    maxq::Int = 5,
    maxP::Int = 2,
    maxD::Int = 1,
    maxQ::Int = 2,
    maxOrder::Int = 5,
    informationCriteria::String = "aicc",
    allowMean::Union{Bool,Nothing} = nothing,
    allowDrift::Union{Bool,Nothing} = nothing,
    integrationTest::String = "kpss",
    seasonalIntegrationTest::String = "seas",
    objectiveFunction::String = "mse",
    assertStationarity::Bool = false,
    assertInvertibility::Bool = false,
    showLogs::Bool = false
)
    # Parameter validation
    @assert seasonality >= 1 "seasonality must be greater than 1. Use 1 for non-seasonal models"
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
    @assert seasonalIntegrationTest ∈ ["seas","ch","ocsb"]
    @assert objectiveFunction ∈ ["mae","mse","ml","bilevel"] 

    ModelFl = eltype(values(y))
    informationCriteriaFunction = getInformationCriteriaFunction(informationCriteria)

    # Deal with constant series
    if isConstant(y)
        showLogs && @info("The series is constant")
        constant = isnothing(allowMean) ? true : allowMean
        return constantSeriesModelSpecification(y, exog, constant)
    end

    # Adjustments based on parameters
    if seasonality == 1
        D = 0
    end

    if D < 0
        D = selectSeasonalIntegrationOrder(deepcopy(values(y)) ,seasonality,seasonalIntegrationTest)
    end

    if d < 0 
        d = selectIntegrationOrder(deepcopy(values(y)), maxd, D, seasonality, integrationTest)
    end

    fixConstant = !isnothing(allowMean) || !isnothing(allowDrift) ||  (d+D > 1)

    allowMean = isnothing(allowMean) ? (d+D == 0) : allowMean  
    allowDrift = isnothing(allowDrift) ?  (d+D == 1) : allowDrift


    # Deal with series constant after differencing
    if d+D > 0 && isConstant(differentiate(y, d, D, seasonality))
        showLogs && @info("The series is constant after differencing")
        return constantDiffSeriesModelSpecification(y, exog, d, D, seasonality, allowMean, allowDrift)
    end

    # Set maximum orders
    maxp = min(maxp, floor(Int,length(values(y)) / 3))
    maxq = min(maxq, floor(Int,length(values(y)) / 3))
    maxP = min(maxP, floor(Int,length(values(y)) / 3 * seasonality))
    maxQ = min(maxQ, floor(Int,length(values(y)) / 3 * seasonality))

    # Include initial models
    candidateModels = Vector{SARIMAModel}()
    visitedModels = Dict{String,Dict{String,Any}}()

    if seasonality == 1
        initialNonSeasonalModels!(candidateModels, y, exog, maxp, d, maxq, allowMean, allowDrift)
    else
        initialSeasonalModels!(candidateModels, y, exog, maxp, d, maxq, maxP, D, maxQ, seasonality, allowMean, allowDrift)
    end

    # Fit models
    bestCriteria, bestModel = localSearch!(candidateModels, visitedModels, informationCriteriaFunction, objectiveFunction, assertStationarity, assertInvertibility,showLogs)
    
    ITERATION_LIMIT = 100
    iterations = 1
    while iterations <= ITERATION_LIMIT

        addNonSeasonalModels!(bestModel, candidateModels, visitedModels, maxp, maxq, maxOrder, allowMean, allowDrift, fixConstant)
        (seasonality > 1) && addSeasonalModels!(bestModel, candidateModels, visitedModels, maxP, maxQ, maxOrder ,allowMean, allowDrift, fixConstant)
        (seasonality > 1) && addNonSeasonalAndSeasonalModels!(bestModel, candidateModels, visitedModels, maxp, maxq, maxP, maxQ, maxOrder, allowMean, allowDrift, fixConstant)

        itBestCriteria, itBestModel = localSearch!(candidateModels, visitedModels, informationCriteriaFunction, objectiveFunction, assertStationarity, assertInvertibility, showLogs)
        showLogs && @info("Iteration $(iterations): Best model found is $(getId(itBestModel)) with $(itBestCriteria) criteria")
        
        (itBestCriteria > bestCriteria) && break
        bestCriteria = itBestCriteria
        bestModel = itBestModel

        iterations += 1
    end
    showLogs && @info("The best model found is $(getId(bestModel)) with $(iterations) iterations")

    return bestModel
end


"""
    getInformationCriteriaFunction(informationCriteria)

Returns the information criteria function corresponding to the given `informationCriteria`.

# Arguments
- `informationCriteria::String`: The name of the information criteria ("aic", "aicc", or "bic").

# Returns
- `Function`: The information criteria function corresponding to the input.

# Throws
- `ArgumentError`: If the provided `informationCriteria` is not one of "aic", "aicc", or "bic".
"""
function getInformationCriteriaFunction(informationCriteria::String)
    if informationCriteria == "aic"
        return aic
    elseif informationCriteria == "aicc"
        return aicc
    elseif informationCriteria == "bic"
        return bic
    end
    throw(ArgumentError("The information criteria '$informationCriteria' is not supported"))
end
    
"""
    constantSeriesModelSpecification(
        y::TimeArray, 
        exog::Union{TimeArray,Nothing}, 
        allowMean::Bool
    )

Returns a SARIMA model for a series that is constant.

# Arguments
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `allowMean::Bool`: Whether to include a mean term in the model.

# Returns
- `SARIMAModel`: The SARIMA model for the constant series.
"""
function constantSeriesModelSpecification(y::TimeArray, exog::Union{TimeArray,Nothing}, allowMean::Bool)
    model = SARIMA(y, exog, 0, 0, 0; allowMean=allowMean)
    fit!(model)
    return model
end

"""
    constantDiffSeriesModelSpecification(
        y::TimeArray, 
        exog::Union{TimeArray,Nothing}, 
        d::Int, 
        D::Int, 
        seasonality::Int, 
        allowMean::Bool, 
        allowDrift::Bool
    )

Returns a SARIMA model for a series that is constant after differencing.

# Arguments
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `d::Int`: The degree of differencing.
- `D::Int`: The degree of seasonal differencing.
- `seasonality::Int`: The seasonality period.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.

# Returns
- `SARIMAModel`: The SARIMA model for the series that is constant after differencing.

"""
function constantDiffSeriesModelSpecification(y::TimeArray, exog::Union{TimeArray,Nothing}, d::Int, D::Int, seasonality::Int, allowMean::Bool, allowDrift::Bool)
    if isnothing(exog)
        if (D > 0 && d == 0)
            # TODO: Check if it is necessary to specify the intercept value 
            # constant should be mean(dy) / seasonality
            model = SARIMA(y, 0, d, 0; P=0, D=D, Q=0, seasonality=seasonality, allowMean=false, allowDrift=true)
        elseif (D > 0 && d > 0)
            model = SARIMA(y, 0, d, 0; P=0, D=D, Q=0, seasonality=seasonality, allowMean=false, allowDrift=false)
        elseif (d == 2)
            model = SARIMA(y, 0, d, 0; allowMean=false, allowDrift=false)
        elseif (d < 2)
            # TODO: Check if it is necessary to specify the intercept value 
            # constant should be mean(dy)
            model = SARIMA(y, 0, d, 0; allowMean=true, allowDrift=false)
        else
            throw(ArgumentError("Data follow a simple polynomial and are not suitable for ARIMA modelling."))
        end
    else
        if (D > 0)
            model = SARIMA(y, model.exog, 0, d, 0; P=0, D=D, Q=0, seasonality=seasonality, allowMean=false, allowDrift=false)
        else
            model = SARIMA(y, model.exog, 0, d, 0; allowMean=false, allowDrift=false)
        end
    end

    fit!(model)

    return model
end

"""
    initialNonSeasonalModels!(
        models::Vector{SARIMAModel}, 
        y::TimeArray,
        exog::Union{TimeArray,Nothing}, 
        maxp::Int, 
        d::Int, 
        maxq::Int, 
        allowMean::Bool,
        allowDrift::Bool
    )

Populates the `models` vector with initial non-seasonal SARIMA models based on the specified parameters.
The models added are:
- SARIMA(0, d, 0)
- SARIMA(1, d, 0)
- SARIMA(0, d, 1)
- SARIMA(2, d, 2)

# Arguments
- `models::Vector{SARIMAModel}`: A vector to which the initial SARIMA models will be appended.
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `maxp::Int`: The maximum autoregressive order.
- `d::Int`: The degree of differencing.
- `maxq::Int`: The maximum moving average order.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
"""
function initialNonSeasonalModels!(
    models::Vector{SARIMAModel}, 
    y::TimeArray,
    exog::Union{TimeArray,Nothing}, 
    maxp::Int, 
    d::Int, 
    maxq::Int, 
    allowMean::Bool,
    allowDrift::Bool
)
    push!(models, SARIMA(y, exog, 0, d, 0; allowMean=allowMean, allowDrift=allowDrift))
    (maxp >= 1) && push!(models, SARIMA(y, exog, 1, d, 0; allowMean=allowMean, allowDrift=allowDrift))
    (maxq >= 1) && push!(models, SARIMA(y, exog, 0, d, 1; allowMean=allowMean, allowDrift=allowDrift))
    (maxp >= 2 && maxq >= 2) && push!(models, SARIMA(y, exog, 2, d, 2; allowMean=allowMean, allowDrift=allowDrift))
end

"""
    initialSeasonalModels!(
        models::Vector{SARIMAModel}, 
        y::TimeArray,
        exog::Union{TimeArray,Nothing}, 
        maxp::Int, 
        d::Int, 
        maxq::Int, 
        maxP::Int, 
        D::Int, 
        maxQ::Int, 
        seasonality::Int, 
        allowMean::Bool,
        allowDrift::Bool
    )

Populates the `models` vector with initial seasonal SARIMA models based on the specified parameters.
The models added are:
- SARIMA(0, d, 0)(0, D, 0)
- SARIMA(1, d, 0)(1, D, 0)
- SARIMA(0, d, 1)(0, D, 1)
- SARIMA(2, d, 2)(1, D, 1)

# Arguments
- `models::Vector{SARIMAModel}`: A vector to which the initial SARIMA models will be appended.
- `y::TimeArray`: The time series data.
- `exog::Union{TimeArray,Nothing}`: Optional exogenous variables. If `Nothing`, no exogenous variables are used.
- `maxp::Int`: The maximum autoregressive order for non-seasonal part.
- `d::Int`: The degree of differencing for non-seasonal part.
- `maxq::Int`: The maximum moving average order for non-seasonal part.
- `maxP::Int`: The maximum autoregressive order for seasonal part.
- `D::Int`: The degree of differencing for seasonal part.
- `maxQ::Int`: The maximum moving average order for seasonal part.
- `seasonality::Int`: The seasonality period.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
"""
function initialSeasonalModels!(
    models::Vector{SARIMAModel}, 
    y::TimeArray,
    exog::Union{TimeArray,Nothing}, 
    maxp::Int, 
    d::Int, 
    maxq::Int, 
    maxP::Int, 
    D::Int, 
    maxQ::Int, 
    seasonality::Int, 
    allowMean::Bool,
    allowDrift::Bool
)
    push!(models, SARIMA(y, exog, 0, d, 0; seasonality=seasonality, P=0, D=D, Q=0, allowMean=allowMean, allowDrift=allowDrift))

    # Add non-seasonal models
    (maxp >= 1) && push!(models, SARIMA(y, exog, 1, d, 0; seasonality=seasonality, P=0, D=D, Q=0, allowMean=allowMean, allowDrift=allowDrift))
    (maxq >= 1) && push!(models, SARIMA(y, exog, 0, d, 1; seasonality=seasonality, P=0, D=D, Q=0, allowMean=allowMean, allowDrift=allowDrift))
    (maxp >= 2 && maxq >= 2) && push!(models, SARIMA(y, exog, 2, d, 2; seasonality=seasonality, P=0, D=D, Q=0, allowMean=allowMean, allowDrift=allowDrift))

    # Add seasonal models
    (maxp >= 1 && maxP >= 1) && push!(models, SARIMA(y, exog, 1, d, 0; seasonality=seasonality, P=1, D=D, Q=0, allowMean=allowMean, allowDrift=allowDrift))
    (maxq >= 1 && maxQ >= 1) && push!(models, SARIMA(y, exog, 0, d, 1; seasonality=seasonality, P=0, D=D, Q=1, allowMean=allowMean, allowDrift=allowDrift))
    (maxp >= 2 && maxq >= 2 && maxP >= 1 && maxQ >= 1) && push!(models, SARIMA(y, exog, 2, d, 2; seasonality=seasonality, P=1, D=D, Q=1, allowMean=allowMean, allowDrift=allowDrift))
end

"""
    getId(model::SARIMAModel)

Returns a string representation of the SARIMA model.

# Arguments
- `model::SARIMAModel`: The SARIMA model.

# Returns
- `String`: A string representation of the SARIMA model.

# Example
```jldoctest

julia> model = SARIMA(1, 0, 1; P=1, D=0, Q=1, seasonality=12, allowMean=true, allowDrift=false)

julia> getId(model)  # Returns "SARIMA(1,0,1)(1,0,1 s=12, c=true, drift=false)"
```
"""
function getId(
    model::SARIMAModel
)
    return "SARIMA($(model.p),$(model.d),$(model.q))($(model.P),$(model.D),$(model.Q) s=$(model.seasonality), c=$(model.allowMean), drift=$(model.allowDrift))"
end

"""
    isVisited(model::SARIMAModel, visitedModels::Dict{String,Dict{String,Any}})

Checks if a SARIMA model has been visited during the search process.

# Arguments
- `model::SARIMAModel`: The SARIMA model to check.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing visited SARIMA models.

# Returns
- `Bool`: `true` if the model has been visited, `false` otherwise.

# Example
```jldoctest
julia> model = SARIMA(1, 0, 1; P=1, D=0, Q=1, seasonality=12, allowMean=true, allowDrift=false)

julia> visitedModels = Dict{String,Dict{String,Any}}("SARIMA(1,0,1)(1,0,1 s=12, c=true, drift=false)" => Dict("criteria" => 123))

julia> isVisited(model, visitedModels)  # Returns true
```
"""
function isVisited(model::SARIMAModel, visitedModels::Dict{String,Dict{String,Any}})
    id = getId(model)
    return haskey(visitedModels, id)
end

"""
    localSearch!(
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        informationCriteriaFunction::Function,
        objectiveFunction::String = "mse",
        assertStationarity::Bool = false,
        assertInvertibility::Bool = false,
        showLogs::Bool = false
    )

Performs a local search to find the best SARIMA model among the candidate models.

# Arguments
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to search from.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `informationCriteriaFunction::Function`: A function to calculate the information criteria for a SARIMA model.
- `objectiveFunction::String`: The objective function to be used for fitting models. Default is "mse".
- `assertStationarity::Bool`: Whether to assert stationarity of the fitted models. Default is false.
- `assertInvertibility::Bool`: Whether to assert invertibility of the fitted models. Default is false.
- `showLogs::Bool`: Whether to suppress output. Default is false.

# Returns
- `Tuple{AbstractFloat, Union{SARIMAModel, Nothing}}`: A tuple containing the best criteria value and the corresponding best model found.

# Example
```jldoctest
julia> candidateModels = [SARIMA(1, 0, 1), SARIMA(0, 1, 1)]

julia> visitedModels = Dict{String,Dict{String,Any}}()

julia> informationCriteriaFunction = aicc

julia> localSearch!(candidateModels, visitedModels, informationCriteriaFunction)  
```
"""
function localSearch!(
            candidateModels::Vector{SARIMAModel},
            visitedModels::Dict{String,Dict{String,Any}},
            informationCriteriaFunction::Function,
            objectiveFunction::String = "mse",
            assertStationarity::Bool = false,
            assertInvertibility::Bool = false,
            showLogs::Bool = false
        )   
    ModelFl = typeofModelElements(candidateModels[1])
    localBestCriteria::ModelFl = Inf
    localBestModel = nothing
    foreach(model ->
        if !isFitted(model) 
            fit!(model;objectiveFunction=objectiveFunction)
            criteria = informationCriteriaFunction(model)
            showLogs && @info("Fitted $(getId(model)) with $(criteria)")
            visitedModels[getId(model)] = Dict(
                "criteria" => criteria
            )

            if criteria < localBestCriteria
                arCoefficients, maCoefficients = completeCoefficientsVector(model)

                invertible = !assertInvertibility || StateSpaceModels.assert_invertibility(maCoefficients)
                showLogs && (invertible || @info("The model $(getId(model)) is not invertible"))

                stationarity = !assertStationarity || StateSpaceModels.assert_stationarity(arCoefficients)
                showLogs && (stationarity || @info("The model $(getId(model)) is not stationary"))

                showLogs && (!invertible || !stationarity) && @info("The model will not be considered")
                if invertible && stationarity
                    localBestCriteria = criteria
                    localBestModel = model
                end
            end
        end
    , candidateModels)
    return localBestCriteria, localBestModel
end

"""
    addNonSeasonalModels!(
        bestModel::SARIMAModel, 
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},  
        maxp::Int, 
        maxq::Int, 
        maxOrder::Int,
        allowMean::Bool,
        allowDrift::Bool,
        fixConstant::Bool
    )

Adds non-seasonal SARIMA models to the candidate models vector based on the best SARIMA model found.

# Arguments
- `bestModel::SARIMAModel`: The best SARIMA model found so far.
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to add new models to.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `maxp::Int`: The maximum autoregressive order for non-seasonal part.
- `maxq::Int`: The maximum moving average order for non-seasonal part.
- `maxOrder::Int`: The maximum order for the non-seasonal part.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
- `fixConstant::Bool`: Whether to fix the constant term.

"""
function addNonSeasonalModels!(
    bestModel::SARIMAModel, 
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},  
    maxp::Int, 
    maxq::Int,
    maxOrder::Int, 
    allowMean::Bool,
    allowDrift::Bool,
    fixConstant::Bool
)
    for p in -1:1, q in -1:1
        newp = bestModel.p + p
        newq = bestModel.q + q
        if newp < 0 || newq < 0 || newp > maxp || newq > maxq || newp + newq == 0
            continue
        end

        if newp + newq + bestModel.P + bestModel.Q > maxOrder
            continue
        end

        newModel = SARIMA(
                    deepcopy(bestModel.y),
                    deepcopy(bestModel.exog),
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
            fixConstant || addChangedConstantModel!(newModel, candidateModels, visitedModels, newModel.d + newModel.D == 1)
        end
    end
end

"""
    addSeasonalModels!(
        bestModel::SARIMAModel, 
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}}, 
        maxP::Int, 
        maxQ::Int, 
        maxOrder::Int,
        allowMean::Bool,
        allowDrift::Bool,
        fixConstant::Bool
    )

Adds seasonal SARIMA models to the candidate models vector based on the best SARIMA model found.

# Arguments
- `bestModel::SARIMAModel`: The best SARIMA model found so far.
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to add new models to.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `maxP::Int`: The maximum autoregressive order for the seasonal part.
- `maxQ::Int`: The maximum moving average order for the seasonal part.
- `maxOrder::Int`: The maximum order of the model.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
- `fixConstant::Bool`: Whether to fix the constant term.

"""
function addSeasonalModels!(
    bestModel::SARIMAModel, 
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}}, 
    maxP::Int, 
    maxQ::Int,
    maxOrder::Int, 
    allowMean::Bool,
    allowDrift::Bool,
    fixConstant::Bool
)
    for P in -1:1, Q in -1:1
        newP = bestModel.P + P
        newQ = bestModel.Q + Q
        modelOrder = bestModel.p + bestModel.q + newP + newQ
        (modelOrder > maxOrder) && continue

        if newP < 0 || newQ < 0 || newP > maxP || newQ > maxQ || newP + newQ == 0 || newP + newQ > 2
            continue
        end

        newModel = SARIMA(
                    deepcopy(bestModel.y),
                    deepcopy(bestModel.exog),
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
            fixConstant || addChangedConstantModel!(newModel, candidateModels, visitedModels, newModel.d + newModel.D == 1)
        end
    end
end

"""
    addNonSeasonalAndSeasonalModels!(
        bestModel::SARIMAModel, 
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        maxp::Int,
        maxq::Int, 
        maxP::Int, 
        maxQ::Int,
        maxOrder::Int, 
        allowMean::Bool,
        allowDrift::Bool,
        fixConstant::Bool
    )

Adds non-seasonal and seasonal SARIMA models variation to the candidate models vector based on the best SARIMA model found.

# Arguments
- `bestModel::SARIMAModel`: The best SARIMA model found so far.
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to add new models to.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `maxp::Int`: The maximum autoregressive order for the non-seasonal part.
- `maxq::Int`: The maximum moving average order for the non-seasonal part.
- `maxP::Int`: The maximum autoregressive order for the seasonal part.
- `maxQ::Int`: The maximum moving average order for the seasonal part.
- `maxOrder::Int`: The maximum order of the model.
- `allowMean::Bool`: Whether to include a mean term in the model.
- `allowDrift::Bool`: Whether to include a drift term in the model.
- `fixConstant::Bool`: Whether to fix the constant term.
"""
function addNonSeasonalAndSeasonalModels!(
    bestModel::SARIMAModel, 
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},
    maxp::Int,
    maxq::Int, 
    maxP::Int, 
    maxQ::Int,
    maxOrder::Int, 
    allowMean::Bool,
    allowDrift::Bool,
    fixConstant::Bool
)
    for p in [-1,1], q in [-1,1], P in [-1,1], Q in [-1,1]
        newp = bestModel.p + p
        newq = bestModel.q + q
        newP = bestModel.P + P
        newQ = bestModel.Q + Q
        if newp < 0 || newq < 0 || newp > maxp || newq > maxq
            continue
        end

        if newP < 0 || newQ < 0 || newP > maxP || newQ > maxQ 
            continue
        end

        if newP + newQ + newp + newq > maxOrder
            continue
        end

        newModel = SARIMA(
                    deepcopy(bestModel.y),
                    deepcopy(bestModel.exog),
                    newp,
                    bestModel.d,
                    newq;
                    seasonality=bestModel.seasonality,
                    P=newP,
                    D=bestModel.D,
                    Q=newQ,
                    allowMean=allowMean,
                    allowDrift=allowDrift
                )
        if !isVisited(newModel,visitedModels)
            push!(candidateModels, newModel)
            fixConstant || addChangedConstantModel!(newModel, candidateModels, visitedModels, newModel.d + newModel.D == 1) 
        end
    end
end

"""
    addChangedConstantModel!(
        bestModel::SARIMAModel,
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        drift::Bool = false
    )

    addChangedConstantModel!(
        bestModel::SARIMAModel,
        candidateModels::Vector{SARIMAModel},
        visitedModels::Dict{String,Dict{String,Any}},
        drift::Bool = false
    )

Adds a SARIMA model with a changed constant term to the candidate models vector based on the best SARIMA model found.

# Arguments
- `bestModel::SARIMAModel`: The best SARIMA model found so far.
- `candidateModels::Vector{SARIMAModel}`: A vector of candidate SARIMA models to add new models to.
- `visitedModels::Dict{String,Dict{String,Any}}`: A dictionary containing information about visited models.
- `drift::Bool`: Whether to change the drift term. Default is false.

"""
function addChangedConstantModel!(
    bestModel::SARIMAModel,
    candidateModels::Vector{SARIMAModel},
    visitedModels::Dict{String,Dict{String,Any}},
    drift::Bool = false
)   
    allowDrift = drift && !bestModel.allowDrift
    allowMean = !drift && !bestModel.allowMean
    newModel = SARIMA(
                deepcopy(bestModel.y),
                deepcopy(bestModel.exog),
                bestModel.p,
                bestModel.d,
                bestModel.q;
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
