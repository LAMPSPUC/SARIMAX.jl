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

function fill_fit_values!(model::SARIMAModel,
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

function fit!(model::SARIMAModel;silent::Bool=true,optimizer::DataType=Ipopt.Optimizer, normalize::Bool=false)
    diff_y = differentiate(model.y,model.d,model.D, model.seasonality)

    T = length(diff_y)

    # Normalizing arrays 
    if normalize
        @info("Normalizing time series")
        diff_y = (diff_y .- mean(values(diff_y)))./std(values(diff_y)) 
    end

    y_values = values(diff_y)

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
        @expression(mod, ŷ[t=lb:T], c + sum(ϕ[i]*y_values[t-i] for i=1:model.p) + sum(θ[j]*ϵ[t-j] for j=1:model.q) + sum(Φ[k]*y_values[t-(model.seasonality*k)] for k=1:model.P) + sum(Θ[w]*ϵ[t-(model.seasonality*w)] for w=1:model.Q))
    else
        @expression(mod, ŷ[t=lb:T], c + sum(ϕ[i]*y_values[t-i] for i=1:model.p) + sum(θ[j]*ϵ[t-j] for j=1:model.q))
    end
    @constraint(mod, [t=lb:T], y_values[t] == ŷ[t] + ϵ[t])
    optimize!(mod)
    termination_status(mod)
    
    # TODO: - The reconciliation works for just d,D <= 1
    fitInSample::TimeArray = TimeArray(timestamp(diff_y)[lb:end], OffsetArrays.no_offset_view(value.(ŷ)))
    
    if model.d > 0 # We differenciated the timeseries
        # Δyₜ = yₜ - y_t-1 => yₜ = Δyₜ + y_t-1
        fitted_values = values(fitInSample)
        y_original = values(model.y)
        for _=1:model.d
            original_index = findfirst(ts -> ts == timestamp(fitInSample)[1], timestamp(model.y))
            for j=1:length(fitInSample)
                fitted_values[j] += y_original[original_index+(j-1)-1]
            end
        end
        fitInSample = TimeArray(timestamp(fitInSample), fitted_values)
    end

    if model.D > 0 # We differenciated the timeseries
        # Δyₜ = yₜ - y_t-12 => yₜ = Δyₜ + y_t-12
        fitted_values = values(fitInSample)
        y_original = values(model.y)
        for i=1:model.D
            original_index = findfirst(ts -> ts == timestamp(fitInSample)[1], timestamp(model.y))
            for j=1:length(fitInSample)
                fitted_values[j] += y_original[original_index+(j-1)-model.seasonality*i]
            end
        end
        fitInSample = TimeArray(timestamp(fitInSample), fitted_values)
    end

    if model.D > 0 && model.d > 0 # We differenciated the timeseries
        fitted_values = values(fitInSample)
        y_original = values(model.y)
        for j=1:length(fitInSample)
            original_index = findfirst(ts -> ts == timestamp(fitInSample)[1], timestamp(model.y))
            fitted_values[j] -= y_original[original_index+(j-1)-(model.seasonality+1)]
        end
        fitInSample = TimeArray(timestamp(fitInSample), fitted_values)
    end
    residuals_variance = var(value.(ϵ)[lb:end])
    fill_fit_values!(model,value(c),value.(ϕ),value.(θ),value.(ϵ),residuals_variance,fitInSample;Φ=value.(Φ),Θ=value.(Θ))
end

function predict!(model::SARIMAModel, stepsAhead::Int64=1)
    diff_y = differentiate(model.y,model.d,model.D,model.seasonality)
    y_values::Vector{Float64} = deepcopy(values(diff_y))
    errors = deepcopy(model.ϵ)
    for _= 1:stepsAhead
        y_for = model.c
        if model.p > 0
            # ∑ϕᵢyₜ -i
            y_for += sum(model.ϕ[i]*y_values[end-i+1] for i=1:model.p)
        end
        if model.q > 0
            # ∑θᵢϵₜ-i
            y_for += sum(model.θ[j]*errors[end-j+1] for j=1:model.q)
        end
        if model.P > 0
            # ∑Φₖyₜ-(s*k)
            y_for += sum(model.Φ[k]*y_values[end-(model.seasonality*k)+1] for k=1:model.P)
        end
        if model.Q > 0
            # ∑Θₖϵₜ-(s*k)
            y_for += sum(model.Θ[w]*errors[end-(model.seasonality*w)+1] for w=1:model.Q)
        end
        push!(errors, 0)
        push!(y_values, y_for)
    end
    forecast_values = integrate(model.y, y_values[end-stepsAhead+1:end], model.d, model.D, model.seasonality)
    model.forecast = forecast_values
end

function predict(model::SARIMAModel, stepsAhead::Int64=1, σ²::Float64=0.0)
    diff_y = differentiate(model.y,model.d,model.D,model.seasonality)
    y_values::Vector{Float64} = deepcopy(values(diff_y))
    errors = deepcopy(model.ϵ)
    for _= 1:stepsAhead
        y_for = model.c
        if model.p > 0
            # ∑ϕᵢyₜ -i
            y_for += sum(model.ϕ[i]*y_values[end-i+1] for i=1:model.p)
        end
        if model.q > 0
            # ∑θᵢϵₜ-i
            y_for += sum(model.θ[j]*errors[end-j+1] for j=1:model.q)
        end
        if model.P > 0
            # ∑Φₖyₜ-(s*k)
            y_for += sum(model.Φ[k]*y_values[end-(model.seasonality*k)+1] for k=1:model.P)
        end
        if model.Q > 0
            # ∑Θₖϵₜ-(s*k)
            y_for += sum(model.Θ[w]*errors[end-(model.seasonality*w)+1] for w=1:model.Q)
        end
        ϵₜ = rand(Normal(0,sqrt(σ²)))
        y_for += ϵₜ
        push!(errors, ϵₜ)
        push!(y_values, y_for)
    end
    forecast_values = integrate(model.y, y_values[end-stepsAhead+1:end], model.d, model.D, model.seasonality)
    return forecast_values
end

function simulate(model::SARIMAModel, stepsAhead::Int64=1, numScenarios::Int64=200)
    scenarios::Vector{Vector{Float64}} = []
    for _=1:numScenarios
        push!(scenarios, predict(model,stepsAhead, model.σ²))
    end
    return scenarios
end