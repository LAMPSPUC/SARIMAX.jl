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
        return new(y,p,d,q,seasonality,P,D,Q,c,ϕ,θ,Φ,Θ,ϵ,fitInSample,forecast,silent)
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
                        fitInSample::TimeArray;
                        Φ::Union{Vector{Float64},Nothing}=nothing,
                        Θ::Union{Vector{Float64},Nothing}=nothing)
    model.c = c
    model.ϕ = ϕ
    model.θ = θ
    model.ϵ = ϵ
    model.Φ = Φ
    model.Θ = Θ
    model.fitInSample = fitInSample
end

function copy(y::TimeArray)
    return TimeArray(copy(timestamp(y)),copy(values(y)))
end

function differentiate(series::TimeArray,d::Int=0, D::Int=0, s::Int=1)
    series = TimeArray(timestamp(series),values(series))
    if D > 0
        @info("Seasonal difference")
        diff_values = []
        original_values = values(series)
        T = length(original_values)
        for i=1:D
            # Δyₜ = yₜ - y_t-s
            for j=i*s+1:T
                push!(diff_values, original_values[j] - original_values[j-i*s])
            end
        end
        series = TimeArray(copy(timestamp(series))[(D*s)+1:end],diff_values)
    end
    # non seasonal diff y
    @info("Non seasonal difference")
    for _ in 1:d
        diff_values = []
        original_values = values(series)
        T = length(original_values)
        # Δyₜ = yₜ - y_t-1
        for j=2:T
            push!(diff_values,original_values[j] - original_values[j-1])
        end
        series = TimeArray(copy(timestamp(series))[2:end],diff_values)
    end
    return series
end

function integrate(series::TimeArray, diff_series::Vector{Fl}, d::Int=0, D::Int=0, s::Int=1) where Fl<:Real
    series = TimeArray(timestamp(series),values(series))
    stepsAhead = length(diff_series)
    y = deepcopy(values(series))
    T = length(y)
    y = vcat(y,diff_series)
    for i=T+1:T+stepsAhead
        # @info("Non seasonal integration")
        recovered_value = y[i]
        # Δyt = y[t] - y[t-1] - y[t-12] + y[t-12-1]
        for _ in 1:d
            # Δyₜ = yₜ - y_t-1 ⇒ yₜ = Δyₜ + y_t-1
            recovered_value += y[i-1]
        end
        # @info("Seasonal integration")
        for _ in 1:D
            # Δyₜ = yₜ - y_t-s ⇒ yₜ = Δyₜ + y_t-s
            recovered_value += y[i-s]
        end
        # @info("Correction for seasonal integration")
        if D > 0 && d > 0
            # Δyₜ = yₜ - y_t-s ⇒ yₜ = Δyₜ + y_t-s
            recovered_value -= y[i-s-1]
        end
        y[i] = recovered_value
    end
    
    return y[T+1:end]
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
    @constraint(mod, sum(ϵ) == 0)
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
    fill_fit_values!(model,value(c),value.(ϕ),value.(θ),value.(ϵ),fitInSample;Φ=value.(Φ),Θ=value.(Θ))
end

function predict!(model::SARIMAModel, stepsAhead::Int64=1)
    diff_y = differentiate(model.y,model.d,model.D,model.seasonality)
    y_values::Vector{Float64} = deepcopy(values(diff_y))
    # DAVI OLHA ESSA GAMBIARRA POR FAVOR
    # errors = deepcopy(model.ϵ .+ model.c)
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