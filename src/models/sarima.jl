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
    forecast::Union{TimeArray,Nothing}
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

function fit!(model::SARIMAModel;silent::Bool=true,optimizer::DataType=Ipopt.Optimizer)
    diff_y = model.y
    # seasonal difference
    if model.D > 0
        @info("Seasonal difference")
        for _ in 1:model.D
            diff_y = diff_y .- TimeSeries.lag(diff_y, model.seasonality)
        end
        diff_y = TimeArray(timestamp(diff_y)[model.D*model.seasonality+1:end],values(diff_y)[model.D*model.seasonality+1:end])
    end

    # non seasonal diff y
    @info("Difference")
    diff_y = TimeSeries.diff(diff_y, differences=model.d)

    T = length(diff_y)

    # Normalizing arrays 
    # diff_y = (diff_y .- mean(values(diff_y)))./std(values(diff_y)) 
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
    
    @objective(mod, Min, sum(ϵ.^2))# + 0.1*(sum(θ.^2)+sum(Θ.^2)))
    
    lb = max(model.p,model.q,model.P*model.seasonality,model.Q*model.seasonality) + 1
    fix.(ϵ[1:lb],0.0)
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
       for _ in 1:model.d
        fitInSample = fitInSample .+ TimeSeries.lag(model.y,1)
       end
    end

    if model.D > 0 # We differenciated the timeseries
        # Δyₜ = yₜ - y_t-12 => yₜ = Δyₜ + y_t-12
        for _ in 1:model.D
            fitInSample = fitInSample .+ TimeSeries.lag(model.y,model.seasonality)
        end
    end

    if model.D > 0 && model.d > 0 # We differenciated the timeseries
        fitInSample = fitInSample .- TimeSeries.lag(model.y,model.d + model.seasonality)
    end

    fill_fit_values!(model,value(c),value.(ϕ),value.(θ),value.(ϵ),fitInSample;Φ=value.(Φ),Θ=value.(Θ))
end

function predict!(model::SARIMAModel, stepsAhead::Int64=1)
    diff_y = model.y
    if model.D > 0
        @info("Seasonal difference")
        for _ in 1:model.D
            diff_y = diff_y .- TimeSeries.lag(diff_y, model.seasonality)
        end
        diff_y = TimeArray(timestamp(diff_y)[model.D*model.seasonality+1:end],values(diff_y)[model.D*model.seasonality+1:end])
    end

    # non seasonal diff y
    @info("Difference")
    diff_y = TimeSeries.diff(diff_y, differences=model.d)
    T = length(diff_y)

    y_values = copy(values(diff_y))

    errors = model.ϵ
    errors = vcat(errors,[0 for _=1:stepsAhead])
    for _=1:stepsAhead
        y_for = model.c
        if model.p > 0
            y_for += sum(model.ϕ[i]*y_values[end-i+1] for i=1:model.p)
        end
        if model.q > 0
            y_for += sum(model.θ[j]*errors[T-j+1] for j=1:model.q)
        end
        if model.P > 0
            y_for += sum(model.Φ[k]*y_values[end-(model.seasonality*k)] for k=1:model.P)
        end
        if model.Q > 0
            y_for += sum(model.Θ[w]*errors[T-(model.seasonality*w)] for w=1:model.Q)
        end
        push!(y_values, y_for)
    end

    forecast_dates = [ timestamp(model.y)[end] + Dates.Month(i) for i=1:stepsAhead ]
    y_pred = TimeArray(forecast_dates,y_values[T+1:end])

    # if model.d > 0 # We differenciated the timeseries
    #     y_pred = y_pred .+ TimeSeries.lag(model.y,1)
    #     y_pred = y_pred .+ TimeSeries.lag(y_pred,1)
    #     # for i=1:stepsAhead
    #     #     y_pred_values[T+i] += y_pred_values[T+i-1]
    #     # end
    # end

    # if model.D > 0 # We differenciated the timeseries
    #     y_pred = y_pred .+ TimeSeries.lag(model.y,model.seasonality)
    #     y_pred = y_pred .+ TimeSeries.lag(y_pred,model.seasonality)
    #     # for i=1:stepsAhead
    #     #     y_pred_values[T+i] += y_pred_values[T+i-model.seasonality] 
    #     # end
    # end

    # if model.D > 0 && model.d > 0 # We differenciated the timeseries
    #     y_pred = y_pred .- TimeSeries.lag(model.y,model.d+model.seasonality)
    #     y_pred = y_pred .- TimeSeries.lag(y_pred,model.d+model.seasonality)
    #     # for i=1:stepsAhead
    #     #     y_pred_values[T+i] -= y_pred_values[T+i-model.seasonality-model.d] 
    #     # end
    # end
    model.forecast = y_pred#TimeArray(forecast_dates,values(y_pred)[end-stepsAhead-1:end])
end