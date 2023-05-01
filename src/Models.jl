module Models

using SCIP,Ipopt, JuMP, MathOptInterface, LinearAlgebra, Statistics, OffsetArrays, Distributions, GLMNet, TimeSeries


export SARIMAModel, OPTSARIMAModel, arima, opt_ari, print

"""

Δyₜ = α + δt + γ y_(t-1) + ∑ ϕ_i Δy_(t-i) + ϵₜ + ∑ θ_j ϵ_(t-j)

"""
mutable struct OPTSARIMAModel
    y::TimeArray
    α::Union{Float64,Nothing}
    δ::Union{Float64,Nothing}
    γ::Union{Float64,Nothing}
    ϕ::Union{Vector{Float64},Nothing}
    θ::Union{Vector{Float64},Nothing}
    I::Union{Vector{Float64},Nothing}
    K::Union{Int64,Nothing}
    maxK::Union{Int64,Nothing}
    maxp::Union{Int64,Nothing}
    maxq::Union{Int64,Nothing}
    ϵ::Union{Vector{Float64},Nothing}
    fitInSample::Union{TimeArray,Nothing}
    forecast::Union{TimeArray,Nothing}
    aicc::Union{Float64,Nothing}
    silent::Bool
    function OPTSARIMAModel(y::TimeArray;
                            α::Union{Float64,Nothing}=nothing,
                            δ::Union{Float64,Nothing}=nothing,
                            γ::Union{Float64,Nothing}=nothing,
                            ϕ::Union{Vector{Float64},Nothing}=nothing,
                            θ::Union{Vector{Float64},Nothing}=nothing,
                            I::Union{Vector{Float64},Nothing}=nothing,
                            K::Union{Int64,Nothing}=nothing,
                            maxK::Union{Int64,Nothing}=10,
                            maxp::Union{Int64,Nothing}=13,
                            maxq::Union{Int64,Nothing}=5,
                            ϵ::Union{Vector{Float64},Nothing}=nothing,
                            fitInSample::Union{TimeArray,Nothing}=nothing,
                            forecast::Union{TimeArray,Nothing}=nothing,
                            aicc::Union{Float64,Nothing}=nothing,
                            silent::Bool=true)
        @assert maxK >= 0
        @assert maxp >= 0
        @assert maxq >= 0
        return new(y,α,δ,γ,ϕ,θ,I,K,maxK,maxp,maxq,ϵ,fitInSample,forecast,aicc,silent)
    end
end

mutable struct SARIMAModel
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

function print(model::OPTSARIMAModel)
    println("=================MODEL===============")
    println("Best K            : ", model.K)
    chosen_index = model.I .> 0
    coeficients = vcat(["δ","γ"],["ϕ_$i" for i =1:model.maxp-1])[chosen_index]
    println("Chosen Coeficients: $coeficients")
    println("Estimated α       : ",model.α)
    println("Estimated δ       : ",model.δ)
    println("Estimated γ       : ",model.γ)
    println("Estimated ϕ       : ", model.ϕ)
end

function print(model::Main.Models.OPTSARIMAModel)
    println("=================MODEL===============")
    println("Best K            : ", model.K)
    chosen_index = model.I .> 0
    coeficients = vcat(["δ","γ"],["ϕ_$i" for i =1:model.maxp-1])[chosen_index]
    println("Chosen Coeficients: $coeficients")
    println("Estimated α       : ",model.α)
    println("Estimated δ       : ",model.δ)
    println("Estimated γ       : ",model.γ)
    println("Estimated ϕ       : ", model.ϕ)
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

function fill_fit_values!(model::OPTSARIMAModel,
                        auxModel::OPTSARIMAModel)
    model.α = auxModel.α
    model.δ = auxModel.δ
    model.γ = auxModel.γ
    model.ϕ = auxModel.ϕ
    model.θ = auxModel.θ
    model.ϵ = auxModel.ϵ
    model.I = auxModel.I
    model.K = auxModel.K
    model.aicc = auxModel.aicc
    model.fitInSample = auxModel.fitInSample
end

function OPTSARIMAModel(α,δ,γ,ϕ,θ,I,K,maxK,maxp,maxq,ϵ,fitInSample,aicc,silent)
    if (maxK < 0 || maxp < 0 || maxq < 0)
        error("Negative values not allowed")
    end

    return OPTSARIMAModel(α,δ,γ,ϕ,θ,I,K,maxK,maxp,maxq,ϵ,fitInSample,aicc,silent)
end

function OPTSARIMAModel!(model,α,δ,γ,ϕ,θ,I,K,ϵ,fitInSample,aicc)
    model.α = α
    model.δ = δ
    model.γ = γ
    model.ϕ = ϕ
    model.θ = θ
    model.I = I
    model.K = K
    model.ϵ = ϵ
    model.fitInSample = fitInSample
    model.aicc = aicc
end

function get_opt_λ(y::Vector{Float64}, X::Matrix{Float64})
    T,p = size(X)
    ratio = p > T ? 0.01*(p/T) : 0.0001
    cv_results = GLMNet.glmnetcv(X, y, alpha=0, nfolds=5, lambda_min_ratio=ratio, standardize=true)
    return cv_results.lambda[argmin(cv_results.meanloss)]
end

function opt_ari(model::OPTSARIMAModel;silent=false, optimizer::DataType=SCIP.Optimizer, reg::Bool=false)
    y_values = values(model.y)
    # Diff y
    diff_y = diff(model.y,differences=1)
    T = length(diff_y)
    diff_y_values = values(diff_y)
    K = 1
    mod = Model(optimizer)
    if solver_name(mod) == "Gurobi"
        set_optimizer_attribute(mod, "NonConvex", 2)
    end
    if silent
        set_silent(mod)
    end
    @variables(mod, begin
        fobj
        α
        δ
        γ
        K_var
        ϕ[1:model.maxp-1] # AR part  
        I[1:1+model.maxp], Bin # 2 + maxp - 1
        ϵ[t = 1:T]
    end)

    all_coefs = vcat([δ],[γ],ϕ)
    lb = model.maxp + 1
    @expression(mod, ŷ[t=lb:T], α + δ*t + γ*y_values[t-1] + sum(ϕ[i]*diff_y_values[t-i] for i=1:model.maxp-1) + ϵ[t])
    @constraint(mod,[t=lb:T], diff_y_values[t] == ŷ[t])
    @constraint(mod, [i = 1:1+model.maxp], all_coefs[i]*(1 - I[i]) == 0) # WARNING: Non linear
    @constraint(mod, sum(I) <= K_var)
    @constraint(mod, fobj == sum(ϵ[i]^2 for i in 1:T))

    X = vcat(zeros(1),y_values[2:end])
    for i=3:model.maxp
        X = hcat(X,vcat(zeros(i-1),y_values[i:end]))
    end
    λ = get_opt_λ(y_values,X)
    reg_multiplier = reg ? 1 : 0

    @objective(mod, Min, fobj + reg_multiplier *  1/(2*λ) * sum(all_coefs.^2)) # Calibrar Ver com o André
    fix.(K_var, K)
    optimize!(mod)

    aiccs = Vector{Float64}()
    aic = 2*K + T*log(var(value.(ϵ)))
    aicc_ = (aic + ((2*K^2 +2*K)/(T - K - 1)))
    fitInSample::TimeArray = TimeArray(timestamp(diff_y)[lb:end], OffsetArrays.no_offset_view(value.(ŷ)))
    fitted_model = OPTSARIMAModel(model.y;α=value(α),δ=value(δ),γ=value(γ),ϕ=value.(ϕ),I=value.(I),K=floor(Int64,sum(value.(I))),ϵ=value.(ϵ),fitInSample=fitInSample,aicc=aicc_)
    fitted_models = [fitted_model]
    push!(aiccs, aicc_)
    K+=1
    while K <= model.maxK
        fix.(K_var, K)
        optimize!(mod)
        @info("Solved for K = $K")

        aic = 2*K + T*log(var(value.(ϵ)))
        aicc_ = (aic + ((2*K^2 +2*K)/(T - K - 1)))
        push!(aiccs, aicc_)

        fitInSample = TimeArray(timestamp(diff_y)[lb:end], OffsetArrays.no_offset_view(value.(ŷ)))
        fitted_model = OPTSARIMAModel(model.y;α=value(α),δ=value(δ),γ=value(γ),ϕ=value.(ϕ),I=value.(I),K=floor(Int64,sum(value.(I))),ϵ=value.(ϵ),fitInSample=fitInSample,aicc=aicc_)
        push!(fitted_models, fitted_model)
        
        if aiccs[end] >= aiccs[end - 1]
            @info("aicc[end]=",aiccs[end])
            @info("aicc[end-1]=",aiccs[end-1])
            @info("Best K found: ", K-1)
            break
        end
       
        K += 1
    end

    # best model
    best_model = fitted_models[K-1]
    adf = best_model.γ/std(best_model.ϵ)
    if adf <= -3.60 # t distribuition for size T=50
        best_model.fitInSample = best_model.fitInSample .+ lag(model.y,1)
    end
    fill_fit_values!(model,best_model)
    #return fitted_models, findfirst(x->x==minimum(aiccs), aiccs)
end


function ari(y;maxp=5, K=1, silent=false, optimizer::DataType=SCIP.Optimizer)
    # Diff y
    Δy = vcat([NaN], diff(y)) 
    T = length(y)
    model = Model(optimizer)

    if solver_name(model) == "Gurobi"
        set_optimizer_attribute(model, "NonConvex", 2)
    end

    if silent
        set_silent(model)
    end
    @variables(model, begin
        epi
        α
        δ
        γ
        K_var
        ϕ[1:maxp-1] # AR part  
        I[1:1+maxp], Bin # 2 + maxp - 1
        ϵ[t = 1:T]
    end)

    all_coefs = vcat([δ],[γ],ϕ)
    @constraint(model,[t = maxp+1:T], Δy[t] == α + δ*t + γ*y[t-1] + sum(ϕ[i]*Δy[t-i] for i=1:maxp-1) + ϵ[t])
    @constraint(model, [i = 1:1+maxp], all_coefs[i]*(1 - I[i]) == 0) # WARNING: Non linear
    @constraint(model, sum(I) == K_var)
    @constraint(model, epi == sum(ϵ[i]^2 for i in 1:T))
    @objective(model, Min, epi)
    fix.(K_var, K)

    optimize!(model)
    return value.(ϵ), value.(ϕ)
end

function arima_start_values(opt_ari_model::OPTSARIMAModel, model, maxq)
    set_start_value(model[:α], opt_ari_model.α)
    set_start_value(model[:δ], opt_ari_model.δ)
    set_start_value(model[:γ], opt_ari_model.γ)

    for i in eachindex(opt_ari_model.I)
        set_start_value(model[:I][i], opt_ari_model.I[i])
    end

    for i in eachindex(opt_ari_model.ϕ)
        set_start_value(model[:ϕ][i], opt_ari_model.ϕ[i])
    end

    for i in eachindex(opt_ari_model.ϵ)
        set_start_value(model[:ϵ][i], opt_ari_model.ϵ[i])
    end

    for i in 1:maxq-1 
        set_start_value(model[:θ][i], 0) # since the opt ari does not have the MA component
    end
end

function arima(y::Vector{Float64};maxp=6,maxq=6,maxK=8,silent=false,optimizer::DataType=SCIP.Optimizer)
    # Diff y
    Δy = vcat([NaN],diff(y)) 
    T = length(Δy)

    model = Model(optimizer)

    if solver_name(model) == "Gurobi"
        set_optimizer_attribute(model, "NonConvex", 2)
    end

    if silent
        set_silent(model)
    end

    @variables(model, begin
        fo
        -2 <= α <= 2
        -5 <= δ <= 5# explicativa deve ser δ tem que achar outro parâmetro
        -5 <= γ <= 5
        K_var 
        -1 <= ϕ[1:maxp-1] <= 1 # AR part  
        -1 <= θ[1:maxq-1] <= 1# MA part
        I[1:maxp+maxq], Bin # 2 + maxp - 1 + maxq - 1 
        -100 <= ϵ[t = 1:T] <= 100
    end)

    fitted_aris, opt_k = opt_ari(y;maxp=maxp,silent=silent)
    K = opt_k 
    if opt_k > 1
        K = opt_k-1
    end

    arima_start_values(fitted_aris[K], model, maxq)
    
    all_coefs = vcat([δ],[γ],ϕ,θ)
    @constraint(model,[t = max(maxp+1, maxq+1):T], Δy[t] == α + δ*t + γ*y[t-1] + sum(ϕ[i]*Δy[t-i] for i=1:maxp-1) + ϵ[t] + sum(θ[j]*ϵ[t-j] for j=1:max(maxq-1,1)))
    @constraint(model, [i = 1:maxp+maxq], all_coefs[i]*(1 - I[i]) == 0) # WARNING: Non linear
    @constraint(model, sum(I) <= K_var)
    @constraint(model, fo == sum(ϵ[i]^2 for i in 1:T))
    @objective(model, Min, fo)
    @expression(model, fit[t=max(maxp+1, maxq+1):T], α + δ*t + γ*y[t-1] + sum(ϕ[i]*Δy[t-i] for i=1:maxp-1) + ϵ[t] + sum(θ[j]*ϵ[t-j] for j=1:maxq-1))
    fix.(K_var, K)
    
    optimize!(model)
    @info("Solved for K= $K")
    aiccs = Vector{Float64}()
    aic = 2*K + T*log(var(value.(ϵ)))
    aicc_ = (aic + ((2*K^2 +2*K)/(T - K - 1)))
    fitted_model = OPTSARIMAModel(value(α),value(δ), value(γ),value.(ϕ),value.(θ),value.(I),K,maxK,maxp,maxq,value.(ϵ),vcat([NaN for _=1:max(maxp,maxq)],  OffsetArrays.no_offset_view(value.(fit))),aicc_,silent)
    push!(aiccs, aicc_)
    K+=1
    while K <= maxK
        OPTSARIMAModel!(fitted_model,value(α),value(δ),value(γ),value.(ϕ),value.(θ),value.(I),K-1,value.(ϵ),vcat([NaN for _=1:max(maxp,maxq)], OffsetArrays.no_offset_view(value.(fit))),aicc_)
        # Start Viable Point
        arima_start_values(fitted_aris[min(K,length(fitted_aris))], model, maxq)
        fix.(K_var, K)
        optimize!(model)
        @info("Solved for K = $K")

        aic = 2*K + T*log(var(value.(ϵ)))
        aicc_ = (aic + ((2*K^2 +2*K)/(T - K - 1)))
        push!(aiccs, aicc_)
        if aiccs[end] >= aiccs[end - 1]
            @info("Best K found: ",K-1)
            break
        end
       
        K += 1
    end
 
    return fitted_model
end

function arima(model::SARIMAModel;silent::Bool=true,optimizer::DataType=Ipopt.Optimizer)
    # seasonal difference
    diff_y = nothing
    if model.seasonality > 1
        diff_y = diff(model.y, differences=model.D)
    end
    # non seasonal diff y
    diff_y = diff(model.y, differences=model.d)
    T = length(diff_y)

    # Normalizing arrays 
    diff_y = (diff_y .- mean(values(diff_y)))./std(values(diff_y)) 
    y_values = values(diff_y)

    mod = Model(optimizer)
    if silent 
        set_silent(mod)
    end

    @variable(mod, ϕ[1:model.p])
    @variable(mod, θ[1:model.q])
    @variable(mod,Φ[1:model.P])
    @variable(mod,Θ[1:model.Q])
    @variable(mod, ϵ[1:T])
    @variable(mod, c)

    for i in 1:model.q 
        set_start_value(mod[:θ][i], 0) 
    end

    for i in 1:model.Q 
        set_start_value(mod[:Θ][i], 0) 
    end

    @objective(mod, Min, sum(ϵ.^2))

    lb = max(model.p,model.q,model.P*model.seasonality,model.Q*model.seasonality) + 1
    if model.seasonality > 1
        @expression(mod, ŷ[t=lb:T], c + sum(ϕ[i]*y_values[t-i] for i=1:model.p) + sum(θ[j]*ϵ[t-j] for j=1:model.q) + sum(Φ[k]*y_values[t-(model.seasonality*k)] for k=1:model.P) + sum(Θ[w]*y_values[t-(model.seasonality*w)] for w=1:model.Q) + ϵ[t])
    else
        @expression(mod, ŷ[t=lb:T], c + sum(ϕ[i]*y_values[t-i] for i=1:model.p) + sum(θ[j]*ϵ[t-j] for j=1:model.q) + ϵ[t])
    end
    @constraint(mod, [t=lb:T], y_values[t] == ŷ[t])
    optimize!(mod)
    termination_status(mod)
    
    fitInSample::TimeArray = TimeArray(timestamp(diff_y)[lb:end], OffsetArrays.no_offset_view(value.(ŷ)))

    # plot(diff_y)
    # plot!(fitInSample)
    # plot(y)
    # original_fit = lag(y,1) .+ fitInSample
    # plot!(original_fit)

    # TODO - Falta resolver a diferenciação sazonal
    if model.d != 0 # We differenciated the timeseries
       # Δyₜ = yₜ - y_t-1 => yₜ = Δyₜ + y_t-1
       fitInSample = fitInSample .+ lag(model.y,1) 
    end

    fill_fit_values!(model,value(c),value.(ϕ),value.(θ),value.(ϵ),fitInSample;Φ=value.(Φ),Θ=value.(Θ))
end

function predict!(model::SARIMAModel, stepsAhead::Int64=1)
    y_values = values(model.y)
    T = length(y_values)
    errors = model.ϵ
    errors = vcat(errors,[0 for _=1:stepsAhead])
    for _=1:stepsAhead
        push!(y_values, model.c + sum(model.ϕ[i]*y_values[end-i+1] for i=1:model.p) + sum(model.θ[i]*errors[T-j+1] for j=1:model.q))
    end
end

# function bic(ϵ::Float64,σ::Float64,K::Int64,N::Int64)
#     return K*(log(N))
# end

# function logpdfNormal(ϵ::T,σ::T) where {T<:Real}
#     return logpdf.(Normal(0.0,σ),ϵ)
# end

# function arima_bic(y::Vector{Float64};maxp=6,maxq=6,maxK=8,silent=false,optimizer::DataType=SCIP.Optimizer)
#     # Diff y
#     Δy = vcat([NaN],diff(y)) 
#     T = length(Δy)
#     model = Model(optimizer)

#     if solver_name(model) == "Gurobi"
#         set_optimizer_attribute(model, "NonConvex", 2)
#     end

#     if silent
#         set_silent(model)
#     end

#     @variables(model, begin
#         fo
#         σ 
#         α
#         δ # explicativa deve ser δ tem que achar outro parâmetro
#         γ
#         K_var 
#         -1 <= ϕ[1:maxp-1] <= 1 # AR part  
#         -1 <= θ[1:maxq-1] <= 1# MA part
#         I[1:maxp+maxq], Bin # 2 + maxp - 1 + maxq - 1 
#         -100 <= ϵ[t = 1:T] <= 100
#     end)
#     register(model, :logpdfNormal, 2, logpdfNormal, autodiff=true)
#     opt_ari_model = opt_ari(y;maxp=maxp,silent=silent)
#     @info("Solved auto ari")

#     set_start_value(model[:α], opt_ari_model.α)
#     set_start_value(model[:δ], opt_ari_model.δ)
#     set_start_value(model[:γ], opt_ari_model.γ)

#     for i in eachindex(opt_ari_model.I)
#         set_start_value(model[:I][i], opt_ari_model.I[i])
#     end

#     for i in eachindex(opt_ari_model.ϕ)
#         set_start_value(model[:ϕ][i], opt_ari_model.ϕ[i])
#     end

#     for i in eachindex(opt_ari_model.ϵ)
#         set_start_value(model[:ϵ][i], opt_ari_model.ϵ[i])
#     end

#     for i in 1:maxq-1 
#         set_start_value(model[:θ][i], 0) # since the opt ari does not have the MA component
#     end

#     fix.(K_var, opt_ari_model.K)
    
#     all_coefs = vcat([δ],[γ],ϕ,θ)
#     @constraint(model,[t = max(maxp+1, maxq+1):T], Δy[t] == α + δ*t + γ*y[t-1] + sum(ϕ[i]*Δy[t-i] for i=1:maxp-1) + ϵ[t] + sum(θ[j]*ϵ[t-j] for j=1:max(maxq-1,1)))
#     @constraint(model, [i = 1:maxp+maxq], all_coefs[i]*(1 - I[i]) == 0) # WARNING: Non linear
#     @constraint(model, sum(I) == K_var)
#     @NLconstraint(model, fo == -2.0*sum(logpdfNormal(ϵ[t],σ) for t=1:T) + K_var*(log(T)))
#     @objective(model, Min, fo)
#     @expression(model, fit[t=max(maxp+1, maxq+1):T], α + δ*t + γ*y[t-1] + sum(ϕ[i]*Δy[t-i] for i=1:maxp-1) + ϵ[t] + sum(θ[j]*ϵ[t-j] for j=1:maxq-1))
    
#     optimize!(model)
#     @info("Solved")
#     fitted_model = OPTSARIMAModel(value(α),value(δ), value(γ),value.(ϕ),value.(θ),value.(I),value(K),maxK,maxp,maxq,value.(ϵ),vcat([NaN for _=1:max(maxp,maxq)],  OffsetArrays.no_offset_view(value.(fit))),aicc_,silent)
#     return fitted_model
# end

    
end # module Model