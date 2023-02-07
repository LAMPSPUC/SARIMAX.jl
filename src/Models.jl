module Models

using SCIP, JuMP, MathOptInterface, LinearAlgebra, Statistics, OffsetArrays

export sarimaxModel, arima

"""

Δyₜ = α + δt + γ y_(t-1) + ∑ ϕ_i Δy_(t-i) + ϵₜ + ∑ θ_j ϵ_(t-j)

"""
mutable struct SarimaxModel
    α::Float64
    δ::Float64
    γ::Float64
    ϕ::Vector{Float64}
    θ::Vector{Float64}
    I::Vector{Float64}
    K::Int64
    maxK::Int64
    maxp::Int64
    maxq::Int64
    ϵ::Vector{Float64}
    fitInSample::Vector{Float64}
    aicc::Float64
    silent::Bool
end

function sarimaxModel(α,δ,γ,ϕ,θ,I,K,maxK,maxp,maxq,ϵ,fitInSample,aicc,silent)
    if (maxK < 0 || maxp < 0 || maxq < 0)
        error("Negative values not allowed")
    end

    return SarimaxModel(α,δ,γ,ϕ,θ,I,K,maxK,maxp,maxq,ϵ,fitInSample,aicc,silent)
end

function sarimaxModel!(model,α,δ,γ,ϕ,θ,I,K,ϵ,fitInSample,aicc)
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

function ari(y;maxp=5, K=1, silent=false, optimizer::DataType=SCIP.Optimizer)
    # Diff y
    Δy = vcat([NaN], diff(y)) 
    T = length(y)
    model = Model(optimizer)
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

function arima(y::Vector{Float64};maxp=6,maxq=6,maxK=8,silent=false,optimizer::DataType=SCIP.Optimizer)
    # Diff y
    Δy = vcat([NaN],diff(y)) 
    T = length(Δy)

    K = 1
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
        δ # explicativa deve ser δ tem que achar outro parâmetro
        γ
        K_var 
        -1 <= ϕ[1:maxp-1] <= 1 # AR part  
        -1 <= θ[1:maxq-1] <= 1# MA part
        I[1:maxp+maxq], Bin # 2 + maxp - 1 + maxq - 1 
        -100 <= ϵ[t = 1:T] <= 100
    end)

    start_ϵ, start_ϕ = ari(y,maxp=maxp,silent=silent,K=K)
    for i in eachindex(start_ϵ)
        set_start_value(model[:ϵ][i], start_ϵ[i])
    end

    for i in eachindex(start_ϕ)
        set_start_value(model[:ϕ][i], start_ϕ[i])
    end
    
    all_coefs = vcat([δ],[γ],ϕ,θ)
    @constraint(model,[t = max(maxp+1, maxq+1):T], Δy[t] == α + δ*t + γ*y[t-1] + sum(ϕ[i]*Δy[t-i] for i=1:maxp-1) + ϵ[t] + sum(θ[j]*ϵ[t-j] for j=1:max(maxq-1,1)))
    @constraint(model, [i = 1:maxp+maxq], all_coefs[i]*(1 - I[i]) == 0) # WARNING: Non linear
    @constraint(model, sum(I) == K_var)
    @constraint(model, epi == sum(ϵ[i]^2 for i in 1:T))
    @objective(model, Min, epi)
    @expression(model, fit[t=max(maxp+1, maxq+1):T], α + δ*t + γ*y[t-1] + sum(ϕ[i]*Δy[t-i] for i=1:maxp-1) + ϵ[t] + sum(θ[j]*ϵ[t-j] for j=1:maxq-1))
    fix.(K_var, K)
    
    optimize!(model)
    @info("Solved for K= $K")
    aiccs = Vector{Float64}()
    aic = 2*K + T*log(var(value.(ϵ)))
    aicc_ = (aic + ((2*K^2 +2*K)/(T - K - 1)))
    fitted_model = sarimaxModel(value(α),value(δ), value(γ),value.(ϕ),value.(θ),value.(I),K,maxK,maxp,maxq,value.(ϵ),vcat([NaN for _=1:max(maxp,maxq)],  OffsetArrays.no_offset_view(value.(fit))),aicc_,silent)
    push!(aiccs, aicc_)
    K+=1
    while K <= maxK
        sarimaxModel!(fitted_model,value(α),value(δ),value(γ),value.(ϕ),value.(θ),value.(I),K-1,value.(ϵ),vcat([NaN for _=1:max(maxp,maxq)], OffsetArrays.no_offset_view(value.(fit))),aicc_)
        fix.(K_var, K)
        optimize!(model)
        @info("Solved for K = $K")

        aic = 2*K + T*log(var(value.(ϵ)))
        aicc_ = (aic + ((2*K^2 +2*K)/(T - K - 1)))
        push!(aiccs, aicc_)
        if aiccs[end] >= aiccs[end - 1]
            @info("Best K found")
            break
        end
       
        K += 1
    end
 
    return fitted_model
end

    
end # module Model