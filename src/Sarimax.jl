module Sarimax

using JuMP, SCIP, Plots

include("src/Parameters.jl")
using .Parameters


include("src/Models.jl")
using .Models

@enum ArimaType ari arima

"""

Split the time series in training and testing sets. 

# Arguments
- `y::Vector{Float64}`: the time series
- `trainSize::Integer=-1`: the size of the training set. If no size is provided the training set will be the first 90% of the data

"""
function splitTrainTest(y::Vector{Float64}; trainSize::Integer=-1)
    trainSize = (trainSize != -1 && trainSize >= 0) ? trainSize : floor(0.9*length(y))
    return y[begin:trainSize], y[trainSize+1:end]
end

function fit(y::Vector{Float64};silent::Bool=true,optimizer::DataType = SCIP.Optimizer,arimaType::ArimaType=arima)
    if arimaType == ari
        return Models.opt_ari(y;silent=silent)
    end

    return Models.arima(y;silent=silent)
end
# Testar Alpine(garante optimalidade)

using ARFIMA, Distributions
sigma = 0.1

# Generate an ARMA(2,0) series
n = 200
phi = rand(Uniform(-0.99,0.99),3)
size_phi = length(phi)
y = arma(n, sigma, SVector{size_phi,Float64}(phi))
plot(y)
models , k = fit(y;silent=false,arimaType=ari)
best_model = models[k]
aiccs = map(x->x.aicc,models)
println("Aicc: ",aiccs)
Models.print(best_model)
real_ϕ = [best_model.γ + best_model.ϕ[1]+1]
foreach(j->push!(real_ϕ,best_model.ϕ[j]-best_model.ϕ[j-1]),[i for i=2:best_model.maxp-1])
push!(real_ϕ,best_model.ϕ[end])
println("Real ϕ = ",real_ϕ)
fit_in_sample = best_model.fitInSample
plot!(fit_in_sample)
end # module Sarimax
