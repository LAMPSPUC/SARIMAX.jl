module Sarimax

using JuMP, SCIP, Plots, TimeSeries, Ipopt, TimeSeries

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

function SARIMA(y::TimeArray,p::Int64,d::Int64,q::Int64;silent::Bool=true)
    return SARIMAModel(y,p,d,q;silent=true)
end

function fit!(model::SARIMAModel;silent::Bool=true,optimizer::DataType=Ipopt.Optimizer)
    return Models.arima(model;silent=silent,optimizer=optimizer)
end

function predict!(model::SARIMAModel, stepsAhead::Int64=12)
    return Models.arima(model;silent=silent,optimizer=optimizer)
end

using CSV, DataFrames, Plots, TimeSeries

y = CSV.read("dataset.csv", DataFrame)
y = TimeArray(y, timestamp = :date)

plot(y)

modelo = SARIMA(y,1,1,0)
fit!(modelo)
Models.print(modelo)


end # module Sarimax
