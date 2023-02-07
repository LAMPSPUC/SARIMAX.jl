module Sarimax

using JuMP, SCIP

include("src/Parameters.jl")
using .Parameters


include("src/Models.jl")
using .Models


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

function fit(y::Vector{Float64};silent::Bool=true, optimizer::DataType = SCIP.Optimizer)
    return Models.arima(y;silent=silent)
end

end # module Sarimax
