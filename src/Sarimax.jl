module Sarimax

import Base: print, copy

using JuMP
using SCIP
using Ipopt
using TimeSeries
using MathOptInterface
using LinearAlgebra
using Statistics
using OffsetArrays
using Distributions
# using GLMNet
# using Lasso

abstract type SarimaxModel end

include("datasets.jl")
include("models/sarima.jl")

# Export types
export SARIMAModel


# Export functions
export splitTrainTest
export print
export copy
export fit!
export predict!
export SARIMA
export differentiate
export integrate
export simulate


end # module Sarimax
