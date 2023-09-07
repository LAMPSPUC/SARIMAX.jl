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
using DataFrames
using CSV
# using GLMNet
# using Lasso

abstract type SarimaxModel end

include("datasets.jl")
include("utils.jl")
include("models/sarima.jl")

# Export types
export SARIMAModel

# Export enums
export Datasets

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
export loadDataset


end # module Sarimax
