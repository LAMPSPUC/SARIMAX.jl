module Sarimax

import Base: print, copy, deepcopy, showerror

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
using StateSpaceModels
# using GLMNet
# using Lasso

abstract type SarimaxModel end

include("datasets.jl")
include("utils.jl")
include("models/sarima.jl")
include("exceptions.jl")
include("fit.jl")

# Export types
export SARIMAModel

# Export Exceptions/Errors
export ModelNotFitted
export MissingMethodImplementation

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
export loglikelihood
export loglike
export hasFitMethods
export hasHyperparametersMethods
export getHyperparametersNumber
export auto
export aic
export aicc
export bic


end # module Sarimax
