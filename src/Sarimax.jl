module Sarimax

import Base: show, print, copy, deepcopy, showerror

using Alpine
using Combinatorics
using CSV
using DataFrames
using Dates
using Distributions
using HiGHS
using Ipopt
using JuMP
using LinearAlgebra
using MathOptInterface
using OffsetArrays
using Optim
using Random
using SCIP
using StateSpaceModels
using Statistics
using TimeSeries
# using GLMNet
# using Lasso

abstract type SarimaxModel end

include("datasets.jl")
include("datetime_utils.jl")
include("exceptions.jl")
include("fit.jl")
include("models/sarima.jl")
include("utils.jl")


# Export types
export SARIMAModel

# Export Exceptions/Errors
export ModelNotFitted
export MissingMethodImplementation
export InconsistentDatePattern
export InvalidParametersCombination

# Export enums
export Datasets

# Export functions
export automaticDifferentiation
export splitTrainTest
export print
export copy
export fit!
export predict!
export SARIMA
export differentiate
export identifyGranularity
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
export buildDatetimes
export differentiatedCoefficients


end # module Sarimax
