module Sarimax

import Base: show, print, copy, deepcopy, showerror

using Alpine
using Combinatorics
using CSV
using DataFrames
using Dates
using Distributions
using Ipopt
using JuMP
using LinearAlgebra
using MathOptInterface
using OffsetArrays
using Optim
using Pkg
using Random
using Requires
using StateSpaceModels
using Statistics
using TimeSeries

abstract type SarimaxModel end

function __init__()
    @require PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0" include("python_init.jl")
    @require RCall = "6f49c342-dc21-5d91-9882-a32aef131414" include("r_init.jl")
end

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
export MissingExogenousData
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
export toMA
export differentiatedCoefficients


end # module Sarimax
