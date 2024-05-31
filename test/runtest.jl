BASE_PATH = joinpath(dirname(@__DIR__()),"Sarimax")

include(joinpath(BASE_PATH,"src/Sarimax.jl"))
using .Sarimax

using Dates
using Statistics
using Test
using Random


# Testes dos modelos
include(joinpath(BASE_PATH,"test/models/sarima.jl"))

include(joinpath(BASE_PATH,"test/models/sarima_fit.jl"))

include(joinpath(BASE_PATH,"test/models/sarima_predict.jl"))

include(joinpath(BASE_PATH,"test/datetime_utils.jl"))

include(joinpath(BASE_PATH,"test/utils.jl"))