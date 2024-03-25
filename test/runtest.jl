BASE_PATH = joinpath(dirname(@__DIR__()),"Sarimax")

include(joinpath(BASE_PATH,"src/Sarimax.jl"))
using .Sarimax

using Dates
using Statistics
using Test

# Testes dos modelos
include(joinpath(BASE_PATH,"test/models/sarima.jl"))

include(joinpath(BASE_PATH,"test/datetime_utils.jl"))