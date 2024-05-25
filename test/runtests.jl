using Dates
using Sarimax
using Statistics
using Test

# Testes dos modelos
include(joinpath(BASE_PATH,"test/models/sarima.jl"))

include(joinpath(BASE_PATH,"test/datetime_utils.jl"))

include(joinpath(BASE_PATH,"test/utils.jl"))