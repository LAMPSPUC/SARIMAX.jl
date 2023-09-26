include(joinpath(dirname(@__DIR__()),"Sarimax/src/Sarimax.jl"))
using .Sarimax
using Test
using Statistics

# Testes dos modelos
include("test/models/sarima.jl")