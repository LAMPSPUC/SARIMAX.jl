using Documenter
include("../src/Sarimax.jl")

DocMeta.setdocmeta!(Sarimax, :DocTestSetup, :(using Sarimax); recursive=true)

makedocs(;
    modules=[Sarimax],
    doctest=true,
    clean=true,
    checkdocs=:none,
    format=Documenter.HTML(; mathengine=Documenter.MathJax2()),
    sitename="SARIMAX.jl",
    authors="Luiz Fernando Duarte",
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(; repo="github.com/LAMPSPUC/SARIMAX.jl.git", push_preview=true)
