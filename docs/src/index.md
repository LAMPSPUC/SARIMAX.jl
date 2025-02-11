```raw html
<!-- Ensure that raw HTML is properly formatted -->
<div style="width:100%; height:150px;border-width:4px;border-style:solid;padding-top:25px;
    border-color:#000;border-radius:10px;text-align:center;background-color:#99DDFF;
    color:#000">
    <h3 style="color: black;">Star us on GitHub!</h3>
    <a class="github-button" href="https://github.com/LAMPSPUC/SARIMAX.jl" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star LAMPSPUC/SARIMAX.jl on GitHub" style="margin:auto">Star</a>
    <script async defer src="https://buttons.github.io/buttons.js"></script>
</div>
```

# Sarimax.jl

## Installation

This package is registered so you can simply `add` it using Julia's `Pkg` manager:
```julia
using Pkg
Pkg.add("SARIMAX")
```

Auto SARIMA implementation
```@docs
Sarimax.auto
```
