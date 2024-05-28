[build-img]: https://github.com/LAMPSPUC/SARIMAX.jl/actions/workflows/ci.yml/badge.svg?branch=master
[build-url]: https://github.com/LAMPSPUC/SARIMAX.jl/actions/workflows/ci.yml

[codecov-img]: https://app.codecov.io/github/LAMPSPUC/SARIMAX.jl/coverage.svg?branch=master
[codecov-url]: https://app.codecov.io/github/LAMPSPUC/SARIMAX.jl?branch=master

# Sarimax.jl

| **Build Status** | **Coverage** |
|:-----------------:|:-----------------:|:-----------------:|
| [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |

Introducing Sarimax.jl, a groundbreaking Julia package that redefines SARIMA (Seasonal Autoregressive Integrated Moving Average) modeling by seamlessly integrating the JuMP framework — a powerful optimization modeling language. Unlike traditional SARIMA methods, Sarimax.jl leverages the optimization capabilities of JuMP, allowing for precise and customizable SARIMA models.

## Quickstart
```julia
import Pkg

Pkg.add("https://github.com/LuizFCDuarte/Sarimax.jl.git")

using Sarimax

y = loadDataset(AIR_PASSENGERS)
yLog = log.(y)

model = SARIMA(yLog, 1, 0, 1; seasonality=12, P=0, D=1, Q=2, silent=false, allowMean=false)
fit!(model; objectiveFunction="mse")
print(model)
predict!(model, 12)
scenarios = simulate(model, 12, 200)

```
## Features

* Fit using Mean Squared Errors objective function
* Fit using Maximum Likelihood estimation
* Auto Sarima Model
* Simulate scenarios


### Auto SARIMA method
```julia
autoModelMSE = auto(y_log; seasonality=12, objectiveFunction="mse")
loglike(autoModelMSE)
aicc(autoModelMSE)

autoModelML = auto(y_log; seasonality=12, objectiveFunction="ml")
bic(autoModelML)
loglikelihood(autoModelML)
```

## Contributing

* PRs such as adding new models and fixing bugs are very welcome!
* For nontrivial changes, you'll probably want to first discuss the changes via issue.
