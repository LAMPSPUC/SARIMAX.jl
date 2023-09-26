@testset "SARIMA" begin

    @test hasFitMethods(SARIMAModel)
    @test hasHyperparametersMethods(SARIMAModel)

    airPassengers = loadDataset(AIR_PASSENGERS)
    airPassengersLog = log.(airPassengers)

    modeloLog = SARIMA(airPassengersLog, 3, 0, 1; seasonality=12, P=1, D=1, Q=1)
    @test Sarimax.isFitted(modeloLog) == false
    @test getHyperparametersNumber(modeloLog) == 7
    fit!(modeloLog)
    @test Sarimax.isFitted(modeloLog) == true
    @test mean(modeloLog.ϵ) ≈ 0 atol=1e-1

    autoModelML = auto(airPassengersLog; seasonality=12 ,objectiveFunction="ml")
    @test autoModelML.d == 0 # Output of forecast package in R
    @test autoModelML.D == 1 # Output of forecast package in R

end