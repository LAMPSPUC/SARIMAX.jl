function generateARseries(p,s,ARcoeff, seasCoeff,trend,seed::Int = 1234, error::Bool = true)
    dates = Date(1991, 7, 1):Month(1):Date(2008, 2, 1)
    Random.seed!(seed)
    #Error terms:
    if error
        whiteNoise = randn(200) # Normal distribution mean = 0 and std error = 1 
    else
        whiteNoise = zeros(200)
    end
    #trend
    x = 1:200
    numInitialValues = max(s,p)
    seriesValues::Vector{Float64} = Vector{Float64}()
    for i in 1:numInitialValues
        value = randn() + trend*x[i] + whiteNoise[i]
        push!(seriesValues,value)
    end
    # seriesValues = randn(s) .+ trend*x[1:max(s,p)] .+ whiteNoise[1:max(s,p)]
    for i in numInitialValues+1:200
        value = seriesValues[i-s]*seasCoeff + sum(ARcoeff[j]*seriesValues[i-j] for j in 1:p) + trend*x[i] + whiteNoise[i]
        push!(seriesValues,value)
    end
    return TimeArray(dates,seriesValues)
end 

@testset "Sarima fit" begin
    @testset "fit p=0 P=1 without white noise" begin
        ARcoeff = [0]
        seasCoeff = 0.5
        trend = 0
        ARseries = generateARseries(1,12,ARcoeff, seasCoeff,trend,1234, false)
        modelMSE = SARIMA(ARseries,1,1,0;seasonality =12,P=1,D=0,Q=0)
        modelML = SARIMA(ARseries,1,1,0;seasonality =12,P=1,D=0,Q=0)
        modelBILEVEL = SARIMA(ARseries,1,1,0;seasonality =12,P=1,D=0,Q=0)
        Sarimax.fit!(modelMSE,objectiveFunction="mse")
        Sarimax.fit!(modelML,objectiveFunction="ml")
        Sarimax.fit!(modelBILEVEL,objectiveFunction="bilevel") 
        @test seasCoeff ≈ modelMSE.Φ[1] atol = 1e-3
        @test seasCoeff ≈ modelML.Φ[1] atol = 1e-3  
        @test seasCoeff ≈ modelBILEVEL.Φ[1] atol = 1e-3 
    end

    @testset "fit (p=1 P=0) and (p=2 P=0) without white noise" begin

        ar1 = generateARseries(1,1,[0.3],0,0,1234,false)
        modelAR1MSE = SARIMA(ar1,1,0,0;seasonality=12,P=0,D=0,Q=0)
        fit!(modelAR1MSE,objectiveFunction="mse")
        @test modelAR1MSE.ϕ ≈ [0.3] atol = 1e-3

        modelAR1ML = SARIMA(ar1,1,0,0;seasonality=12,P=0,D=0,Q=0)
        fit!(modelAR1ML,objectiveFunction="ml")
        @test modelAR1ML.ϕ ≈ [0.3] atol = 1e-3

        modelAR1BI = SARIMA(ar1,1,0,0;seasonality=12,P=0,D=0,Q=0)
        fit!(modelAR1BI,objectiveFunction="bilevel")
        @test modelAR1BI.ϕ ≈ [0.3] atol = 1e-3

        ar2 = generateARseries(2,1,[0.3,0.4],0,0,1234,false)
        modelAR2MSE = SARIMA(ar2,2,0,0;seasonality=12,P=0,D=0,Q=0)
        fit!(modelAR2MSE,objectiveFunction="mse")
        @test modelAR2MSE.ϕ ≈ [0.3,0.4] atol = 1e-3

        modelAR2ML = SARIMA(ar2,2,0,0;seasonality=12,P=0,D=0,Q=0)
        fit!(modelAR2ML,objectiveFunction="ml")
        @test modelAR2ML.ϕ ≈ [0.3,0.4] atol = 1e-3

        modelAR2BI = SARIMA(ar2,2,0,0;seasonality=12,P=0,D=0,Q=0)
        fit!(modelAR2BI,objectiveFunction="bilevel")
        @test modelAR2BI.ϕ ≈ [0.3,0.4] atol = 1e-3

    end

    @testset "fit p=2 P=1 without white Noise" begin
        ARcoeff = [-0.3,-0.2]
        seasCoeff = -0.4
        trend = 0.1
        ARseries = generateARseries(2,12,ARcoeff, seasCoeff,trend,1234,false)
        modelMSE = SARIMA(ARseries,2,1,0;seasonality =12,P=1,D=0,Q=0)
        modelML = SARIMA(ARseries,2,1,0;seasonality =12,P=1,D=0,Q=0)
        modelBILEVEL = SARIMA(ARseries,2,1,0;seasonality =12,P=1,D=0,Q=0)
        fit!(modelMSE,objectiveFunction="mse")
        fit!(modelML,objectiveFunction="ml")
        fit!(modelBILEVEL,objectiveFunction="bilevel")
        @test ARcoeff ≈ modelMSE.ϕ atol = 1e-3 
        @test seasCoeff ≈ modelMSE.Φ[1] atol = 1e-3 
        @test ARcoeff ≈ modelML.ϕ atol = 1e-3 
        @test seasCoeff ≈ modelML.Φ[1] atol = 1e-3 
        @test ARcoeff ≈ modelBILEVEL.ϕ atol = 1e-3 
        @test seasCoeff ≈ modelBILEVEL.Φ[1] atol = 1e-3 
    end

    @testset "fit M4 series" begin 
        test_series_json = JSON.parsefile("datasets/series_38351.json")
        train_dict = Dict{String,Vector{Float64}}("train" => test_series_json["train"])
        test_series_df = DataFrame(train_dict)
        series = loadDataset(test_series_df)
        autoModel = auto(series; seasonality = 12, seasonalIntegrationTest="ocsb", assertStationarity=true, assertInvertibility=true)
        @test autoModel.d == 1
        @test autoModel.D == 1
    end
end 