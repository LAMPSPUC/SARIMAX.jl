@testset "utils" begin

    @testset "Differentiate" begin
        airPassengers = loadDataset(AIR_PASSENGERS)
        diff_0_0 = differentiate(airPassengers, 0, 0, 12)
        @test size(diff_0_0) == (204,)
        @test values(diff_0_0) == values(airPassengers)

        diff_1_0 = differentiate(airPassengers, 1, 0, 12)
        @test size(diff_1_0) == (203,)
        @test values(diff_1_0) == [values(airPassengers)[i] - values(airPassengers)[i - 1] for i in 2:204]

        diff_0_1 = differentiate(airPassengers, 0, 1, 12)
        @test size(diff_0_1) == (192,)
        @test values(diff_0_1) == [values(airPassengers)[i] - values(airPassengers)[i - 12] for i in 13:204]

        diff_1_1 = differentiate(airPassengers, 1, 1, 12)
        @test size(diff_1_1) == (191,)
        @test isapprox(values(diff_1_1), [values(airPassengers)[i] - values(airPassengers)[i - 1] - values(airPassengers)[i - 12] + values(airPassengers)[i - 13] for i in 14:204], atol=1e-6)
    end
    @testset "Test Differentiated Coefficients Function" begin
        # Test case 1
        d = 1
        D = 0
        s = 1
        expected_output = [1.0, -1.0]
        @test differentiatedCoefficients(d, D, s) == expected_output
        
        # Test case 2
        d = 2
        D = 1
        s = 4
        expected_output = [1.0, -2.0, 1.0, 0.0, -1.0, 2.0, -1.0]
        @test differentiatedCoefficients(d, D, s) == expected_output
        
        # Test case 3
        d = 1
        D = 1
        s = 12
        expected_output = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]
        @test differentiatedCoefficients(d, D, s) == expected_output
    end

    @testset "Testing integrate function" begin
        # Load dataset and differentiate series
        y = loadDataset(AIR_PASSENGERS)
        diff_1_1 = differentiate(y, 1, 1, 12)
        diff_0_1 = differentiate(y, 0, 1, 12)
        diff_1_0 = differentiate(y, 1, 0, 12)
        diff_2_0 = differentiate(y, 2, 0, 12)
        diff_2_1 = differentiate(y, 2, 1, 12)

        # Extract values from differentiated series
        values_diff_1_0::Vector{Float64} = values(diff_1_0)
        values_diff_0_1::Vector{Float64} = values(diff_0_1)
        values_diff_1_1::Vector{Float64} = values(diff_1_1)
        values_diff_2_0::Vector{Float64} = values(diff_2_0)
        values_diff_2_1::Vector{Float64} = values(diff_2_1)

        @test isapprox(integrate(values(y[1:1]), values_diff_1_0, 1, 0, 12), values(y); atol = 1e-5)
        @test isapprox(integrate(values(y[1:12]), values_diff_0_1, 0, 1, 12), values(y); atol = 1e-5)
        @test isapprox(integrate(values(y[1:13]), values_diff_1_1, 1, 1, 12), values(y); atol = 1e-5)
        @test isapprox(integrate(values(y[1:2]), values_diff_2_0, 2, 0, 12), values(y); atol = 1e-5)
        @test isapprox(integrate(values(y[1:14]), values_diff_2_1, 2, 1, 12), values(y); atol = 1e-5)
    end

    @testset "selectSeasonalIntegrationOrder" begin
        airPassengers = loadDataset(AIR_PASSENGERS)
        @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12,"seas") == 1
        @test Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12,"ch") == 0
        @test_throws ArgumentError Sarimax.selectSeasonalIntegrationOrder(values(airPassengers), 12,"hegy")
    end

    @testset "selectIntegrationOrder" begin
        airPassengers = loadDataset(AIR_PASSENGERS)
        @test Sarimax.selectIntegrationOrder(values(airPassengers), 2, 0, 12,"kpss") == 1
        @test_throws ArgumentError Sarimax.selectIntegrationOrder(values(airPassengers), 2, 0, 12,"hegy")
    end

    @testset "automaticDifferentiation" begin 
        gdpc1Data = loadDataset(GDPC1)
        nrouData = loadDataset(NROU)
        seriesVector::Vector{TimeArray} = [gdpc1Data, nrouData]
        mergedTimeArray = Sarimax.merge(seriesVector)

        @test_throws AssertionError automaticDifferentiation(mergedTimeArray; integrationTest = "test")
        @test_throws AssertionError automaticDifferentiation(mergedTimeArray; seasonalIntegrationTest = "test")
        @test_throws AssertionError automaticDifferentiation(mergedTimeArray; seasonalPeriod = -1)

        mergedDiffSeries, diffMetadata = automaticDifferentiation(mergedTimeArray; integrationTest = "kpss", seasonalIntegrationTest = "ch", seasonalPeriod = 12)
        
        @test size(mergedDiffSeries,2) == size(mergedTimeArray,2)
        @test colnames(mergedDiffSeries) == colnames(mergedTimeArray)

        for col in colnames(mergedTimeArray)
            @test diffMetadata[col][:d] == 2
            @test diffMetadata[col][:D] == 0
            @test size(mergedDiffSeries[col],1) == size(mergedTimeArray[col],1) - diffMetadata[col][:d]
        end
    end

    @testset "logLikelihood and loglike" begin
        mutable struct TestModelUtil <: Sarimax.SarimaxModel
        end
        
        @test_throws MissingMethodImplementation loglikelihood(TestModelUtil())
        @test_throws MissingMethodImplementation loglike(TestModelUtil())

        airPassengers = loadDataset(AIR_PASSENGERS)
        airPassengersLog = log.(airPassengers)
        testModel = SARIMA(airPassengersLog, 3, 0, 1; seasonality=12, P=1, D=1, Q=1)
        
        @test_throws ModelNotFitted loglikelihood(testModel)
        @test_throws ModelNotFitted loglike(testModel)

        fit!(testModel)

        @test loglikelihood(testModel) ≈ 254.01147989290882
        @test loglike(testModel) ≈ 254.01147989290882
    end

end