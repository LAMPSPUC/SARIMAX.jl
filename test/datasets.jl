@testset "Datasets" begin

    @testset "Datasets Enum" begin
        @test AIR_PASSENGERS == Datasets(1)
        @test GDPC1 == Datasets(2)
        @test NROU == Datasets(3)
    end

    @testset "loadDataset" begin
        @testset "loadDataset(Datasets)" begin
            @testset "AIR_PASSENGERS" begin
                airPassengersData = loadDataset(AIR_PASSENGERS)
                @test size(airPassengersData, 1) == 204
                @test values(airPassengersData)[1] == 3.526591
                @test values(airPassengersData)[end] == 19.43174
            end

            @testset "GDPC1" begin
                GDPC1Data = loadDataset(GDPC1)
                @test size(GDPC1Data, 1) == 344
                @test values(GDPC1Data)[1] == 2260.807
                @test values(GDPC1Data)[end] == 27944.500
            end

            @testset "NROU" begin
                NROUData = loadDataset(NROU)
                @test size(NROUData, 1) == 344
                @test values(NROUData)[1] == 5.2550525665283200
                @test values(NROUData)[end] == 4.2031234672198900
            end
        end

        @testset "loadDataset(DataFrame)" begin
            @testset "Air Passengers" begin
                airPassengersDf = CSV.File("../datasets/airpassengers.csv") |> DataFrame
                airPassengersData = loadDataset(airPassengersDf)
                @test size(airPassengersData, 1) == 204
                @test values(airPassengersData)[1] == 3.526591
                @test values(airPassengersData)[end] == 19.43174
            end

            @testset "GDPC1" begin
                GDPC1Df = CSV.File("../datasets/GDPC1.csv") |> DataFrame
                GDPC1Data = loadDataset(GDPC1Df)
                @test size(GDPC1Data, 1) == 344
                @test values(GDPC1Data)[1] == 2260.807
                @test values(GDPC1Data)[end] == 27944.500
            end

            @testset "NROU" begin
                NROUDf = CSV.File("../datasets/NROU.csv") |> DataFrame
                NROUData = loadDataset(NROUDf)
                @test size(NROUData, 1) == 344
                @test values(NROUData)[1] == 5.2550525665283200
                @test values(NROUData)[end] == 4.2031234672198900
            end

            @testset "Date in not the first column" begin
                df = DataFrame(Datas = ["2020-01-01", "2020-01-02", "2020-01-03"], Values = [1, 2, 3])
                data = loadDataset(df, true)
                println(data)
                println(values(data))
                @test size(data, 1) == 3
                @test values(data)[1,2] == 1
                @test values(data)[end,2] == 3
            end
        end
    end

    @testset "splitTrainTest" begin
        @testset "splitTrainTest(Datasets)" begin
            @testset "AIR_PASSENGERS" begin
                airPassengers = loadDataset(AIR_PASSENGERS)
                train, test = splitTrainTest(airPassengers; trainPercentage = 0.8)
                @test size(train, 1) == 163
                @test size(test, 1) == 41
            end

            @testset "GDPC1" begin
                GDPC1Data = loadDataset(GDPC1)
                train, test = splitTrainTest(GDPC1Data; trainPercentage = 0.8)
                @test size(train, 1) == 275
                @test size(test, 1) == 69
            end

            @testset "NROU" begin
                NROUData = loadDataset(NROU)
                train, test = splitTrainTest(NROUData; trainPercentage = 0.8)
                @test size(train, 1) == 275
                @test size(test, 1) == 69
            end
        end
    end
end
