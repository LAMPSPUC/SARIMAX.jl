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

    @test isapprox(integrate(values(y[1:1]), values_diff_1_0, 1, 0, 12), values(y); atol = 1e-3)
    @test isapprox(integrate(values(y[1:12]), values_diff_0_1, 0, 1, 12), values(y); atol = 1e-3)
    @test isapprox(integrate(values(y[1:13]), values_diff_1_1, 1, 1, 12), values(y); atol = 1e-3)
    @test isapprox(integrate(values(y[1:2]), values_diff_2_0, 2, 0, 12), values(y); atol = 1e-3)
    @test isapprox(integrate(values(y[1:14]), values_diff_2_1, 2, 1, 12), values(y); atol = 1e-3)
end
