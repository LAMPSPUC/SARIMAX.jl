@testset "buildDatetimes tests" begin
    # Test case for building timestamps annually
    @testset "Test annually" begin
        start_datetime = DateTime("2020-01-01T00:00:00")
        granularity = Dates.Year(1)
        weekDaysOnly = false
        datetimes_length = 10
        result = buildDatetimes(start_datetime, granularity, weekDaysOnly, datetimes_length)
        expected_result = [DateTime("2021-01-01T00:00:00"), DateTime("2022-01-01T00:00:00"),
                           DateTime("2023-01-01T00:00:00"), DateTime("2024-01-01T00:00:00"), DateTime("2025-01-01T00:00:00"),
                           DateTime("2026-01-01T00:00:00"), DateTime("2027-01-01T00:00:00"), DateTime("2028-01-01T00:00:00"),
                           DateTime("2029-01-01T00:00:00"),DateTime("2030-01-01T00:00:00")]
        @test result == expected_result
    end
    
    # Test case for building timestamps daily with weekdays only
    @testset "Test daily weekdays only" begin
        start_datetime = DateTime("2024-03-27T00:00:00")
        granularity = Dates.Day(1)
        weekDaysOnly = true
        datetimes_length = 10
        result = buildDatetimes(start_datetime, granularity, weekDaysOnly, datetimes_length)
        expected_result = [DateTime("2024-03-28T00:00:00"), DateTime("2024-03-29T00:00:00"),
                           DateTime("2024-04-01T00:00:00"), DateTime("2024-04-02T00:00:00"), DateTime("2024-04-03T00:00:00"),
                           DateTime("2024-04-04T00:00:00"), DateTime("2024-04-05T00:00:00"), DateTime("2024-04-08T00:00:00"),
                           DateTime("2024-04-09T00:00:00"), DateTime("2024-04-10T00:00:00")]
        @test result == expected_result
    end
end

# Define the timestamps for testing
timestampsAnnually = [DateTime("2020-01-01T00:00:00") + Dates.Year(i) for i in 0:9]
timestampsMonthly = [DateTime("2024-01-01T00:00:00") + Dates.Month(i) for i in 0:9]
timestampsQuarterly = [DateTime("2024-01-01T00:00:00") + Dates.Month(3*i) for i in 0:9]
timestamps15Days = [DateTime("2024-03-24T00:00:00") + Dates.Day(15*i) for i in 0:9]
timestampsWeekly = [DateTime("2024-01-01T00:00:00") + Dates.Week(i) for i in 0:9]
timestampsDaily = [DateTime("2024-03-24T00:00:00") + Dates.Day(i) for i in 0:9]
timestampsHourly = [DateTime("2024-03-24T00:00:00") + Dates.Hour(i) for i in 0:9]
timestampsLessHourly = [DateTime("2024-03-24T00:00:00") + Dates.Minute(15*i) for i in 0:9]
timestampsInconsistent = [DateTime("2024-03-24T00:00:00")]
for i in 1:8
    times = timestampsInconsistent[i] + Dates.Hour(rand(1:3))
    push!(timestampsInconsistent, times)
end

timestampsWeekdays =[DateTime("2024-03-25T00:00:00") + Dates.Day(i) for i in 0:4]
timestampsWeekdays = vcat(timestampsWeekdays, [DateTime("2024-04-01T00:00:00") + Dates.Day(i) for i in 0:4])

timestampsNotWeekdays = [DateTime("2024-03-28T00:00:00") + Dates.Day(i) for i in 0:10]


# Write test cases
@testset "identifyGranularity tests" begin
    @testset "Test annually" begin
        result = identifyGranularity(timestampsAnnually)
        @test result == (granularity = :Year, frequency = 1, weekdays = false)
    end
    
    @testset "Test monthly" begin
        result = identifyGranularity(timestampsMonthly)
        @test result == (granularity = :Month, frequency = 1, weekdays = false)
    end
    
    @testset "Test quarterly" begin
        result = identifyGranularity(timestampsQuarterly)
        @test result == (granularity = :Month, frequency = 3, weekdays = false)
    end
    
    @testset "Test 15 days" begin
        result = identifyGranularity(timestamps15Days)
        @test result == (granularity = :Day, frequency = 15, weekdays = false)
    end
    
    @testset "Test weekly" begin
        result = identifyGranularity(timestampsWeekly)
        @test result == (granularity = :Week, frequency = 1, weekdays = false)
    end
    
    @testset "Test daily" begin
        result = identifyGranularity(timestampsDaily)
        @test result == (granularity = :Day, frequency = 1, weekdays = false)
    end
    
    @testset "Test hourly" begin
        result = identifyGranularity(timestampsHourly)
        @test result == (granularity = :Hour, frequency = 1, weekdays = false)
    end
    
    @testset "Test less than hourly" begin
        result = identifyGranularity(timestampsLessHourly)
        @test result == (granularity = :Minute, frequency = 15, weekdays = false)
    end
    
    @testset "Test inconsistent" begin
        @test_throws InconsistentDatePattern identifyGranularity(timestampsInconsistent)
    end
    
    @testset "Test weekdays" begin
        result = identifyGranularity(timestampsWeekdays)
        @test result == (granularity = :Day, frequency = 1, weekdays = true)

        result = identifyGranularity(timestampsNotWeekdays)
        @test result == (granularity = :Day, frequency = 1, weekdays = false)
    end
end
