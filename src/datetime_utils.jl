"""
    buildDatetimes(startDatetime, granularity, weekDaysOnly, datetimesLength)

Builds an array of DateTime objects based on a given starting DateTime, granularity, and length.

# Arguments
- `startDatetime::DateTime`: The DateTime from which the dateTime array will be computed. It won't be included in the final array
- `granularity::Dates.Period`: The granularity by which to increment the timestamps.
- `weekDaysOnly::Bool`: Whether to include only weekdays (Monday to Friday) in the timestamps.
- `datetimesLength::Int64`: The length of the array of DateTime objects to build.

# Returns
An array of DateTime objects.

"""
function buildDatetimes(
    startDatetime::DateTime, 
    granularity::Dates.Period,
    weekDaysOnly::Bool, 
    datetimesLength::Int64,
)::Vector{DateTime}
    if datetimesLength == 0
        return DateTime[]
    end

    # Initialize the array with the starting datetime
    datetimes = []

    currentDatetime = startDatetime

    # Loop to generate timestamps based on granularity
    for _ in 1:datetimesLength
        if weekDaysOnly && dayofweek(currentDatetime) == 5
            currentDatetime += Dates.Day(3)
        else
            currentDatetime += granularity
        end
        push!(datetimes, currentDatetime)
    end

    return datetimes
end



"""
    identifyGranularity(datetimes::Vector{DateTime})

Identifies the granularity of an array of timestamps.

# Arguments
- `datetimes::Vector{DateTime}`: An array of DateTime objects.

# Returns
A tuple `(granularity, frequency, weekdays)` where:
- `granularity`: The identified granularity, which could be one of [:Second, :Minute, :Hour, :Day, :Week, :Month, :Year].
- `frequency`: The frequency of the identified granularity.
- `weekdays`: A boolean indicating whether the series uses weekdays only.

# Errors
Throws an error if the timestamps do not follow a consistent pattern.

"""
function identifyGranularity(datetimes::Vector{DateTime})
    # Define base units and periods
    baseUnits = [Second(1), Minute(1), Hour(1), Day(1), Week(1)]
    basePeriods = [:Second, :Minute, :Hour, :Day, :Week, :Month, :Year]
    
    unitPeriod = nothing
    diffBetweenTimestamps = nothing
    weekDaysSeries = false
    
    for (i, unit) in enumerate(baseUnits)
        differences = diff(datetimes) ./ unit
        min_difference = minimum(differences)

        lessThanOne = all(differences .< 1)
        if lessThanOne
            break
        end
        
        # Check if all elements are equal
        regularDistribution = all(differences .== differences[1])
        if regularDistribution
            unitPeriod = basePeriods[i]
            diffBetweenTimestamps = differences[1]
            
            if unit in [Minute(1), Second(1)]
                if diffBetweenTimestamps < 60
                    break
                else
                    continue
                end
            elseif unit == Hour(1)
                if diffBetweenTimestamps < 24
                    break
                else
                    continue
                end
            elseif unit == Day(1)
                if diffBetweenTimestamps < 7
                    break
                elseif diffBetweenTimestamps % 7 != 0
                    break
                else
                    continue
                end
            end

            break
        else
            if unit == Day(1)
                # Check if the series uses weekdays only
                allWeekdays = all([dayofweek(dt) <= 5 for dt in datetimes])
                if allWeekdays && min_difference < 7
                    unitPeriod = :Day
                    diffBetweenTimestamps = 1
                    weekDaysSeries = true
                    break
                end
            end
        end
 
        amplitude = maximum(differences) - min_difference
        if amplitude < 1
            unitPeriod = basePeriods[i]
            diffBetweenTimestamps = round(Int, min_difference)
            break
        end
    end

    if isnothing(unitPeriod)
        throw(InconsistentDatePattern())
    end

    # Analyze the minimum difference to determine granularity
    if unitPeriod == :Week
        if diffBetweenTimestamps % 52 == 0
            unitPeriod = :Year
            diffBetweenTimestamps = diffBetweenTimestamps / 52
        elseif diffBetweenTimestamps % 13 == 0
            unitPeriod = :Month
            diffBetweenTimestamps = 3
        elseif diffBetweenTimestamps % 4 == 0
            unitPeriod = :Month
            diffBetweenTimestamps = diffBetweenTimestamps / 4
        end 
    end
    
    return (granularity=unitPeriod, frequency=diffBetweenTimestamps, weekdays=weekDaysSeries)
end


