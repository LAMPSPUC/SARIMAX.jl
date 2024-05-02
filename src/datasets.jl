export AIR_PASSENGERS, GDPC1, NROU


@enum Datasets begin
    AIR_PASSENGERS=1
    GPDC1=2 
    NROU=3
end

datasetsPaths = [
    joinpath(dirname(@__DIR__()), "datasets", "airpassengers.csv"), 
    joinpath(dirname(@__DIR__()), "datasets", "GDPC1.csv"),
    joinpath(dirname(@__DIR__()), "datasets", "NROU.csv")
]



"""
    loadDataset(
        dataset::Datasets
    )

Loads a dataset from the `Datasets` enum. 

# Example
```jldoctest
julia> airPassengers = loadDataset(AIR_PASSENGERS)
204×1 TimeArray{Float64, 1, Date, Vector{Float64}} 1991-07-01 to 2008-06-01
│            │ value   │
├────────────┼─────────┤
│ 1991-07-01 │ 3.5266  │
│ 1991-08-01 │ 3.1809  │
│ ⋮          │ ⋮       │
│ 2008-06-01 │ 19.4317 │

```
"""
function loadDataset(dataset::Datasets)
    datasetIndex = Integer(dataset)
    seriesData = CSV.read(datasetsPaths[datasetIndex], DataFrame)
    y = TimeArray(seriesData, timestamp = :date)
    return y
end

"""
    loadDataset(
        df::DataFrame
    )

Loads a dataset from a Dataframe. If the DataFrame does not have a column named
`date` a new column will be created with the index of the DataFrame. 

# Example
```jldoctest
julia> airPassengersDf = CSV.File("datasets/airpassengers.csv") |> DataFrame
julia> airPassengers = loadDataset(airPassengersDf)
204×1 TimeArray{Float64, 1, Date, Vector{Float64}} 1991-07-01 to 2008-06-01
│            │ value   │
├────────────┼─────────┤
│ 1991-07-01 │ 3.5266  │
│ 1991-08-01 │ 3.1809  │
│ ⋮          │ ⋮       │
│ 2008-06-01 │ 19.4317 │

```
"""
function loadDataset(df::DataFrame)
    auxiliarDF = deepcopy(df)
    if !(:date in propertynames(auxiliarDF))
        @info("The DataFrame does not have a column named 'date'.")
        @info("Creating a date column with the index of the DataFrame")
        auxiliarDF[!,:date] = [Date(i) for i=1:size(auxiliarDF,1)]
    end
    y = TimeArray(auxiliarDF, timestamp = :date)
    return y
end

"""
    splitTrainTest(
        data::TimeArray;
        trainSize::Float64=0.8
    )

Splits the time series in training and testing sets. 
"""
function splitTrainTest(data::TimeArray; trainPercentage::Float64=0.8)
    trainingSetEndIndex = floor(Int64, trainPercentage*length(data))
    trainingSet = TimeArray(timestamp(data)[1:trainingSetEndIndex], values(data)[1:trainingSetEndIndex])
    testingSet = TimeArray(timestamp(data)[trainingSetEndIndex+1:end], values(data)[trainingSetEndIndex+1:end])
    return trainingSet, testingSet
end