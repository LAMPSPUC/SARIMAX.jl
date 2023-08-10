using CSV
using DataFrames
using Plots
using TimeSeries
using Ipopt
include("/Users/luizfernandoduarte/Documents/Sarimax Code /Sarimax/src/Sarimax.jl")
using .Sarimax

y = CSV.read("/Users/luizfernandoduarte/Documents/Sarimax Code /Sarimax/dataset.csv", DataFrame)
y = TimeArray(y, timestamp = :date)

y_log = log.(y)
modelo_log = SARIMA(y_log,0,1,1;seasonality=12,P=0,D=1,Q=1)
fit!(modelo_log)
print(modelo_log)
plot(modelo_log.ϵ)
sum(modelo_log.ϵ.^2)
plot(modelo_log.y)
plot!(modelo_log.fitInSample)
predict!(modelo_log,12)
forecast = values(modelo_log.forecast)
length(modelo_log.ϵ)

for i=1:12
    forecast[i] = forecast[i] + values(modelo_log.y)[end-12+i] 
end

forecast[1] += values(modelo_log.y)[end]
for i=2:12
    forecast[i] = forecast[i] + forecast[i-1] 
end

for i=1:12
    forecast[i] = forecast[i] - values(modelo_log.y)[end-13+i]
end
forecast = TimeArray(timestamp(modelo_log.forecast),forecast)
plot!(forecast)
plot!(modelo_log.forecast)


modelo = SARIMA(y,0,1,1;seasonality=12,P=0,D=1,Q=1)
fit!(modelo)
Models.print(modelo)
sum(modelo.ϵ.^2)
plot(modelo.y)
plot!(modelo.fitInSample)
predict!(modelo,12)
plot!(modelo.forecast)


modelo_ari = OPTSARIMA(y)
fit!(modelo_ari)
Models.print(modelo_ari)
sum(modelo_ari.ϵ.^2)
plot(modelo_ari.y)
plot!(modelo_ari.fitInSample)
predict!(modelo_ari,12)
plot!(modelo_ari.forecast)

y_log = log.(y)
modelo_ari = OPTSARIMA(y_log)
fit!(modelo_ari)
Models.print(modelo_ari)
sum(modelo_ari.ϵ.^2)
plot(modelo_ari.y)
plot!(modelo_ari.fitInSample)
predict!(modelo_ari,12)
plot!(modelo_ari.forecast)