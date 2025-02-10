using RCall

export seasonal_diffsR, kpssR

try 
    R"""
    library(forecast)
    """
catch e
    @warn "Could not load R libraries. Please make sure you have forecast installed."
end

function seasonal_diffsR(y, seasonality)
    @rput y seasonality
    R"""
    # Example time series data (replace with your actual data)
    ts_data <- ts(y, frequency = seasonality)
    D <- nsdiffs(ts_data)
    """
    D::Int = @rget D
    return D
end

function kpssR(y, maxd, D, seasonality)
    @rput y maxd D seasonality
    R"""
    diffy <- y
    if (D > 0 & seasonality > 1) {
        diffy <- diff(y, differences = D, lag = seasonality)
    }
    d <- ndiffs(diffy, test="kpss", max.d = maxd)
    """
    d::Int = @rget d
    return d
end