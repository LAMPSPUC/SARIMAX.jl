using PyCall

export seasonal_diffs

try
    pyimport("pmdarima")
    pyimport("numpy")
catch e
    @warn "Could not load python libraries. Please make sure you have numpy and pmdarima installed."
end

function seasonal_diffs(y, seasonality)
    py"""
    def seasonal_diffs(ts, seasonal_period):
        ts_np = numpy.array(ts)
        return pmdarima.arima.nsdiffs(ts_np, m=seasonal_period)
    """
    return py"seasonal_diffs"(y, seasonality)
end
