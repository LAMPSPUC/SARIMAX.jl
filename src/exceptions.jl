"""
    struct ModelNotFitted <: Exception

An exception type that indicates the model has not been fitted yet.

# Usage
This exception can be thrown when an operation that requires a fitted model is attempted on an unfitted model.

"""
mutable struct ModelNotFitted <: Exception end
Base.showerror(io::IO, e::ModelNotFitted) =
    print(io, "The model has not been fitted yet. Please run fit!(model)")

"""
    MissingMethodImplementation <: Exception

Custom exception type for indicating that a required method is not implemented in the model.

# Fields
- `method::String`: The name of the method that is missing.

"""
mutable struct MissingMethodImplementation <: Exception
    method::String
end
Base.showerror(io::IO, e::MissingMethodImplementation) =
    print(io, "The model does not implement the ", e.method, " method.")

"""
    struct InconsistentDatePattern <: Exception

An exception type to indicate that the timestamps do not follow a consistent pattern.

"""
mutable struct InconsistentDatePattern <: Exception end
Base.showerror(io::IO, e::InconsistentDatePattern) =
    print(io, "The timestamps do not follow a consistent pattern.")

"""
    MissingExogenousData <: Exception

An exception type that indicates the absence of exogenous data required for forecasting the requested horizon.

"""
mutable struct MissingExogenousData <: Exception end
Base.showerror(io::IO, e::MissingExogenousData) =
    print(io, "There is no exogenous data to forecast the horizon requested")

"""
    InvalidParametersCombination <: Exception

A custom exception type to indicate that the combination of parameters provided
to the model is invalid.

# Fields
- `msg::String`: A message describing why the parameters are invalid.

"""
mutable struct InvalidParametersCombination <: Exception
    msg::String
end
Base.showerror(io::IO, e::InvalidParametersCombination) =
    print(io, "The parameters provided are invalid for the model \n", e.msg)
