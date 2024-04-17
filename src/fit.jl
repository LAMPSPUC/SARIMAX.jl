"""
    hasFitMethods(modelType::Type{<:SarimaxModel}) -> Bool

Check if a given `SarimaxModel` type has the `fit!` method implemented.

# Arguments
- `modelType::Type{<:SarimaxModel}`: Type of the SARIMAX model to check.

# Returns
A boolean indicating whether the `fit!` method is implemented for the specified model type.

"""
function hasFitMethods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(fit!, tupleModelType)
end

"""
    hasHyperparametersMethods(modelType::Type{<:SarimaxModel}) -> Bool

Checks if a given `SarimaxModel` type has methods related to hyperparameters.

# Arguments
- `modelType::Type{<:SarimaxModel}`: Type of the SARIMAX model to check.

# Returns
A boolean indicating whether the hyperparameter-related methods are implemented for the specified model type.

"""
function hasHyperparametersMethods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(getHyperparametersNumber, tupleModelType)
end

"""
    aic(K::Int64, loglikeVal::Float64) -> Float64

Calculate the Akaike Information Criterion (AIC) for a given number of parameters and log-likelihood value.

# Arguments
- `K::Int64`: Number of parameters in the model.
- `loglikeVal::Float64`: Log-likelihood value of the model.

# Returns
The AIC value calculated using the formula: AIC = 2*K - 2*loglikeVal.

"""
function aic(K::Int64, loglikeVal::Float64)
    return  2*K - 2*loglikeVal
end

"""
    aicc(T::Int64, K::Int64, loglikeVal::Float64) -> Float64

Calculate the corrected Akaike Information Criterion (AICc) for a given number of observations, number of parameters, and log-likelihood value.

# Arguments
- `T::Int64`: Number of observations in the data.
- `K::Int64`: Number of parameters in the model.
- `loglikeVal::Float64`: Log-likelihood value of the model.

# Returns
The AICc value calculated using the formula: AICc = AIC(K, loglikeVal) + ((2*K*K + 2*K) / (T - K - 1)).

"""
function aicc(T::Int64, K::Int64, loglikeVal::Float64)
    return aic(K, loglikeVal) + ((2*K*K + 2*K) / (T - K - 1))
end

"""
    bic(T::Int64, K::Int64, loglikeVal::Float64) -> Float64

Calculate the Bayesian Information Criterion (BIC) for a given number of observations, number of parameters, and log-likelihood value.

# Arguments
- `T::Int64`: Number of observations in the data.
- `K::Int64`: Number of parameters in the model.
- `loglikeVal::Float64`: Log-likelihood value of the model.

# Returns
The BIC value calculated using the formula: BIC = log(T) * K - 2 * loglikeVal.

"""
function bic(T::Int64, K::Int64, loglikeVal::Float64)
    return log(T)*K - 2*loglikeVal
end

"""
    aic(model::SarimaxModel) -> Float64

Calculate the Akaike Information Criterion (AIC) for a SARIMAX model.

# Arguments
- `model::SarimaxModel`: The SARIMAX model for which AIC is calculated.

# Returns
The AIC value calculated using the number of parameters and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `getHyperparametersNumber` method is not implemented for the given model type.

"""
function aic(model::SarimaxModel)
    !hasHyperparametersMethods(typeof(model)) && throw(MissingMethodImplementation("getHyperparametersNumber"))
    K = getHyperparametersNumber(model)
    return aic(K, loglike(model))
end

"""
    aicc(model::SarimaxModel) -> Float64

Calculate the Corrected Akaike Information Criterion (AICc) for a SARIMAX model.

# Arguments
- `model::SarimaxModel`: The SARIMAX model for which AICc is calculated.

# Returns
The AICc value calculated using the number of parameters, sample size, and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `getHyperparametersNumber` method is not implemented for the given model type.

"""
function aicc(model::SarimaxModel)
    !hasHyperparametersMethods(typeof(model)) && throw(MissingMethodImplementation("getHyperparametersNumber"))
    K = getHyperparametersNumber(model)
    T = length(model.ϵ)
    return aicc(T, K, loglike(model))
end

"""
    bic(model::SarimaxModel) -> Float64

Calculate the Bayesian Information Criterion (BIC) for a SARIMAX model.

# Arguments
- `model::SarimaxModel`: The SARIMAX model for which BIC is calculated.

# Returns
The BIC value calculated using the number of parameters, sample size, and log-likelihood value of the model.

# Errors
- Throws a `MissingMethodImplementation` if the `getHyperparametersNumber` method is not implemented for the given model type.

"""
function bic(model::SarimaxModel)
    !hasHyperparametersMethods(typeof(model)) && throw(MissingMethodImplementation("getHyperparametersNumber"))
    K = getHyperparametersNumber(model)
    T = length(model.ϵ)
    return bic(T, K, loglike(model))
end