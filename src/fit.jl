"""
    hasFitMethods(
        model_type::Type{<:StateSpaceModel}
    ) -> Bool

Verify if a certain `SarimaxModel` has the `fit!` method.
"""
function hasFitMethods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(fit!, tupleModelType)
end

"""
    hasHyperparametersMethods(
        model_type::Type{<:StateSpaceModel}
    ) -> Bool

Verify if a certain `SarimaxModel` has the methods related to the hyperparameters.
"""
function hasHyperparametersMethods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(getHyperparametersNumber, tupleModelType)
end

function aic(K::Int64, loglikeVal::Float64)
    return  2*K - 2*loglikeVal
end

function aicc(T::Int64, K::Int64, loglikeVal::Float64)
    return aic(K, loglikeVal) + ((2*K*K + 2*K) / (T - K - 1))
end

function bic(T::Int64, K::Int64, loglikeVal::Float64)
    return log(T)*K - 2*loglikeVal
end

function aic(model::SarimaxModel)
    !hasHyperparametersMethods(typeof(model)) && throw(MissingMethodImplementation("getHyperparametersNumber"))
    K = getHyperparametersNumber(model)
    return aic(K, loglike(model))
end

function aicc(model::SarimaxModel)
    !hasHyperparametersMethods(typeof(model)) && throw(MissingMethodImplementation("getHyperparametersNumber"))
    K = getHyperparametersNumber(model)
    T = length(model.ϵ)
    return aicc(T, K, loglike(model))
end

function bic(model::SarimaxModel)
    !hasHyperparametersMethods(typeof(model)) && throw(MissingMethodImplementation("getHyperparametersNumber"))
    K = getHyperparametersNumber(model)
    T = length(model.ϵ)
    return bic(T, K, loglike(model))
end