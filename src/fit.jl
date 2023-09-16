"""
    hasFitMethods(model_type::Type{<:StateSpaceModel}) -> Bool

Verify if a certain `SarimaxModel` has the `fit!` method.
"""
function hasFitMethods(modelType::Type{<:SarimaxModel})
    tupleModelType = Tuple{modelType}
    return hasmethod(fit!, tupleModelType)
end