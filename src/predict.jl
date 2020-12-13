"""
predict(𝒱::FluxData, model)

#Description
Returns a tuple of (1) the model predictions for the variable associated with the object 𝒱
and (2) the truth data for the same variable.

#Arguments
- 𝒱: (FluxData) object containing the training data for the associates variable
- model: model returned by gp_model or nn_model function
"""
function predict(𝒱, model)
    predictions = (𝒱.unscale_fn(model(𝒱.training_data[i][1])) for i in 1:length(𝒱.training_data))
    targets = (𝒱.unscale_fn(𝒱.training_data[i][2]) for i in 1:length(𝒱.training_data))
    return (cat(predictions...,dims=2), cat(targets...,dims=2))
end
