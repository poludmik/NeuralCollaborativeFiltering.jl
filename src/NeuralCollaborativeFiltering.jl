module NeuralCollaborativeFiltering

using Random

# Write your package code here.
include("train_model.jl")
include("models/dot_product_ncf.jl")
export train, a
export build_model

end
