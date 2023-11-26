module NeuralCollaborativeFiltering


# Write your package code here.
include("train_model.jl")
export train

include("models/dot_product_ncf.jl")
export build_model

include("formatting.jl")

end
