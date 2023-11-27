module NeuralCollaborativeFiltering

include("models/dot_product_ncf.jl") # needs to be before train_model.jl

include("train_model.jl")

include("formatting.jl")

end
