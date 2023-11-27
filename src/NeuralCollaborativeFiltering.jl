module NeuralCollaborativeFiltering

include("models/models_types.jl")

include("evaluation/metrics.jl")

include("evaluation/evaluate_model.jl")

include("models/dot_product_ncf.jl") # needs to be before train_model.jl

include("train_model.jl")

include("formatting.jl")

end
