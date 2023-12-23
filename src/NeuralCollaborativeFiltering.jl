module NeuralCollaborativeFiltering

include("models/models_types.jl")

include("evaluation/metrics.jl")

include("evaluation/evaluate_model.jl")

include("models/dot_product_ncf.jl")
include("models/mlp_similarity_ncf.jl")
include("models/gmf_and_mlp_ncf.jl")

include("train_model.jl")

include("formatting.jl")

end
