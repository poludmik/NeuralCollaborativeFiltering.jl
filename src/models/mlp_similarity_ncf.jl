using DataFrames
using CSV
using Flux
using Flux: params, throttle
using Flux.Losses: mse, logitcrossentropy
using JLD2
using LinearAlgebra

export build_model, MLPSimilarityModel


"""
    MLPSimilarityModel(df_train::DataFrame, df_test::DataFrame, emb_size::Int64, model::Chain) 

Recommender system based on Multi Layer Perceptron: concatenate 2 Embedding layers and pass the result to the MLP.
Sizes of both embedding layers depend on a number of movies and on a number of users respectively.

# Fields
- `df_train::DataFrame`: Training set to be learnt on.
- `df_test::DataFrame`: Testing set to be learnt on.
- `emb_size::Int64`: Size of a single embedding vector for one user or for one movie.
- `model::Chain`: Flux.jl Chain object representing the NN.
- `folder_name::String`: Assigned to `mlp_similarity_ncf` in the constructor automatically.

# References
- Source: [Figure 1 (right) from `Neural Collaborative Filtering vs. Matrix Factorization Revisited`](https://arxiv.org/pdf/2005.09683.pdf)
"""
mutable struct MLPSimilarityModel <: NCFModel
    df_train::DataFrame
    df_test::DataFrame
    emb_size::Int64
    model::Chain
    folder_name::String
end

MLPSimilarityModel(df_train::DataFrame, df_test::DataFrame, emb_size::Int64, model::Chain) = MLPSimilarityModel(df_train, df_test, emb_size, model, "mlp_similarity_ncf")

Base.show(io::IO, m::MLPSimilarityModel) = println(io, "$(BLUE)MLPSimilarityModel$(RESET):\n    "*
                                                       "#Train instances: $(nrow(m.df_train))\n    "*
                                                       "#Test instances: $(nrow(m.df_test))\n    "*
                                                       "$(m.model)\n")


"""
    build_model(x::Type{MLPSimilarityModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50)

Constructs and returns an instance of MLPSimilarityModel using training and testing data sizes along with specified model parameters.

# Arguments
- `x::Type{MLPSimilarityModel}`: The type of the model to be created. Multiple dispatch allows to define `build_model` for every model type differently.
- `df_train::DataFrame`: The training data frame. Used to get the size of the embedding layers based on the numbers of movies and users.
- `df_test::DataFrame`: The testing data frame. Same as `df_train`.
- `embeddings_size::Int=50 (optional)`: The size of the embedding vectors for both users and movies.
- `share_embeddings::Any=nothing (optional)`: Redundant here. Used in `GMFAndMLPModel`.

# Returns
`MLPSimilarityModel`: An instance of MLPSimilarityModel with the initialized Flux Chain.

# Example
```julia
m = build_model(MLPSimilarityModel, df_train, df_test, embeddings_size=60)
```
"""
function build_model(x::Type{MLPSimilarityModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50, share_embeddings=nothing)
    println("Creating an object of type $(GREEN)$(x)$(RESET).")

    # Determining the sizes of embedding layers
    user_n = maximum(df_train[:, "user"])
    movie_n = maximum([maximum(df_train[:, "movie"]), maximum(df_test[:, "movie"])])

    # Initializing the embedding layers
    emb_init = randn32
    xusers_emb = Embedding(user_n => embeddings_size; init=emb_init)
    xproducts_emb = Embedding(movie_n => embeddings_size;  init=emb_init)

    # Constructing the model with a MLP at the end
    squeeeeze(x) = dropdims(x, dims=1)
    flux_model = Chain(
        Parallel(vcat, xusers_emb, xproducts_emb),
        Dense(2*embeddings_size => 512),
        NNlib.leakyrelu,
        Dense(512 => 1),
        NNlib.sigmoid,
        squeeeeze
        )
    
    return MLPSimilarityModel(df_train, df_test, embeddings_size, flux_model)
end
