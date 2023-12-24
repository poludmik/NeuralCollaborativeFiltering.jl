using DataFrames
using CSV
using Flux
using Flux: params, throttle
using Flux.Data: DataLoader
using Flux.Losses: mse, logitcrossentropy
using JLD2
using LinearAlgebra

export build_model, GMFAndMLPModel

"""
    GMFAndMLPModel(df_train::DataFrame, df_test::DataFrame, emb_size::Int64, model::Chain)

Recommender system based on a combination of Generalized Matrix Factorization and Multi Layer Perceptron similarity.
Sizes of both embedding layers depend on a number of movies and on a number of users respectively.

# Fields
- `df_train::DataFrame`: Training set to be learnt on.
- `df_test::DataFrame`: Testing set to be learnt on.
- `emb_size::Int64`: Size of a single embedding vector for one user or for one movie.
- `model::Chain`: Flux.jl Chain object representing the NN.
- `folder_name::String`: Assigned to `gmf_and_mlp_ncf` in the constructor automatically.

# References
- Source: [Figure 3 from Neural Collaborative Filtering 2017](https://arxiv.org/pdf/1708.05031.pdf)
"""
mutable struct GMFAndMLPModel <: NCFModel
    df_train::DataFrame
    df_test::DataFrame
    emb_size::Int64
    model::Chain
    folder_name::String
end

GMFAndMLPModel(df_train::DataFrame, df_test::DataFrame, emb_size::Int64, model::Chain) = GMFAndMLPModel(df_train, df_test, emb_size, model, "gmf_and_mlp_ncf")

Base.show(io::IO, m::GMFAndMLPModel) = println(io, "$(BLUE)GMFAndMLPModel$(RESET):\n    "*
                                                       "#Train instances: $(nrow(m.df_train))\n    "*
                                                       "#Test instances: $(nrow(m.df_test))\n    "*
                                                       "$(m.model)\n")


"""
    build_model(x::Type{GMFAndMLPModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50, share_embeddings=false)

Constructs and returns an instance of GMFAndMLPModel using training and testing data sizes along with specified model parameters.

# Arguments
- `x::Type{GMFAndMLPModel}`: The type of the model to be created. Multiple dispatch allows to define `build_model` for every model type differently.
- `df_train::DataFrame`: The training data frame. Used to get the size of the embedding layers based on the numbers of movies and users.
- `df_test::DataFrame`: The testing data frame. Same as `df_train`.
- `embeddings_size::Int=50 (optional)`: The size of the embedding vectors for both users and movies.
- `share_embeddings::Bool=false`: If set `true`, the embeddings will be shared between the GMF and MLP parts of the model. Otherwise, two separate sets of embeddings (4 Embedding layers in total).

# Returns
`GMFAndMLPModel`: An instance of GMFAndMLPModel with the initialized Flux Chain.

# Example
```julia
m = build_model(model_type, df_train, df_test, embeddings_size=60, share_embeddings=true)
```
"""
function build_model(x::Type{GMFAndMLPModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50, share_embeddings=false)
    println("Creating an object of type $(GREEN)$(x)$(RESET).")

    squeeeeze(x) = dropdims(x, dims=1)
    split(x) = (x, x)

    # Determining the sizes of embedding layers
    user_n = maximum(df_train[:, "user"])
    movie_n = maximum([maximum(df_train[:, "movie"]), maximum(df_test[:, "movie"])])
    emb_init = randn32
    
    # Creating 4 embedding matrices. Only 2 of them will later be used if `share_embeddings == true`.
    xusers_emb = Embedding(user_n => embeddings_size; init=emb_init)
    xproducts_emb = Embedding(movie_n => embeddings_size;  init=emb_init)
    xusers_emb2 = Embedding(user_n => embeddings_size; init=emb_init)
    xproducts_emb2 = Embedding(movie_n => embeddings_size;  init=emb_init)

    # Dot product (Generalized Matrix Factorization) part
    flux_model_dot = Chain(
                        Parallel(batched_dot_product, xusers_emb, xproducts_emb),
                        squeeeeze,
                        )

    # Multi Layer Perceptron similarity part
    flux_model_mlp = Chain(
                        Parallel(vcat, share_embeddings ? xusers_emb : xusers_emb2, share_embeddings ? xproducts_emb : xproducts_emb2),
                        Dense(2*embeddings_size => 512),
                        NNlib.leakyrelu,
                        Dense(512 => 25),
                        NNlib.leakyrelu
                        )

    # Merge both parts nd pass to an another fully connected layer
    result_model = Chain(
                        split, # Pass the inputs to both parts. I.e. ((in1, in2), (in1, in2))
                        Parallel(vcat, flux_model_mlp, flux_model_dot),
                        Dense(26, 1),
                        NNlib.sigmoid,
                        squeeeeze
                        )

    return GMFAndMLPModel(df_train, df_test, embeddings_size, result_model)
end
