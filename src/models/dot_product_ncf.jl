using DataFrames
using CSV
using Flux
using Flux: params, throttle
using Flux.Data: DataLoader
using Flux.Losses: mse, logitcrossentropy
using JLD2
using LinearAlgebra

export build_model, DotProductModel, batched_dot_product

"""
    DotProductModel(df_train, df_test, embeddings_size, flux_model)

Recommender system based on Generalized Matrix Factorization: 2 Embedding layers with a dot product.
Sizes of both embedding layers depend on a number of movies and on a number of users respectively.

# Fields
- `df_train::DataFrame`: Training set to be learnt on.
- `df_test::DataFrame`: Testing set to be learnt on.
- `emb_size::Int64`: Size of a single embedding vector for one user or for one movie.
- `model::Chain`: Flux.jl Chain object representing the NN.
- `folder_name::String`: Assigned to 'dot_product_ncf' in the constructor automatically.

# References
- Source: [Figure 1 (left) from `Neural Collaborative Filtering vs. Matrix Factorization Revisited`](https://arxiv.org/pdf/2005.09683.pdf)
"""
mutable struct DotProductModel <: NCFModel
    df_train::DataFrame
    df_test::DataFrame
    emb_size::Int64
    model::Chain
    folder_name::String
end

DotProductModel(df_train::DataFrame, df_test::DataFrame, emb_size::Int64, model::Chain) = DotProductModel(df_train, df_test, emb_size, model, "dot_product_ncf")

Base.show(io::IO, m::DotProductModel) = println(io, "$(BLUE)DotProductModel$(RESET):\n    "*
                                                       "#Train instances: $(nrow(m.df_train))\n    "*
                                                       "#Test instances: $(nrow(m.df_test))\n    "*
                                                       "$(m.model)\n")


"""
    batched_dot_product(x, y)

Compute the batched dot product between two arrays `x` and `y`. Element-wise dot product.

# Arguments
- `x`: An array of any dimension. Represents a batch of vectors.
- `y`: An array with the same number of dimensions as `x`. Represents second batch of vectors.

# Returns
- An array of size (1, 1, batch_size) containing the dot products of corresponding vectors from `x` and `y`.

# Example
```julia
x = [2 2 2; 1 0 0]
y = [5 6 1 ; 7 8 8]
result = batched_dot_product(x, y)

# result after removing unit dimentions:
[17, 12, 2]
```
"""
function batched_dot_product(x, y)
    x_expanded = reshape(x, (1, size(x)...))
    y_expanded = reshape(y, (1, size(y)...))
    y_T = NNlib.batched_transpose(y_expanded)
    # println("size(x_expanded): ", size(x_expanded))
    # println("typeof(x_expanded): ", typeof(x_expanded))
    # println("size(y_T): ", size(y_T))
    # println("typeof(y_T): ", typeof(y_T))
    # println("")
    return NNlib.batched_mul(x_expanded, y_T) # Fails in Julia nightly for some reason
    # println(size(x_expanded .* y_T))
    # return x_expanded .* y_T
end


"""
    build_model(x::Type{DotProductModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50, share_embeddings=nothing)

Constructs and returns an instance of DotProductModel using training and testing data sizes along with specified model parameters.

# Arguments 
- `x::Type{DotProductModel}`: The type of the model to be created. Multiple dispatch allows to define `build_model` for every model type differently.
- `df_train::DataFrame`: The training data frame. Used to get the size of the embedding layers based on the numbers of movies and users.
- `df_test::DataFrame`: The testing data frame. Same as `df_train`.
- `embeddings_size::Int=50 (optional)`: The size of the embedding vectors for both users and movies.
- `share_embeddings::Any=nothing (optional)`: Redundant here. Used in `GMFAndMLPModel`.

# Returns
`DotProductModel`: An instance of DotProductModel with the initialized Flux Chain.

# Example
```julia
m = build_model(DotProductModel, df_train, df_test, embeddings_size=60)
```
"""
function build_model(x::Type{DotProductModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50, share_embeddings=nothing)
    println("Creating an object of type $(GREEN)$(x)$(RESET).")
    
    # Determining the sizes of embedding layers
    user_n = maximum(df_train[:, "user"])
    movie_n = maximum([maximum(df_train[:, "movie"]), maximum(df_test[:, "movie"])])

    # Initializing the embedding layers
    emb_init = randn32
    xusers_emb = Embedding(user_n => embeddings_size; init=emb_init)
    xproducts_emb = Embedding(movie_n => embeddings_size;  init=emb_init)

    # Constructing the model
    squeeeeze(x) = dropdims(dropdims(x, dims=1), dims=1)
    flux_model = Chain(
        Parallel(batched_dot_product, xusers_emb, xproducts_emb),
        squeeeeze,
        NNlib.sigmoid
        )
    
    return DotProductModel(df_train, df_test, embeddings_size, flux_model)
end
