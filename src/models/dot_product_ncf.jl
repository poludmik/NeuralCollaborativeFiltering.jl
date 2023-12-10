using DataFrames
using CSV
using Flux
using Flux: params, throttle
using Flux.Data: DataLoader
using Flux.Losses: mse, logitcrossentropy
using JLD2
using LinearAlgebra

export build_model, DotProductModel, batched_dot_product

# Recommender system based on Generalized Matrix Factorization: 2 Embedding layers with a dot product.

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

function batched_dot_product(x, y)
    x_expanded = reshape(x, (1, size(x)...))
    # x_T = NNlib.batched_transpose(x_expanded)
    # magnitude_x = sqrt.(squeeeeze(NNlib.batched_mul(x_expanded, x_T)))

    y_expanded = reshape(y, (1, size(y)...))
    y_T = NNlib.batched_transpose(y_expanded)
    # magnitude_y = sqrt.(squeeeeze(NNlib.batched_mul(y_expanded, y_T)))
    # println(size(squeeeeze(NNlib.batched_mul(x_expanded, y_T))))
    return NNlib.batched_mul(x_expanded, y_T) # ./ (magnitude_x .* magnitude_y)
end

function build_model(x::Type{DotProductModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50, share_embeddings=nothing)
    println("Creating an object of type $(GREEN)$(x)$(RESET).")
    user_n = maximum(df_train[:, "user"])
    movie_n = maximum([maximum(df_train[:, "movie"]), maximum(df_test[:, "movie"])])
    # emb_init = Flux.glorot_uniform(MersenneTwister(1))
    # emb_init = Flux.identity_init(gain=22)
    # emb_init = Flux.glorot_normal
    emb_init = randn32
    xusers_emb = Embedding(user_n => embeddings_size; init=emb_init) # , init=Flux.identity_init(gain=22) ?
    xproducts_emb = Embedding(movie_n => embeddings_size;  init=emb_init)
    squeeeeze(x) = dropdims(dropdims(x, dims=1), dims=1)
    flux_model = Chain(
        Parallel(batched_dot_product, xusers_emb, xproducts_emb),
        squeeeeze,
        NNlib.sigmoid
        )
    return DotProductModel(df_train, df_test, embeddings_size, flux_model)
end
