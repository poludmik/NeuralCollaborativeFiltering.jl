using DataFrames
using CSV
using Flux
using Flux: params, throttle
using Flux.Losses: mse, logitcrossentropy
using JLD2
using LinearAlgebra

export build_model, MLPSimilarityModel

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
                                                       "$(m.model)")

function build_model(x::Type{MLPSimilarityModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50)
    println("Creating an object of type $(GREEN)$(x)$(RESET).")
    user_n = maximum(df_train[:, "user"])
    movie_n = maximum([maximum(df_train[:, "movie"]), maximum(df_test[:, "movie"])])
    emb_init = randn32
    xusers_emb = Embedding(user_n => embeddings_size; init=emb_init) # , init=Flux.identity_init(gain=22) ?
    xproducts_emb = Embedding(movie_n => embeddings_size;  init=emb_init)
    # expand_dim(y) = reshape(y, (1, size(y)...))
    squeeeeze(x) = dropdims(x, dims=1)
    flux_model = Chain(
        Parallel(vcat, xusers_emb, xproducts_emb),
        Dense(2*embeddings_size => 256),
        # NNlib.relu,
        Dense(256 => 1),
        NNlib.sigmoid,
        squeeeeze
        )
    return MLPSimilarityModel(df_train, df_test, embeddings_size, flux_model)
end
