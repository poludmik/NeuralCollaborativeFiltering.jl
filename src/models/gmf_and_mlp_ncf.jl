using DataFrames
using CSV
using Flux
using Flux: params, throttle
using Flux.Data: DataLoader
using Flux.Losses: mse, logitcrossentropy
using JLD2
using LinearAlgebra

export build_model, GMFAndMLPModel

# Recommender system based on a combination of Generalized Matrix Factorization and Multi Layer Perceptron similarity.
# Figure 3 from Neural Collaborative Filtering 2017: https://arxiv.org/pdf/1708.05031.pdf

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


function build_model(x::Type{GMFAndMLPModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50, share_embeddings=false)
    println("Creating an object of type $(GREEN)$(x)$(RESET).")
    user_n = maximum(df_train[:, "user"])
    movie_n = maximum([maximum(df_train[:, "movie"]), maximum(df_test[:, "movie"])])
    emb_init = randn32

    function lol(x)
        println(size(x))
        x
    end

    squeeeeze(x) = dropdims(x, dims=1)
    
    # Creating 4 or 2 embedding matrices (depends on share_embeddings parameter)
    xusers_emb = Embedding(user_n => embeddings_size; init=emb_init) # , init=Flux.identity_init(gain=22) ?
    xproducts_emb = Embedding(movie_n => embeddings_size;  init=emb_init)

    xusers_emb2 = Embedding(user_n => embeddings_size; init=emb_init) # , init=Flux.identity_init(gain=22) ?
    xproducts_emb2 = Embedding(movie_n => embeddings_size;  init=emb_init)

    # Dot product (Generalized Matrix Factorization)
    flux_model_dot = Chain(
                        Parallel(batched_dot_product, xusers_emb, xproducts_emb),
                        squeeeeze,
                        # NNlib.sigmoid,
                        )

    # Multi Layer Perceptron similarity
    flux_model_mlp = Chain(
                        Parallel(vcat, share_embeddings ? xusers_emb : xusers_emb2, share_embeddings ? xproducts_emb : xproducts_emb2),
                        Dense(2*embeddings_size => 512),
                        NNlib.leakyrelu,
                        Dense(512 => 25),
                        NNlib.leakyrelu
                        )

    # Merge both and pass to an another fully connected layer
    split(x) = (x, x)
    result_model = Chain(
                        split,
                        Parallel(vcat, flux_model_mlp, flux_model_dot),
                        Dense(26, 1),
                        NNlib.sigmoid,
                        squeeeeze
                        )

    return GMFAndMLPModel(df_train, df_test, embeddings_size, result_model)
end
