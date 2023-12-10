using DataFrames
using CSV
# using StatsBase
using Random
using Flux
# using Flux: params, throttle
# using Flux.Data: DataLoader
# using Flux.Losses: mse, logitcrossentropy
using JLD2
# using LinearAlgebra
# using Plots
# using Dates

using NeuralCollaborativeFiltering

Random.seed!(228)

path_train = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_train.csv"
df_train = DataFrame(CSV.File(path_train))

path_test = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_test.csv"
df_test = DataFrame(CSV.File(path_test))

# model_type = DotProductModel
# weights_path = "weights\\dot_product_ncf\\model_dim60_bs1024_ep1_lr1.0e-6.jld2"

model_type = MLPSimilarityModel
weights_path = "weights\\mlp_similarity_ncf\\model_dim60_bs1024_ep101_lr0.005.jld2"

# m = build_model(model_type, df_train, df_test, embeddings_size=60)
# weights_path, plot_path = train_model(df_train, df_test, m, n_epochs=1, lr=0.000001, bs=1024)

filename = weights_path
model_state = JLD2.load(filename, "model_state")
emb_size = Int(JLD2.load(filename, "emb_size"))
model = build_model(model_type, df_train, df_test, embeddings_size=emb_size)
model.emb_size = emb_size
Flux.loadmodel!(model.model, model_state)

evaluate_model_on_1_user(model, 1, df_test, top_n_mrr=5)
evaluate_model(df_test, model)
