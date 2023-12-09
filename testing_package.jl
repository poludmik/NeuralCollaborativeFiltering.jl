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

m = build_model(df_train, df_test, embeddings_size=60)
weights_path, plot_path = train_model(df_train, df_test, m, n_epochs=21, lr=0.005, bs=1024)

# weights_path = "weights\\dot_product_ncf\\model_dim100_bs1024_ep21_lr0.01.jld2"
filename = weights_path
model_state = JLD2.load(filename, "model_state")
emb_size = Int(JLD2.load(filename, "emb_size"))
model = build_model(df_train, df_test, embeddings_size=emb_size)
model.emb_size = emb_size
Flux.loadmodel!(model.model, model_state)

evaluate_model_on_1_user(model, 1, df_test, top_n_mrr=5)
# evaluate_model(df_test, model)
