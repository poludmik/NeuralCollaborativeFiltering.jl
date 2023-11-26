using DataFrames
using CSV
# using StatsBase
using Random
# using Flux
# using Flux: params, throttle
# using Flux.Data: DataLoader
# using Flux.Losses: mse, logitcrossentropy
# using JLD2
# using LinearAlgebra
# using Plots
# using Dates

using NeuralCollaborativeFiltering

Random.seed!(228)

path_train = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_train.csv"
df_train = DataFrame(CSV.File(path_train))

path_test = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_test.csv"
df_test = DataFrame(CSV.File(path_test))

m = build_model(df_train, df_test, embeddings_size=50)

train_model(df_train, df_test, m, n_epochs=21, lr=0.01, bs=1024)
