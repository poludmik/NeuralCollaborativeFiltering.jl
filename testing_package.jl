using NeuralCollaborativeFiltering
# using DataFrames
# using CSV
# using StatsBase
# using Random
# using Flux
# using Flux: params, throttle
# using Flux.Data: DataLoader
# using Flux.Losses: mse, logitcrossentropy
# using JLD2
# using LinearAlgebra
# using Plots
# using Dates

Random.seed!(228)

path = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_train.csv"
df_train = DataFrame(CSV.File(path))

path = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_test.csv"
df_test = DataFrame(CSV.File(path))

