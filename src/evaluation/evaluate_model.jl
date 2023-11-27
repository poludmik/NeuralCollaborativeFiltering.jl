using DataFrames
using CSV
using StatsBase
using Random
using Flux
using Flux: params, throttle
using Flux.Data: DataLoader
using Flux.Losses: mse
using JLD2
using LinearAlgebra
using Plots
using Dates

# Random.seed!(228)

# path = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_train.csv"
# df_train = DataFrame(CSV.File(path))
# path = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_test.csv"
# df_test = DataFrame(CSV.File(path))


# emb_size = 50
# filename = "weights\\model_sig_dim50_bs512_ep102_lr0.015.jld2"
# model_state = JLD2.load(filename, "model_state");
# # @show JLD2.load(filename, "test_loss");
# model = build_model(df_train, df_test, emb_size)
# Flux.loadmodel!(model, model_state)

export evaluate_model_on_1_user, evaluate_model

function get_all_reviews_from_one_user(user_id, data_df)
    filter(row -> row.user == user_id, data_df)
end

function evaluate_model_on_1_user(m::T, user_id::Int, df_test::DataFrame; top_n_mrr=5) where T <: NCFModel
    df_test_one_user = get_all_reviews_from_one_user(user_id, df_test)
    println("Model evaluation on user $(YELLOW)$(user_id)$(RESET):")
    println("Total number of test cases for user $(user_id): $(BLUE)$(nrow(df_test_one_user))$(RESET)")
    x_test_1 = vec(Matrix(df_test_one_user[:, [:user]]))
    x_test_2 = vec(Matrix(df_test_one_user[:, [:movie]]))
    y_test = df_test_one_user.score
    ŷ_test = m.model((x_test_1, x_test_2))
    df_test_one_user.ŷ_score = ŷ_test

    sort!(df_test_one_user, [:ŷ_score], rev=true) # [:ŷ_score, :score]
    top_5_ŷ_movie_ids = df_test_one_user[1:top_n_mrr, :movie]

    sort!(df_test_one_user, [:score], rev=true) # [:ŷ_score, :score]
    top_5_y_movie_ids = df_test_one_user[1:top_n_mrr, :movie]

    show_df = DataFrame()
    show_df.y = top_5_y_movie_ids
    show_df.ŷ = top_5_ŷ_movie_ids
    println("Top $(BLUE)$(top_n_mrr)$(RESET) predictions:\n", show_df)

    mrr = mean_reciprocal_rank(top_5_y_movie_ids, top_5_ŷ_movie_ids)
    acc = accuracy(top_5_y_movie_ids, top_5_ŷ_movie_ids)
    println("MRR: $(BLUE)$(round(mrr, digits=4))$(RESET), ACC: $(BLUE)$(round(acc, digits=4))$(RESET)")
end

function evaluate_model(test_df, m::T) where T <: NCFModel # calculate metrics for all test instances (every user: mrr, acc)
    mrrs = []
    accs = []
    unique_values = unique(test_df.user)
    for user_id in unique_values
        df_test_one_user = get_all_reviews_from_one_user(user_id, test_df)

        x_test_1 = vec(Matrix(df_test_one_user[:, [:user]]))
        x_test_2 = vec(Matrix(df_test_one_user[:, [:movie]]))
        y_test = df_test_one_user.score

        if length(y_test) < 10
            continue
        end

        ŷ_test = m.model((x_test_1, x_test_2))
        df_test_one_user.ŷ_score = ŷ_test

        # top_n = max(convert(Int, round(length(y_test) * 0.4)), 1)
        top_n = 5

        # println(top_n)

        sort!(df_test_one_user, [:ŷ_score], rev=true) # [:ŷ_score, :score]
        top_5_ŷ_movie_ids = df_test_one_user[1:top_n, :movie]
        all_ŷ_sorted = df_test_one_user[:, :movie]
        
        sort!(df_test_one_user, [:score], rev=true) # [:ŷ_score, :score]
        top_5_y_movie_ids = df_test_one_user[1:top_n, :movie]
        all_y_sorted = df_test_one_user[:, :movie]
        
        push!(mrrs, mean_reciprocal_rank(top_5_y_movie_ids, top_5_ŷ_movie_ids))
        push!(accs, accuracy(all_y_sorted, all_ŷ_sorted))
    end
    return (MRR=round(mean(mrrs), digits=4), ACC=round(mean(accs), digits=4))
end

