using DataFrames
using StatsBase
using Random
using LinearAlgebra

export evaluate_model_on_1_user, evaluate_model


"""
    get_all_reviews_from_one_user(user_id, data_df) 

Returns a DataFrame object that only contains a single user with user_id.

# Arguments
- `user_id`: Int.
- `data_df`: DataFrame object with a 'user' column.
"""
function get_all_reviews_from_one_user(user_id, data_df)
    filter(row -> row.user == user_id, data_df)
end


"""
    evaluate_model_on_1_user(m::T, user_id::Int, df_test::DataFrame; top_n_mrr=nothing) where T <: NCFModel

Predicts ranks for movies present in the test set of user with `user_id` and calculates 4 different metrics. 

# Arguments
- `m<:NCFModel`: The learned model.
- `user_id::Int`: Our user's id.
- `df_test::DataFrame`: The whole test set.
- `top_n_mrr`: Int or nothing. Number of top predictions to be considered. 

# Returns
- `NamedTuple` with fields: {ExtRR, RR, AP, ACC}, representing 4 different metrics.

# Example
```jldoctest
julia> evaluate_model_on_1_user(model, 1, df_test, top_n_mrr=5);
```
"""
function evaluate_model_on_1_user(m::T, user_id::Int, df_test::DataFrame; top_n_mrr=nothing) where T <: NCFModel
    df_test_one_user = get_all_reviews_from_one_user(user_id, df_test)
    println("Model evaluation on user $(YELLOW)$(user_id)$(RESET):")
    println("Total number of test cases for user $(user_id): $(BLUE)$(nrow(df_test_one_user))$(RESET)")
    if (top_n_mrr |> isnothing)
        top_n_mrr = nrow(df_test_one_user)
    end
    if (nrow(df_test_one_user) == 0)
        println("$(RED)Number of test cases is 0. Nothing to test for user $(user_id).$(RESET)")
        return
    end
    if (top_n_mrr > nrow(df_test_one_user))
        println("$(RED)Number of test cases for user $(user_id) is $(nrow(df_test_one_user)), but top_n_mrr=$(top_n_mrr) has to be smaller. Setting top_n_mrr to $(nrow(df_test_one_user))!$(RESET)")
        top_n_mrr = nrow(df_test_one_user)
    end
    x_test_1 = vec(Matrix(df_test_one_user[:, [:user]]))
    x_test_2 = vec(Matrix(df_test_one_user[:, [:movie]]))
    y_test = df_test_one_user.score
    ŷ_test = m.model((x_test_1, x_test_2))
    df_test_one_user.ŷ_score = vec(ŷ_test)

    sort!(df_test_one_user, [:ŷ_score], rev=true) # [:ŷ_score, :score]
    top_5_ŷ_movie_ids = df_test_one_user[1:top_n_mrr, :movie]
    all_ŷ_sorted = df_test_one_user[:, :movie]

    sort!(df_test_one_user, [:score], rev=true) # [:ŷ_score, :score]
    top_5_y_movie_ids = df_test_one_user[1:top_n_mrr, :movie]
    all_y_sorted = df_test_one_user[:, :movie]

    show_df = DataFrame()
    show_df.y = top_5_y_movie_ids
    show_df.ŷ = top_5_ŷ_movie_ids
    println("Top $(BLUE)$(top_n_mrr)$(RESET) predictions:")
    visualize_comparison(show_df)

    rr = reciprocal_rank(top_5_y_movie_ids, top_5_ŷ_movie_ids)
    acc = accuracy(top_5_y_movie_ids, top_5_ŷ_movie_ids)
    ap = average_precision(top_5_y_movie_ids, top_5_ŷ_movie_ids)
    ext_rr = extended_reciprocal_rank(top_5_y_movie_ids, top_5_ŷ_movie_ids)
    println("ExtRR: $(BLUE)$(round(ext_rr, digits=4))$(RESET), RR: $(BLUE)$(round(rr, digits=4))$(RESET), AP: $(BLUE)$(round(ap, digits=4))$(RESET), ACC: $(BLUE)$(round(acc, digits=4))$(RESET)")
    return (ExtRR=round(ext_rr, digits=4), RR=round(rr, digits=4), AP=round(ap, digits=4), ACC=round(acc, digits=4))
end


"""
evaluate_model(test_df, m::T; minimal_y_length=10, top_n_map=5) where T <: NCFModel

In contrast to `evaluate_model_on_1_user(...)`, calculates metrics on every valid user and averages them by the total number of valid users.

# Arguments
- `m<:NCFModel`: The learned model.
- `test_df`: The whole test set in a DataFrame.
- `minimal_y_length`: Minimum number of test instances for a user to be counted. E.g. if 10, then all users with the number of ranked movies under 10 will be skipped.
- `top_n_mrr`: Int or nothing. Number of top predictions to be considered.

# Returns
- `NamedTuple` with fields: {MeanExtRR, MRR, MAP, MeanACC}, representing 4 different metrics averaged by the total number of valid users.

# Example
```jldoctest
julia> evaluate_model(df_test, model);
```
"""
function evaluate_model(test_df, m::T; minimal_y_length=10, top_n_map=5) where T <: NCFModel # 
    rrs = []
    accs = []
    aps = []
    ext_rrs = []
    unique_values = unique(test_df.user)
    for user_id in unique_values
        df_test_one_user = get_all_reviews_from_one_user(user_id, test_df)

        x_test_1 = vec(Matrix(df_test_one_user[:, [:user]]))
        x_test_2 = vec(Matrix(df_test_one_user[:, [:movie]]))
        y_test = df_test_one_user.score

        if length(y_test) < minimal_y_length
            continue
        end

        ŷ_test = m.model((x_test_1, x_test_2))
        df_test_one_user.ŷ_score = vec(ŷ_test)

        sort!(df_test_one_user, [:ŷ_score], rev=true) # [:ŷ_score, :score]
        top_5_ŷ_movie_ids = df_test_one_user[1:top_n_map, :movie]
        all_ŷ_sorted = df_test_one_user[:, :movie]
        
        sort!(df_test_one_user, [:score], rev=true) # [:ŷ_score, :score]
        top_5_y_movie_ids = df_test_one_user[1:top_n_map, :movie]
        all_y_sorted = df_test_one_user[:, :movie]
        
        push!(rrs, reciprocal_rank(all_y_sorted, all_ŷ_sorted))
        push!(accs, accuracy(all_y_sorted, all_ŷ_sorted))
        push!(ext_rrs, extended_reciprocal_rank(all_y_sorted, all_ŷ_sorted))
        push!(aps, average_precision(top_5_y_movie_ids, top_5_ŷ_movie_ids))
    end
    println("\n$(YELLOW)Whole test set:$(RESET)")
    println("MeanExtRR: $(BLUE)$(round(mean(ext_rrs), digits=4))$(RESET), MRR: $(BLUE)$(round(mean(rrs), digits=4))$(RESET), MAP: $(BLUE)$(round(mean(aps), digits=4))$(RESET), MeanACC: $(BLUE)$(round(mean(accs), digits=4))$(RESET)\n")
    return (MeanExtRR=round(mean(ext_rrs), digits=4), MRR=round(mean(rrs), digits=4), MAP=round(mean(aps), digits=4), MeanACC=round(mean(accs), digits=4))
end

