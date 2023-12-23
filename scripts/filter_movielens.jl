"""
An ugly script to extract needed training and testing data from the movielens dataset.
Leaving code as is because it would always be different for different datasets.
Scaling the rankings from 1 to 5 stars to <0.1, 1> by MinMaxScale.
Splitting the train/test randomly.
Resulting dataset used in the NeuralCollaborativeFiltering training is of form:

user, movie, score
1   ,    47,   1.0
2   ,   256,   0.7
...

Also, I extracted the movie/genre pairs, which could potentially be used in the recommendation system that would consider those as additional features.
"""

using DataFrames
using CSV
using StatsBase
using Random

Random.seed!(228)

path = "..\\datasets\\ml-latest-small\\ratings.csv"
df_ratings = DataFrame(CSV.File(path))

path = "..\\datasets\\ml-latest-small\\movies.csv"
df_movies = DataFrame(CSV.File(path))
df_movies = sort!(df_movies, [:movieId])

max_user_id = maximum(df_ratings.userId)
max_movie_idx = length(df_movies.movieId)


function movie_id_to_col_idx(movie_id, df_movies)
    findfirst(df_movies.movieId .== movie_id)
end

function create_nan_dataframe(rows, cols)
    df = DataFrame()
    for col in 1:cols
        col_name = Symbol("$col")
        df[!, col_name] = fill(0.0, rows)
    end
    return df
end

user_movie_df = create_nan_dataframe(max_user_id, max_movie_idx)

function fill_user_movie_matrix(user_movie_df, df_ratings)
    for (r, row) in enumerate(eachrow(df_ratings))
        user_id = row.userId
        movie_id = row.movieId
        rating = row.rating
        user_movie_df[user_id, movie_id_to_col_idx(movie_id, df_movies)] = rating
    end
    user_movie_df
end

user_movie_df = fill_user_movie_matrix(user_movie_df, df_ratings)

function min_max_scale(df, min_val, max_val)
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    for (c, col) in enumerate(eachcol(df))
        min_col = minimum(col)
        max_col = maximum(col)
        if min_col == max_col # for division by 0
            max_col = 1
        end
        df[:,c] .= ((df[:,c] .- min_col) ./ (max_col - min_col)) .* (max_val - min_val) .+ min_val
    end
    df
end

# Scale values to <0.1, 1.0> with MinMaxScale
copy_for_zeros = copy(user_movie_df)
min_value, max_value = 0.1, 1.0
user_movie_df = min_max_scale(user_movie_df, min_value, max_value)

user_movie_pairs = DataFrame()
for (r, row) in enumerate(eachrow(copy_for_zeros))
    for (c, col) in enumerate(eachcol(copy_for_zeros))
        if copy_for_zeros[r, c] == 0.0 # In paper, they leave it to NaN
            user_movie_df[r, c] = 0.0
        else
            new_row = (user = r, movie = c, score = user_movie_df[r, c]) # Isn't 
            push!(user_movie_pairs, new_row)
        end
    end
end

function genre_to_idx(movie_df)
    curr_idx = 1
    genre2idx = Dict()
    for (r, row) in enumerate(eachrow(movie_df))
        words = split(row.genres, '|')
        for w in words
            if !haskey(genre2idx, w)
                genre2idx[w] = curr_idx
                curr_idx += 1
            end
        end
    end
    genre2idx
end

genre2idx = genre_to_idx(df_movies)

movie_genre_df = create_nan_dataframe(max_movie_idx, length(genre2idx))

function fill_movie_genre_matrix(movie_genre_df, movie_df, genre2idx)
    for (r, row) in enumerate(eachrow(movie_df))
        words = split(row.genres, '|')
        for w in words
            movie_genre_df[r, genre2idx[w]] = 1.0
        end
    end
    movie_genre_df
end

movie_genre_df = fill_movie_genre_matrix(movie_genre_df, df_movies, genre2idx)

random_test_idxs = sort(sample(1:max_movie_idx, 742, replace = false)) # 728 test instances of unseen movies
train_idxs = (1:max_movie_idx)[setdiff(1:max_movie_idx, random_test_idxs)] # remaining train instances

movie_genre_df_test = view(movie_genre_df, random_test_idxs, :)
user_movie_df_test = view(user_movie_df, :, random_test_idxs)

movie_genre_df_train = view(movie_genre_df, train_idxs, :)
user_movie_df_train = view(user_movie_df, :, train_idxs)

output_path = "..\\datasets\\ml-latest-small"
CSV.write(joinpath(output_path, "movie_genre_df_test.csv"), movie_genre_df_test)
CSV.write(joinpath(output_path, "user_movie_df_test.csv"), user_movie_df_test)
CSV.write(joinpath(output_path, "movie_genre_df_train.csv"), movie_genre_df_train)
CSV.write(joinpath(output_path, "user_movie_df_train.csv"), user_movie_df_train)


function collect_user_movie_pairs(df)
    res_pairs = DataFrame()
    for (r, row) in enumerate(eachrow(df))
        for (c, col) in enumerate(eachcol(df))
            if df[r, c] != 0.0
                new_row = (user = r, movie = parse(Int, names(df)[c]), score = df[r, c])
                push!(res_pairs, new_row)
            end
        end
    end
    res_pairs
end

user_movie_pairs_test = collect_user_movie_pairs(user_movie_df_test)
user_movie_pairs_train = collect_user_movie_pairs(user_movie_df_train)
CSV.write(joinpath(output_path, "user_movie_pairs_for_coll_filtr_test.csv"), user_movie_pairs_test)
CSV.write(joinpath(output_path, "user_movie_pairs_for_coll_filtr_train.csv"), user_movie_pairs_train)
