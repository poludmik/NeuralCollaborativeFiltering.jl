function reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real
    movie_id = y_vec[1]
    rr = 0.0
    if movie_id in ŷ_vec
        rr = 1.0 / (findfirst(x -> x == movie_id, ŷ_vec)) # Julia has 1-based indexing, no need to do +1
    end
    rr
end

function average_precision(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real
    score = []
    c = 0
    d = 0
    for (d, movie_id) in enumerate(ŷ_vec)
        pr = 0.0
        if movie_id in y_vec
            c += 1
            pr = c / d # Julia has 1-based indexing, no need to do +1
        end
        push!(score, pr)
    end
    if (length(score) == 0)
        return 0.0
    end
    sum(score) / length(score)
end

function extended_reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real
    """
    The idea was taken from https://towardsdatascience.com/extended-reciprocal-rank-ranking-evaluation-metric-5929573c778a
    """
    movie_id = y_vec[1]
    y_is = []
    for (mep_i, movie_id) in enumerate(y_vec)
        y_i = 0.0
        rank_i = findfirst(x -> x == movie_id, ŷ_vec)
        if (rank_i |> isnothing)
            continue
        end
        if (rank_i <= mep_i)
            y_i = 1.0
        else
            y_i = rank_i - mep_i + 1
        end
        push!(y_is, 1/y_i)
    end
    if (length(y_is) == 0)
        return 0.0
    end
    sum(y_is) / length(y_is)
end

function accuracy(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real
    num_same = 0
    for i in eachindex(y_vec)
        if y_vec[i] == ŷ_vec[i]
            num_same += 1
        end
    end
    num_same / length(y_vec)
    # count(==(t...), zip(y_vec, ŷ_vec)) / length(y_vec)
end

export reciprocal_rank, accuracy, average_precision, extended_reciprocal_rank
