

reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real  =  y_vec[1] in ŷ_vec ? 1.0 / findfirst(isequal(y_vec[1]), ŷ_vec) : 0.0


accuracy(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real  =  sum(y_vec[i] == ŷ_vec[i] for i in eachindex(y_vec)) / length(y_vec)


function average_precision(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real
    c = 0
    scores = [(movie_id in y_vec ? (c += 1) / d : 0.0) for (d, movie_id) in enumerate(ŷ_vec)]
    isempty(scores) ? 0.0 : sum(scores) / length(scores)
end


function extended_reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real
    """
    The idea was taken from https://towardsdatascience.com/extended-reciprocal-rank-ranking-evaluation-metric-5929573c778a

    Leaving the function in easy-to-read format, because the idea isn't well known.
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

export reciprocal_rank, accuracy, average_precision, extended_reciprocal_rank
