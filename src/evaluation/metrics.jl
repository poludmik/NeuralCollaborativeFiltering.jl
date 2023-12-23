

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
    y_i_sum = 0.0
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
        y_i_sum += 1 / y_i
    end
    sum(y_i_sum) / length(y_vec)
end

export reciprocal_rank, accuracy, average_precision, extended_reciprocal_rank
