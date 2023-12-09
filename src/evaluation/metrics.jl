function mean_reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real
    score = []
    for movie_id in y_vec
        rr = 0.0
        if movie_id in ŷ_vec
            rr = 1.0 / findfirst(x -> x == movie_id, ŷ_vec) # Julia has 1-based indexing, no need to do +1
        end
        push!(score, rr)
    end
    # mean([movie_id in ŷ_vec ? 1.0 / findfirst(x -> x == movie_id, ŷ_vec) : 0.0 for movie_id in y_vec])
    sum(score) / length(score)
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

export mean_reciprocal_rank, accuracy
