"""
    extended_reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real

Calculate the Extended Reciprocal Rank (ExtRR) between two vectors, `y_vec` and `ŷ_vec`.

# Arguments
- `y_vec::Vector{T}`: The ground truth vector (e.g., correct movie IDs).
- `ŷ_vec::Vector{T}`: The predicted vector (e.g., ranked movie IDs).

# Returns
- `Float64`: The calculated ExtRR.

# Example
```jldoctest
julia> extended_reciprocal_rank([3, 1, 4, 2], [1, 3, 2, 4])
0.75
```

# Note
Leaving the function in easy-to-read format, because it is easier to understand.

# References
- Source: [Towards Data Science](https://towardsdatascience.com/extended-reciprocal-rank-ranking-evaluation-metric-5929573c778a)
"""
function extended_reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real
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


"""
    reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real

Calculate the Reciprocal Rank (RR) for a given pair of vectors, `y_vec` and `ŷ_vec`. Used in MRR calculation in evaluate_model.jl.

# Arguments
- `y_vec::Vector{T}`: The ground truth vector (e.g., correct movie IDs).
- `ŷ_vec::Vector{T}`: The predicted vector (e.g., ranked movie IDs).

# Returns
- `Float64`: The calculated RR, which is the reciprocal of the rank at which the first relevant item (the first item of `y_vec`) appears in `ŷ_vec`.

# Example
```jldoctest
julia> reciprocal_rank([3, 1, 4, 2], [1, 3, 2, 4])
0.5
```
"""
reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real  =  y_vec[1] in ŷ_vec ? 1.0 / findfirst(isequal(y_vec[1]), ŷ_vec) : 0.0


"""
    accuracy(y_vec::Vector{T}, ŷ_vec::Vector{T}) -> Float64 where T <: Real

Calculate the accuracy between two vectors. This function computes the proportion of elements that match between the ground truth vector `y_vec` and the predicted vector `ŷ_vec`.

# Arguments
- `y_vec::Vector{T}`: The ground truth vector (e.g., correct movie IDs).
- `ŷ_vec::Vector{T}`: The predicted vector (e.g., ranked movie IDs).

# Returns
- `Float64`: The accuracy, calculated as the number of matching elements in `y_vec` and `ŷ_vec`, divided by the total number of elements.

# Examples
```jldoctest
julia> accuracy([1, 2, 3], [3, 2, 4])
0.3333333333333333
```
"""
accuracy(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real  =  sum(y_vec[i] == ŷ_vec[i] for i in eachindex(y_vec)) / length(y_vec)


"""
    average_precision(y_vec::Vector{T}, ŷ_vec::Vector{T}) -> Float64 where T <: Real

Calculate the average precision of a ranked list of items. This function compares the ranked list `ŷ_vec` against the ground truth `y_vec` to compute the average precision.

# Arguments
- `y_vec::Vector{T}`: A vector of ground truth items (e.g., correct movie IDs).
- `ŷ_vec::Vector{T}`: A vector of predicted items (e.g., ranked movie IDs).

# Returns
- `Float64`: The average precision calculated over the ranked list.

# Examples
```jldoctest
julia> average_precision([1, 3, 4], [4, 2, 3])
0.5555555555555555
```
"""
function average_precision(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real
    c = 0
    scores = [(movie_id in y_vec ? (c += 1) / d : 0.0) for (d, movie_id) in enumerate(ŷ_vec)]
    isempty(scores) ? 0.0 : sum(scores) / length(scores)
end


export reciprocal_rank, accuracy, average_precision, extended_reciprocal_rank
