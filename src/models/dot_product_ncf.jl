
function squeeeeze(x)
    return dropdims(dropdims(x, dims=1), dims=1)
end

function third_dot(x, y)
    # println(size(x))
    # println(size(reshape(x, (1, size(x)...))))
    # println("\n$(size(x))")
    x_expanded = reshape(x, (1, size(x)...))
    x_T = NNlib.batched_transpose(x_expanded)
    # magnitude_x = sqrt.(squeeeeze(NNlib.batched_mul(x_expanded, x_T)))
    # println(size(magnitude_x))

    y_expanded = reshape(y, (1, size(y)...))
    y_T = NNlib.batched_transpose(y_expanded)
    # magnitude_y = sqrt.(squeeeeze(NNlib.batched_mul(y_expanded, y_T)))
    # println(size(squeeeeze(NNlib.batched_mul(x_expanded, y_T))))

    return squeeeeze(NNlib.batched_mul(x_expanded, y_T)) # ./ (magnitude_x .* magnitude_y)
end

function build_model(df_train, df_test, embeddings_size=50)
    user_n = maximum(df_train[:, "user"])
    movie_n = maximum([maximum(df_train[:, "movie"]), maximum(df_test[:, "movie"])])
    # emb_init = Flux.glorot_uniform(MersenneTwister(1))
    # emb_init = Flux.identity_init(gain=22) 
    # emb_init = Flux.glorot_normal
    emb_init = randn32
    xusers_emb = Embedding(user_n => embeddings_size; init=emb_init) # , init=Flux.identity_init(gain=22) ?
    xproducts_emb = Embedding(movie_n => embeddings_size;  init=emb_init)
    # out = Dense(1, 1)
    # NNlib.batched_mul
    return Chain(
        Parallel(third_dot, xusers_emb, xproducts_emb), 
        # squeeeeze,
        NNlib.sigmoid
        )
end

export build_model