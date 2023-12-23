using DataFrames
using CSV
using StatsBase
using Random
using Flux
using Flux: params, throttle
using Flux.Data: DataLoader
using Flux.Losses: mse, logitcrossentropy
using JLD2
using LinearAlgebra
using Plots
using Dates

function train_model(df_train::DataFrame, df_test::DataFrame, m::Union{DotProductModel, MLPSimilarityModel, GMFAndMLPModel};
                    bs=512,
                    lr = 0.015,
                    n_epochs = 102,
                    weights_folder = "..\\weights\\",
                    plots_folder = "..\\plots\\"
                    )

    x_train_1 = vec(Matrix(df_train[:, [:user]]))
    x_train_2 = vec(Matrix(df_train[:, [:movie]]))
    y_train = df_train.score

    x_test_1 = vec(Matrix(df_test[:, [:user]]))
    x_test_2 = vec(Matrix(df_test[:, [:movie]]))
    y_test = df_test.score

    train_data = DataLoader((x_train_1, x_train_2, y_train), batchsize=bs, shuffle=true)
    test_data = DataLoader((x_test_1, x_test_2, y_test), batchsize=bs)

    println("Dataloader 'train_data' with length = ", length(train_data))
    println("Dataloader 'test_data' with length = ", length(test_data))

    for (i, x) in enumerate(test_data) # TODO: unittest
        # @show i
        @assert length(x) == 3
        @assert length(x[1]) == bs || i == length(test_data)
    end

    loss(x1, x2, y) = mse(m.model((x1, x2)), y)
    # loss(x1, x2, y) = mse(m.model((x1, x2)), y) + 0.001 * sum(norm, params(m.model)) / length(x_train_1) # doesn't go to 0.5 for at least 100 epochs

    # L2reg(m) = sum(sum(p.^2) for p in Flux.params(m))
    # loss(x1, x2, y) = mse(m.model((x1, x2)), y) + L2reg(m.model)

    # sqnorm(x) = sum(abs2, x)
    # loss(x1, x2, y) = mse(m.model((x1, x2)), y) + sum(norm, Flux.params(m.model))

    function loss_all(dataloader, model)
        testmode!(model)
        l = 0f0
        for (x1, x2, y) in dataloader
            l += loss(x1, x2, y)
        end
        l/length(dataloader)
    end

    @show loss_all(test_data, m.model)

    # evalcb = () -> @show(loss_all(test_data, m.model)) # callback for Flux.train!()

    eval_data = (dataloader) -> round(loss_all(dataloader, m.model), digits=4)

    opt = Adam(lr)

    losses_train = []
    losses_test = []
    x_epochs = []
    for epoch in 1:n_epochs
        # @info "Epoch $epoch"
        trainmode!(m.model, true)
        Flux.train!(loss, params(m.model), train_data, opt) # , cb = evalcb
        if epoch % 10 == 0
            train_loss = eval_data(train_data)
            test_loss = eval_data(test_data)
            println("Ep: $(epoch), train: $(train_loss), test: $(test_loss)")
            push!(losses_train, train_loss)
            push!(losses_test, test_loss)
            push!(x_epochs, epoch)
        end
    end
    weights_path = weights_folder * "$(m.folder_name)\\model_dim$(m.emb_size)_bs$(bs)_ep$(n_epochs)_lr$(lr).jld2"
    plot_path = plots_folder * "$(m.folder_name)\\model_dim$(m.emb_size)_bs$(bs)_ep$(n_epochs)_lr$(lr).png"
    jldsave(weights_path, 
            model_state = Flux.state(m.model), 
            test_loss = loss_all(test_data, m.model),
            trained_on_df = df_train,
            tested_on_df = df_test,
            emb_size = m.emb_size
            )

    plot(x_epochs, [losses_train, losses_test], label=["train" "test"], xlabel="Epoch", ylabel="Loss", title="MSE losses with lr=$(lr), d=$(m.emb_size)")
    savefig(plot_path)
    println("$(GREEN)Training was finished, weights and the plot were saved!$(RESET)")
    return (weights_path, plot_path)
end


export train_model
