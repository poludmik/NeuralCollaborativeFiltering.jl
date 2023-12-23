using NeuralCollaborativeFiltering
using Test

using DataFrames
using CSV
using Random
using Flux
using JLD2

@testset "NeuralCollaborativeFiltering.jl" begin

    ############## Recommendation quality metrics ##############

    @testset "reciprocal_rank" begin
        @test reciprocal_rank([5, 2, 1], [3, 5, 7]) ≈ 1/2
        @test reciprocal_rank([2, 6, 4], [1, 3, 5]) == 0.0
        @test reciprocal_rank([10], [10, 20, 30]) ≈ 1/1
        @test reciprocal_rank([8, 0, -1, 5, 4, 6, 7], [0, -1, 3, 8, 4, 5, 7]) == 0.25
        @test_throws MethodError reciprocal_rank([8, 0, -1, 5, 4.0, 6, 7], [0, -1, 3, 8, 4, 5, 7])  # y <: Vector{Float64}
    end

    @testset "accuracy" begin
        @test accuracy([1, 2, 3], [1, 2, 3]) == 1.0
        @test accuracy([1, 2, 3], [1, 4, 3]) ≈ 2/3
        @test accuracy([4, 5, 6], [7, 8, 9]) == 0.0
        @test accuracy([8, 0, -1, 5, 4, 6, 7], [0, -1, 3, 8, 4, 5, 7]) ≈ 0.2857142857142857
        @test_throws MethodError accuracy([8, 0, -1, 5, 4.0, 6, 7], [0, -1, 3, 8, 4, 5, 7])
    end

    @testset "average_precision" begin
        @test average_precision([1, 2, 3], [1, 2, 3]) == 1.0
        @test average_precision([2, 6, 4], [1, 3, 5]) == 0.0
        @test average_precision([4, 2, 1], [1, 2, 3]) ≈ 2/3
        @test average_precision([8, 0, -1, 5, 4, 6, 7], [0, -1, 3, 8, 4, 5, 7]) ≈ 0.7486394557823128
        @test_throws MethodError average_precision([8, 0, -1, 5, 4.0, 6, 7], [0, -1, 3, 8, 4, 5, 7])
    end

    @testset "extended_reciprocal_rank" begin
        @test extended_reciprocal_rank([1, 2, 4, 7], [7, 8, 1, 5]) ≈ 1/3
        @test extended_reciprocal_rank([1, 2, 4], [4, 2, 3]) ≈ 2/3
        @test extended_reciprocal_rank([10, 20, 4], [1, 2, 3]) == 0.0
        @test extended_reciprocal_rank([8, 0, -1, 5, 4, 6, 7], [0, -1, 3, 8, 4, 5, 7]) ≈ 0.6547619047619049
        @test_throws MethodError extended_reciprocal_rank([8, 0, -1, 5, 4.0, 6, 7], [0, -1, 3, 8, 4, 5, 7])
    end

    # Smoke tests on learning and model loading for 3 types of models
    @testset "learning" begin

        function create_folder_with_subfolders(main_folder::String, subfolders::Vector{String})
            println("isdir(main_folder1) = ", isdir(main_folder))
            if !isdir(main_folder)
                mkdir(main_folder)
            end
            println("isdir(main_folder2) = ", isdir(main_folder))
            for subfolder in subfolders
                if !isdir(joinpath(main_folder, subfolder))
                    mkdir(joinpath(main_folder, subfolder))
                end
            end
        end

        println(pwd())

        main_folder = "weights\\test"
        subfolders = ["dot_product_ncf", "mlp_similarity_ncf", "gmf_and_mlp_ncf"]
        create_folder_with_subfolders(main_folder, subfolders)
        main_folder = "plots\\test"
        create_folder_with_subfolders(main_folder, subfolders)

        @test isdir("weights\\test\\dot_product_ncf")
        @test isdir("weights\\test\\mlp_similarity_ncf")
        @test isdir("weights\\test\\gmf_and_mlp_ncf")
        @test isdir("plots\\test\\dot_product_ncf")
        @test isdir("plots\\test\\mlp_similarity_ncf")
        @test isdir("plots\\test\\gmf_and_mlp_ncf")

        for type in [DotProductModel, MLPSimilarityModel, GMFAndMLPModel]
            Random.seed!(228)

            path_train = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_train.csv"
            df_train = DataFrame(CSV.File(path_train))
            @test df_train |> typeof == DataFrame

            path_test = "datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_test.csv"
            df_test = DataFrame(CSV.File(path_test))
            @test df_test |> typeof == DataFrame

            model_type = type
            share_embeddings = true
            emb_size = 60

            m = build_model(model_type, df_train, df_test, embeddings_size=emb_size, share_embeddings=share_embeddings)
            @test m |> typeof == type

            weights_path, plot_path = train_model(df_train, df_test, m, n_epochs=5, lr=0.001, bs=1024, weights_folder="weights\\test\\", plots_folder="plots\\test\\")
            @test (weights_path |> typeof == String) && (plot_path |> typeof == String)

            filename = weights_path
            model_state = JLD2.load(filename, "model_state")
            @test Int(JLD2.load(filename, "emb_size")) == emb_size
            emb_size = Int(JLD2.load(filename, "emb_size"))

            model = build_model(model_type, df_train, df_test, embeddings_size=emb_size, share_embeddings=share_embeddings)
            @test m |> typeof == type

            model.emb_size = emb_size
            Flux.loadmodel!(model.model, model_state)

            res_one_user = evaluate_model_on_1_user(model, 1, df_test, top_n_mrr=5);

            @test :ExtRR in fieldnames(typeof(res_one_user))
            @test :RR in fieldnames(typeof(res_one_user))
            @test :AP in fieldnames(typeof(res_one_user))
            @test :ACC in fieldnames(typeof(res_one_user))
            @test res_one_user.ExtRR <= 1.0 && res_one_user.ExtRR >= 0.0
            @test res_one_user.RR <= 1.0 && res_one_user.RR >= 0.0
            @test res_one_user.AP <= 1.0 && res_one_user.AP >= 0.0
            @test res_one_user.ACC <= 1.0 && res_one_user.ACC >= 0.0

            res_all_users = evaluate_model(df_test, model);

            @test :MeanExtRR in fieldnames(typeof(res_all_users))
            @test :MRR in fieldnames(typeof(res_all_users))
            @test :MAP in fieldnames(typeof(res_all_users))
            @test :MeanACC in fieldnames(typeof(res_all_users))
            @test res_all_users.MeanExtRR <= 1.0 && res_all_users.MeanExtRR >= 0.0
            @test res_all_users.MRR <= 1.0 && res_all_users.MRR >= 0.0
            @test res_all_users.MAP <= 1.0 && res_all_users.MAP >= 0.0
            @test res_all_users.MeanACC <= 1.0 && res_all_users.MeanACC >= 0.0
        end

        # if isdir("..\\weights\\test\\")
        #     rm("..\\weights\\test\\", recursive=true)
        # else
        #     println("Directory does not exist: ..\\weights\\test\\")
        # end

        # if isdir("..\\plots\\test\\")
        #     rm("..\\plots\\test\\", recursive=true)
        # else
        #     println("Directory does not exist: ..\\plots\\test\\")
        # end

    end

end
