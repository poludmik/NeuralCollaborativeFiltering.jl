var documenterSearchIndex = {"docs":
[{"location":"eval/#Model-evaluation","page":"Model evaluation","title":"Model evaluation","text":"","category":"section"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"","category":"page"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"Metrics for model evaluation are defined in src/evaluation/ folder. ","category":"page"},{"location":"eval/#Metrics","page":"Model evaluation","title":"Metrics","text":"","category":"section"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"File metrics.jl defines metrics to compare the quality of the vector of recommended items relative to the ground truth vector (see README):","category":"page"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"accuracy","category":"page"},{"location":"eval/#NeuralCollaborativeFiltering.accuracy","page":"Model evaluation","title":"NeuralCollaborativeFiltering.accuracy","text":"accuracy(y_vec::Vector{T}, ŷ_vec::Vector{T}) -> Float64 where T <: Real\n\nCalculate the accuracy between two vectors. This function computes the proportion of elements that match between the ground truth vector y_vec and the predicted vector ŷ_vec.\n\nArguments\n\ny_vec::Vector{T}: The ground truth vector (e.g., correct movie IDs).\nŷ_vec::Vector{T}: The predicted vector (e.g., ranked movie IDs).\n\nReturns\n\nFloat64: The accuracy, calculated as the number of matching elements in y_vec and ŷ_vec, divided by the total number of elements.\n\nExamples\n\njulia> accuracy([1, 2, 3], [3, 2, 4])\n0.3333333333333333\n\n\n\n\n\n","category":"function"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"average_precision","category":"page"},{"location":"eval/#NeuralCollaborativeFiltering.average_precision","page":"Model evaluation","title":"NeuralCollaborativeFiltering.average_precision","text":"average_precision(y_vec::Vector{T}, ŷ_vec::Vector{T}) -> Float64 where T <: Real\n\nCalculate the average precision of a ranked list of items. This function compares the ranked list ŷ_vec against the ground truth y_vec to compute the average precision.\n\nArguments\n\ny_vec::Vector{T}: A vector of ground truth items (e.g., correct movie IDs).\nŷ_vec::Vector{T}: A vector of predicted items (e.g., ranked movie IDs).\n\nReturns\n\nFloat64: The average precision calculated over the ranked list.\n\nExamples\n\njulia> average_precision([1, 3, 4], [4, 2, 3])\n0.5555555555555555\n\n\n\n\n\n","category":"function"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"reciprocal_rank","category":"page"},{"location":"eval/#NeuralCollaborativeFiltering.reciprocal_rank","page":"Model evaluation","title":"NeuralCollaborativeFiltering.reciprocal_rank","text":"reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real\n\nCalculate the Reciprocal Rank (RR) for a given pair of vectors, y_vec and ŷ_vec. Used in MRR calculation in evaluate_model.jl.\n\nArguments\n\ny_vec::Vector{T}: The ground truth vector (e.g., correct movie IDs).\nŷ_vec::Vector{T}: The predicted vector (e.g., ranked movie IDs).\n\nReturns\n\nFloat64: The calculated RR, which is the reciprocal of the rank at which the first relevant item (the first item of y_vec) appears in ŷ_vec.\n\nExample\n\njulia> reciprocal_rank([3, 1, 4, 2], [1, 3, 2, 4])\n0.5\n\n\n\n\n\n","category":"function"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"extended_reciprocal_rank","category":"page"},{"location":"eval/#NeuralCollaborativeFiltering.extended_reciprocal_rank","page":"Model evaluation","title":"NeuralCollaborativeFiltering.extended_reciprocal_rank","text":"extended_reciprocal_rank(y_vec::Vector{T}, ŷ_vec::Vector{T}) where T <: Real\n\nCalculate the Extended Reciprocal Rank (ExtRR) between two vectors, y_vec and ŷ_vec.\n\nArguments\n\ny_vec::Vector{T}: The ground truth vector (e.g., correct movie IDs).\nŷ_vec::Vector{T}: The predicted vector (e.g., ranked movie IDs).\n\nReturns\n\nFloat64: The calculated ExtRR.\n\nExample\n\njulia> extended_reciprocal_rank([3, 1, 4, 2], [1, 3, 2, 4])\n0.75\n\nNote\n\nLeaving the function in easy-to-read format, because it is easier to understand.\n\nReferences\n\nSource: Towards Data Science\n\n\n\n\n\n","category":"function"},{"location":"eval/#User-wise-evaluation","page":"Model evaluation","title":"User-wise evaluation","text":"","category":"section"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"Then, the evaluate_model.jl utilizes the above metrics to evaluate the model on a specific user or on all the users available in the provided dataset.","category":"page"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"evaluate_model_on_1_user","category":"page"},{"location":"eval/#NeuralCollaborativeFiltering.evaluate_model_on_1_user","page":"Model evaluation","title":"NeuralCollaborativeFiltering.evaluate_model_on_1_user","text":"evaluate_model_on_1_user(m::T, user_id::Int, df_test::DataFrame; top_n_mrr=nothing) where T <: NCFModel\n\nPredicts ranks for movies present in the test set of user with user_id and calculates 4 different metrics. \n\nArguments\n\nm<:NCFModel: The learned model.\nuser_id::Int: Our user's id.\ndf_test::DataFrame: The whole test set.\ntop_n_mrr: Int or nothing. Number of top predictions to be considered. \n\nReturns\n\nNamedTuple with fields: {ExtRR, RR, AP, ACC}, representing 4 different metrics.\n\nExample\n\njulia> evaluate_model_on_1_user(model, 1, df_test, top_n_mrr=5);\n\n\n\n\n\n","category":"function"},{"location":"eval/","page":"Model evaluation","title":"Model evaluation","text":"evaluate_model","category":"page"},{"location":"eval/#NeuralCollaborativeFiltering.evaluate_model","page":"Model evaluation","title":"NeuralCollaborativeFiltering.evaluate_model","text":"evaluatemodel(testdf, m::T; minimalylength=10, topnmap=5) where T <: NCFModel\n\nIn contrast to evaluate_model_on_1_user(...), calculates metrics on every valid user and averages them by the total number of valid users.\n\nArguments\n\nm<:NCFModel: The learned model.\ntest_df: The whole test set in a DataFrame.\nminimal_y_length: Minimum number of test instances for a user to be counted. E.g. if 10, then all users with the number of ranked movies under 10 will be skipped.\ntop_n_mrr: Int or nothing. Number of top predictions to be considered.\n\nReturns\n\nNamedTuple with fields: {MeanExtRR, MRR, MAP, MeanACC}, representing 4 different metrics averaged by the total number of valid users.\n\nExample\n\njulia> evaluate_model(df_test, model);\n\n\n\n\n\n","category":"function"},{"location":"scripts/#Scripts","page":"Scripts","title":"Scripts","text":"","category":"section"},{"location":"scripts/#testing_package.jl","page":"Scripts","title":"testing_package.jl","text":"","category":"section"},{"location":"scripts/","page":"Scripts","title":"Scripts","text":"Contains all the basic functionality of the package:","category":"page"},{"location":"scripts/","page":"Scripts","title":"Scripts","text":"Loading the dataset\nLoading the weights or training a new model\nEvaluating the model on the provided test set.","category":"page"},{"location":"scripts/#filter_movielens.jl","page":"Scripts","title":"filter_movielens.jl","text":"","category":"section"},{"location":"scripts/","page":"Scripts","title":"Scripts","text":"An ugly script to extract needed training and testing data from the movielens dataset. Leaving code as it is because it would always be different for different datasets.","category":"page"},{"location":"scripts/","page":"Scripts","title":"Scripts","text":"Scaled the rankings from 1 to 5 stars to <0.1, 1> by MinMaxScale.","category":"page"},{"location":"scripts/","page":"Scripts","title":"Scripts","text":"Splitted data to the train/test randomly.","category":"page"},{"location":"scripts/","page":"Scripts","title":"Scripts","text":"Resulting dataset that is used in the NeuralCollaborativeFiltering training is of form:","category":"page"},{"location":"scripts/","page":"Scripts","title":"Scripts","text":"user movie score\n1 47 1.0\n2 256 0.7","category":"page"},{"location":"scripts/","page":"Scripts","title":"Scripts","text":"...","category":"page"},{"location":"scripts/","page":"Scripts","title":"Scripts","text":"Also, I extracted the movie/genre pairs, which could potentially be used in the recommendation system that would consider those as additional features (see README).","category":"page"},{"location":"models/#NN-Models","page":"NN Models","title":"NN Models","text":"","category":"section"},{"location":"models/","page":"NN Models","title":"NN Models","text":"Models are in the src/models/ folder of the repository. For each model, there is a struct that is a subtype of NCFModel. The constructor ensures the proper folder_name member parameter setting.","category":"page"},{"location":"models/#Generalized-Matrix-Factorization","page":"NN Models","title":"Generalized Matrix Factorization","text":"","category":"section"},{"location":"models/","page":"NN Models","title":"NN Models","text":"DotProductModel","category":"page"},{"location":"models/#NeuralCollaborativeFiltering.DotProductModel","page":"NN Models","title":"NeuralCollaborativeFiltering.DotProductModel","text":"DotProductModel(df_train, df_test, embeddings_size, flux_model)\n\nRecommender system based on Generalized Matrix Factorization: 2 Embedding layers with a dot product. Sizes of both embedding layers depend on a number of movies and on a number of users respectively.\n\nFields\n\ndf_train::DataFrame: Training set to be learnt on.\ndf_test::DataFrame: Testing set to be learnt on.\nemb_size::Int64: Size of a single embedding vector for one user or for one movie.\nmodel::Chain: Flux.jl Chain object representing the NN.\nfolder_name::String: Assigned to dot_product_ncf in the constructor automatically.\n\nReferences\n\nSource: Figure 1 (left) from Neural Collaborative Filtering vs. Matrix Factorization Revisited\n\n\n\n\n\n","category":"type"},{"location":"models/#Multi-Layer-Perceptron","page":"NN Models","title":"Multi Layer Perceptron","text":"","category":"section"},{"location":"models/","page":"NN Models","title":"NN Models","text":"MLPSimilarityModel","category":"page"},{"location":"models/#NeuralCollaborativeFiltering.MLPSimilarityModel","page":"NN Models","title":"NeuralCollaborativeFiltering.MLPSimilarityModel","text":"MLPSimilarityModel(df_train::DataFrame, df_test::DataFrame, emb_size::Int64, model::Chain)\n\nRecommender system based on Multi Layer Perceptron: concatenate 2 Embedding layers and pass the result to the MLP. Sizes of both embedding layers depend on a number of movies and on a number of users respectively.\n\nFields\n\ndf_train::DataFrame: Training set to be learnt on.\ndf_test::DataFrame: Testing set to be learnt on.\nemb_size::Int64: Size of a single embedding vector for one user or for one movie.\nmodel::Chain: Flux.jl Chain object representing the NN.\nfolder_name::String: Assigned to mlp_similarity_ncf in the constructor automatically.\n\nReferences\n\nSource: Figure 1 (right) from Neural Collaborative Filtering vs. Matrix Factorization Revisited\n\n\n\n\n\n","category":"type"},{"location":"models/#Combination-of-GMF-and-MLP-similarity","page":"NN Models","title":"Combination of GMF and MLP similarity","text":"","category":"section"},{"location":"models/","page":"NN Models","title":"NN Models","text":"GMFAndMLPModel","category":"page"},{"location":"models/#NeuralCollaborativeFiltering.GMFAndMLPModel","page":"NN Models","title":"NeuralCollaborativeFiltering.GMFAndMLPModel","text":"GMFAndMLPModel(df_train::DataFrame, df_test::DataFrame, emb_size::Int64, model::Chain)\n\nRecommender system based on a combination of Generalized Matrix Factorization and Multi Layer Perceptron similarity. Sizes of both embedding layers depend on a number of movies and on a number of users respectively.\n\nFields\n\ndf_train::DataFrame: Training set to be learnt on.\ndf_test::DataFrame: Testing set to be learnt on.\nemb_size::Int64: Size of a single embedding vector for one user or for one movie.\nmodel::Chain: Flux.jl Chain object representing the NN.\nfolder_name::String: Assigned to gmf_and_mlp_ncf in the constructor automatically.\n\nReferences\n\nSource: Figure 3 from Neural Collaborative Filtering 2017\n\n\n\n\n\n","category":"type"},{"location":"models/#build_model()-function","page":"NN Models","title":"build_model() function","text":"","category":"section"},{"location":"models/","page":"NN Models","title":"NN Models","text":"\"Wrapper\" function build_model() uses Julia's multiple dispatch to create a neural network model based on the type of arguments:","category":"page"},{"location":"models/","page":"NN Models","title":"NN Models","text":"build_model","category":"page"},{"location":"models/#NeuralCollaborativeFiltering.build_model","page":"NN Models","title":"NeuralCollaborativeFiltering.build_model","text":"build_model(x::Type{DotProductModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50, share_embeddings=nothing)\n\nConstructs and returns an instance of DotProductModel using training and testing data sizes along with specified model parameters.\n\nArguments\n\nx::Type{DotProductModel}: The type of the model to be created. Multiple dispatch allows to define build_model for every model type differently.\ndf_train::DataFrame: The training data frame. Used to get the size of the embedding layers based on the numbers of movies and users.\ndf_test::DataFrame: The testing data frame. Same as df_train.\nembeddings_size::Int=50 (optional): The size of the embedding vectors for both users and movies.\nshare_embeddings::Any=nothing (optional): Redundant here. Used in GMFAndMLPModel.\n\nReturns\n\nDotProductModel: An instance of DotProductModel with the initialized Flux Chain.\n\nExample\n\nm = build_model(DotProductModel, df_train, df_test, embeddings_size=60)\n\n\n\n\n\nbuild_model(x::Type{MLPSimilarityModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50)\n\nConstructs and returns an instance of MLPSimilarityModel using training and testing data sizes along with specified model parameters.\n\nArguments\n\nx::Type{MLPSimilarityModel}: The type of the model to be created. Multiple dispatch allows to define build_model for every model type differently.\ndf_train::DataFrame: The training data frame. Used to get the size of the embedding layers based on the numbers of movies and users.\ndf_test::DataFrame: The testing data frame. Same as df_train.\nembeddings_size::Int=50 (optional): The size of the embedding vectors for both users and movies.\nshare_embeddings::Any=nothing (optional): Redundant here. Used in GMFAndMLPModel.\n\nReturns\n\nMLPSimilarityModel: An instance of MLPSimilarityModel with the initialized Flux Chain.\n\nExample\n\nm = build_model(MLPSimilarityModel, df_train, df_test, embeddings_size=60)\n\n\n\n\n\nbuild_model(x::Type{GMFAndMLPModel}, df_train::DataFrame, df_test::DataFrame; embeddings_size=50, share_embeddings=false)\n\nConstructs and returns an instance of GMFAndMLPModel using training and testing data sizes along with specified model parameters.\n\nArguments\n\nx::Type{GMFAndMLPModel}: The type of the model to be created. Multiple dispatch allows to define build_model for every model type differently.\ndf_train::DataFrame: The training data frame. Used to get the size of the embedding layers based on the numbers of movies and users.\ndf_test::DataFrame: The testing data frame. Same as df_train.\nembeddings_size::Int=50 (optional): The size of the embedding vectors for both users and movies.\nshare_embeddings::Bool=false: If set true, the embeddings will be shared between the GMF and MLP parts of the model. Otherwise, two separate sets of embeddings (4 Embedding layers in total).\n\nReturns\n\nGMFAndMLPModel: An instance of GMFAndMLPModel with the initialized Flux Chain.\n\nExample\n\nm = build_model(model_type, df_train, df_test, embeddings_size=60, share_embeddings=true)\n\n\n\n\n\n","category":"function"},{"location":"#NeuralCollaborativeFiltering.jl-[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://poludmik.github.io/NeuralCollaborativeFiltering.jl/dev/)-[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JuliaTeachingCTU/ImageInspector.jl/blob/master/LICENSE)-[![Build-Status](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/actions/workflows/CI.yml/badge.svg?branchmaster)](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/actions/workflows/CI.yml?querybranch%3Amaster)","page":"Home","title":"NeuralCollaborativeFiltering.jl (Image: ) (Image: License) (Image: Build Status)","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: image info)","category":"page"},{"location":"","page":"Home","title":"Home","text":"A Julia implementation of three different recommender systems based on the concept of Neural Collaborative Filtering. It utilizes the MovieLens dataset, consisting of users, their movie ratings, movie genres, etc. The goal is to predict how much a certain user will like a particular movie, i.e., (user, movie) -> rating prediction.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For a quick start, check out an example of the training process in  scripts/testing_package.jl. Activate the environment by running the ] activate . command from within the scripts/ folder and run the script with include(\"testing_package.jl\").","category":"page"},{"location":"#Implemented-recommender-systems","page":"Home","title":"Implemented recommender systems","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"As for now, 3 models are implemented in this package:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Model Reference Details\nRecommender system based on Generalized Matrix Factorization Figure 1 (left) from Neural Collaborative Filtering vs. Matrix Factorization Revisited (2020) 2 Embedding layers with a dot product\nRecommender system based on Multi Layer Perceptron Figure 1 (right) from Neural Collaborative Filtering vs. Matrix Factorization Revisited (2020) Concatenate 2 Embedding layers and pass the result to the MLP\nRecommender system based on a combination of Generalized Matrix Factorization and Multi Layer Perceptron similarity Figure 3 from Neural Collaborative Filtering 2017 Combining both of the above by concatenating their outputs and passing to the NeuMF layer afterwards","category":"page"},{"location":"","page":"Home","title":"Home","text":"All three models utilize Embedding layers that maintain an embedding vector for each user and each movie in the dataset.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The build_model() function leverages Julia's multiple dispatch capability, constructing a model object based on the argument types, e.g.:","category":"page"},{"location":"","page":"Home","title":"Home","text":"m_dot = build_model(DotProductModel, df_train, df_test, embeddings_size=60)\nm_both = build_model(GMFAndMLPModel, df_train, df_test, embeddings_size=60, share_embeddings=true)","category":"page"},{"location":"#Dataset","page":"Home","title":"Dataset","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"I downloaded the MovieLens dataset from here and stored it in the datasets/ folder. Then I wrote an ugly and unoptimized filter_movielens.jl to extract the data that is actually needed for this project. The main resulting data files (train CSV file and test CSV file) are structured in a format compatible for training all three models:","category":"page"},{"location":"","page":"Home","title":"Home","text":"user movie score\n159 2752 0.85\n610 3228 0.19\n... ... ...","category":"page"},{"location":"","page":"Home","title":"Home","text":"However, this project can easily be extended contain other types of recommender systems, that, e.g. would also consider the features of movies. That's why I've also extracted 20 movie genre features for each of them and divided data to trainset and testset according to the previous 'user' x 'movie' x 'score' split. Features are boolean encoded, each column represents a specified genre.","category":"page"},{"location":"#Model-evaluation","page":"Home","title":"Model evaluation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Other than staring at the decreasing test loss during training it is also good to see how the rankings produced by the system effect the order and the accuracy of several recommendations to a single user.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For example, let's take the case from the image below: for user number 1, we have the ground truth order of recommendations with movie ids: y = 131 2766 47 1517 . The movie 131 is the most relevant one. The recommender system produces the following order of movie relevances: hat y = 131 2766 47 1688 . We can see that the top 3 reccomendations were predicted right where they should be, which is good. The recommended movie number 1688 isn't even relevant according to y; this is bad. Then, the 1517 2227 and 2146 are all shifted down by one rank; this is not that bad, but not good. Describing it with \"it's good\" or \"it's bad\" terms is ok, but we want a numerical evaluation.","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: image info)","category":"page"},{"location":"","page":"Home","title":"Home","text":"There are 4 different metrics implemented in metrics.jl. More specifically:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Accuracy computes the proportion of elements that match element-wise between the ground truth vector y and the predicted vector hat y.\nAverage precision calculates the average precision of a ranked list of items.\nReciprocal rank, a well-known concept, calculates the inverse relative position of the first true item. Later used in the MRR calculation.\nExtended Reciprocal rank, inspired by [3]. Measures the order as well as the relative distance of the rankings.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Then, those functions are utilized by the functions in evaluate_model.jl to calculate MRR (Mean Reciprocal Rank), MAP (Mean Average Precision), MeanAcc (Mean Accuracy), and MeanExtRR (Mean Extended Reciprocal Rank), which are the same metrics but averaged across all users.","category":"page"},{"location":"#Results","page":"Home","title":"Results","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"My primary objective for this project was to create a Julia package featuring examples of neural network models from both [1] and [2], which, to my knowledge, don't yet exist.","category":"page"},{"location":"","page":"Home","title":"Home","text":"So far, I have not dedicated significant time to optimizing the hyperparameters of the models (such as the sizes of embeddings, learning rates, etc.), resulting in inconsistent performance. However, after running multiple tests to ensure the models' capacity for learning, I've empirically observed that the most sophisticated of the three models, the GMFAndMLPModel, tends to outperform the others by a few percentage points. It's worth noting that these results could improve substantially with more focused efforts on hyperparameter tuning. ","category":"page"},{"location":"","page":"Home","title":"Home","text":"The best results I have achieved are as follows: (MeanExtRR = 0.6229, MRR = 0.4728, MAP = 0.2803, MeanACC = 0.0858).","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"[1] Neural Collaborative Filtering (2017)","category":"page"},{"location":"","page":"Home","title":"Home","text":"[2] Neural Collaborative Filtering vs. Matrix Factorization Revisited (2020)","category":"page"},{"location":"","page":"Home","title":"Home","text":"[3] Extended Reciprocal Rank","category":"page"}]
}
