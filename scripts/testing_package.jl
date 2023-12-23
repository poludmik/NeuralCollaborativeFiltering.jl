using NeuralCollaborativeFiltering

using DataFrames
using CSV
using Random
using Flux
using JLD2

Random.seed!(228)



######### Load dataset: ###########
path_train = "..\\datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_train.csv"
df_train = DataFrame(CSV.File(path_train))
path_test = "..\\datasets\\ml-latest-small\\user_movie_pairs_for_coll_filtr_test.csv"
df_test = DataFrame(CSV.File(path_test))



######## Select the model: ########
# model_type = DotProductModel
# weights_path = "..\\weights\\dot_product_ncf\\model_dim60_bs1024_ep5_lr0.001.jld2"

# model_type = MLPSimilarityModel
# weights_path = "..\\weights\\mlp_similarity_ncf\\model_dim60_bs1024_ep101_lr0.005.jld2"

model_type = GMFAndMLPModel
weights_path = "..\\weights\\gmf_and_mlp_ncf\\model_dim60_bs1024_ep5_lr0.001.jld2"
share_embeddings = true



######## Train the model: #########
# m = build_model(model_type, df_train, df_test, embeddings_size=60, share_embeddings=share_embeddings)
# print(m)
# weights_path, plot_path = train_model(df_train, df_test, m, n_epochs=5, lr=0.001, bs=1024)



######## Load the model: ##########
filename = weights_path
model_state = JLD2.load(filename, "model_state")
emb_size = Int(JLD2.load(filename, "emb_size"))
model = build_model(model_type, df_train, df_test, embeddings_size=emb_size, share_embeddings=share_embeddings)
model.emb_size = emb_size
Flux.loadmodel!(model.model, model_state)



####### Evaluate the model: ########
evaluate_model_on_1_user(model, 1, df_test, top_n_mrr=5);
evaluate_model(df_test, model);
