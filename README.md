# NeuralCollaborativeFiltering.jl [![Build Status](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/actions/workflows/CI.yml?query=branch%3Amaster)

﻿<img src="README/dot_mlp_and_both.png">

A Julia implementation of 3 different recommender systems based on the idea of Neural Collaborative Filtering. It utilizes the MovieLens dataset consising of users, their movie ratings, movie genres, etc. The goal is to predict how much a certain user will like a certain movie, i.e. (user, movie) -> rating prediction.

:information_source: Quickstart with an example of the training process: [scripts/testing_package.jl](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/blob/master/scripts/testing_package.jl). Activate the environment with a `] activate .` command from the `scripts/` folder and run `include("testing_package.jl")`.

# Implemented recommender systems
So far, 3 models are present in this package:

| Model | Reference | Details |
|-------|------|------|
| Recommender system based on Generalized Matrix Factorization | Figure 1 (left) from [Neural Collaborative Filtering vs. Matrix Factorization Revisited (2020)](https://arxiv.org/pdf/2005.09683.pdf) | 2 Embedding layers with a dot product |
| Recommender system based on Multi Layer Perceptron | Figure 1 (right) from [Neural Collaborative Filtering vs. Matrix Factorization Revisited (2020)](https://arxiv.org/pdf/2005.09683.pdf) | Concatenate 2 Embedding layers and pass the result to the MLP |
| Recommender system based on a combination of Generalized Matrix Factorization and Multi Layer Perceptron similarity | Figure 3 from [Neural Collaborative Filtering 2017](https://arxiv.org/pdf/1708.05031.pdf) | Combining both of the above by concatenating their outputs and passing to the NeuMF layer afterwards |

All 3 use Embedding layers that store an embedding vector for each user and for each movie in the dataset.

Method `build_model` uses Julia's multiple dispatch and constructs a model object based on the types of the arguments, e.g.:
```julia
m_dot_ = build_model(DotProductModel, df_train, df_test, embeddings_size=60)
m_both = build_model(GMFAndMLPModel, df_train, df_test, embeddings_size=60, share_embeddings=true)
```

# Dataset
I've downloaded the MovieLens dataset from [here](https://grouplens.org/datasets/movielens/#:~:text=recommended%20for%20education%20and%20development) and stored it in the `datasets/` folder. Then I wrote an ugly and unoptimized script to extract the data that was actually needed for this project. The main resulting data files ([train CSV file](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/blob/master/datasets/ml-latest-small/user_movie_pairs_for_coll_filtr_train.csv) and [test CSV file](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/blob/master/datasets/ml-latest-small/user_movie_pairs_for_coll_filtr_test.csv)) are in the form, which is used to train all 3 models:

| user | movie | score |
|------|-------|-------|
| 159  | 2752  | 0.85  |
| 610  | 3228  | 0.19  |
| ...  | ...   | ...   |

However, this project can easily be extended contain other types of recommender systems, that, e.g. would also consider the features of movies.
That's why I've also extracted 20 movie genre features for each of them and divided data to [trainset](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/blob/master/datasets/ml-latest-small/movie_genre_df_train.csv) and [testset](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/blob/master/datasets/ml-latest-small/movie_genre_df_test.csv) according to the previous 'user' x 'movie' x 'score' split. Features are boolean encoded, each column represents a specified genre, i.e.:
1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20
-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-
0|0|0|0|0|0|1|0|0|0|0|0|1|0|0|0|0|0|0|0
0|0|0|0|0|1|1|0|0|0|0|0|0|0|0|0|1|0|0|0

# Model evaluation
Other than staring at the decreasing test loss during training it is also good to see how the rankings produced by the system effect the order and the accuracy of several recommendations to a single user.

> *For example*: for some user number $228$, we have the ground truth order of recommendations with movie ids: $y = [35, 12, 48]$. And the recommender system produces the following order of movie relevances: $\hat y = [12, 55, 48]$. We can see that the $12$ was ranked one rank higher, which is not good. The recommended movie number $55$ isn't even relevant according to $y$; this is bad. The movie $48$ is on the right spot, which is good. Describing it in 'good' or 'bad' terms is ok, but we want to a numerical evaluation.

There are 4 different metrics used ([metrics.jl](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/blob/master/src/evaluation/metrics.jl)). More specifically:

* **Accuracy** computes the proportion of elements that match element-wise between the ground truth vector $y$ and the predicted vector $\hat y$.

* **Average precision** calculates the average precision of a ranked list of items.

* **Reciprocal rank**, a well-known concept, calculates the inverse relative position of the first true item. Later used in the MRR calculation.

* **Extended Reciprocal rank**, inspired by this [article](https://towardsdatascience.com/extended-reciprocal-rank-ranking-evaluation-metric-5929573c778a). Measures the order as well as the relative distance of the rankings.

Then, those functions are also used by the [evaluate_model.jl](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/blob/master/src/evaluation/metrics.jl) to calculate MRR, MAP, MeanAcc and MeanExtRR on all users.




