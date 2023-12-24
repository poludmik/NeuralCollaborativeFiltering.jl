# NeuralCollaborativeFiltering.jl [![Build Status](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/actions/workflows/CI.yml?query=branch%3Amaster)

﻿<img src="README/dot_mlp_and_both.png">

A Julia implementation of 3 different recommender systems based on the idea of Neural Collaborative Filtering. It utilizes the MovieLens dataset consising of users, their movie ratings, movie genres, etc. The goal is to predict how much a certain user will like a certain movie, i.e. (user, movie) -> rating prediction.

:information_source: Quickstart with an example of the training process: [scripts/testing_package.jl](https://github.com/poludmik/NeuralCollaborativeFiltering.jl/blob/master/scripts/testing_package.jl). Activate the environment with `] activate .` from the `scripts/` folder and run `include("testing_package.jl")`.

# Implemented recommender systems
So far, 3 models are present in this package:

| Model | Reference | Details |
|-------|------|------|
| Recommender system based on Generalized Matrix Factorization | Figure 1 (left) from [Neural Collaborative Filtering vs. Matrix Factorization Revisited (2020)](https://arxiv.org/pdf/2005.09683.pdf) | 2 Embedding layers with a dot product |
| Recommender system based on Multi Layer Perceptron | Figure 1 (right) from [Neural Collaborative Filtering vs. Matrix Factorization Revisited (2020)](https://arxiv.org/pdf/2005.09683.pdf) | Concatenate 2 Embedding layers and pass the result to the MLP |
| Recommender system based on a combination of Generalized Matrix Factorization and Multi Layer Perceptron similarity | Figure 3 from [Neural Collaborative Filtering 2017](https://arxiv.org/pdf/1708.05031.pdf) | Combining both of the above by concatenating their outputs and passing to the NeuMF layer afterwards |

