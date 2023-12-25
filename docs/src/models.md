# NN Models

Models are in the `src/models/` folder of the repository. For each model, there is a struct that is a subtype of NCFModel. The constructor ensures the proper `folder_name` member parameter setting.

## Generalized Matrix Factorization 
```@docs
DotProductModel
```

## Multi Layer Perceptron
```@docs
MLPSimilarityModel
```

## Combination of GMF and MLP similarity
```@docs
GMFAndMLPModel
```

## build_model() function
"Wrapper" function `build_model()` uses Julia's **multiple dispatch** to create a neural network model based on the type of arguments:
```@docs
build_model
```
