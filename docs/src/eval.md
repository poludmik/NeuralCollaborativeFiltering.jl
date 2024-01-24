# Model evaluation

Metrics for model evaluation are defined in `src/evaluation/` folder. 

## Metrics
File `metrics.jl` defines metrics to compare the quality of the vector of recommended items relative to the ground truth vector (see [README](index.md)):

```@docs
accuracy
```

```@docs
average_precision
```

```@docs
reciprocal_rank
```

```@docs
extended_reciprocal_rank
```

## User-wise evaluation
Then, the `evaluate_model.jl` utilizes the above metrics to evaluate the model on a specific user or on all the users available in the provided dataset.

```@docs
evaluate_model_on_1_user
```

```@docs
evaluate_model
```