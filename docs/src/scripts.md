# Scripts

## testing_package.jl
Contains all the basic functionality of the package:
- Loading the dataset
- Loading the weights or training a new model
- Evaluating the model on the provided test set.

## filter_movielens.jl
An ugly script to extract needed training and testing data from the `movielens` dataset.
Leaving code as it is because it would always be different for different datasets.

Scaled the rankings from 1 to 5 stars to <0.1, 1> by MinMaxScale.

Splitted data to the train/test randomly.

Resulting dataset that is used in the NeuralCollaborativeFiltering training is of form:

| user | movie | score|
|-------|------|------|
| 1    |    47 |   1.0|
| 2    |   256 |   0.7|
...

Also, I extracted the movie/genre pairs, which could potentially be used in the recommendation system that would consider those as additional features (see [README](index.md)).
