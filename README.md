# How do you feel my dear?

This project belongs to the Information Retrieval field and presents a solution for predicting the emotion of a tweet. 

In this work we will focus on textual data and employ Natural Language Processing (NLP) techniques. Specifically, we will use two twitter based datasets. In order to reduce the complexity and balance the merged dataset, we discarded out tweets that were not labeled with a primary emotion according to Plutchik theory.

Aiming at experimenting both sequence learning techniques and classical ones we decided to use three different models.

### RNN based on static embeddings

Fixed Architecture:
-Embedding Layer, 200 dimensions
-Dropout, 0.2 rate, ReLU
-LSTM, 50 units, tanh
-Output, 5 units, Softmax

Fixed Hyperparamters:
-learning rate = 0.0001
-batch size = 64
-optimizer = Adam
-Max no. of epochs = 10

Meta Architecture:
-Embedding Layer, dimensions: min=100; max=400; step=100
-Dropout, rate: min=0.0; max=0.9; step=0.1, ReLU
-LSTM, units: min=50; max=200; step=50, tanh
-Output, 5 units, Softmax

Meta Hyperparameters:
-learning rate: min = 1e-5, max = 1e-2
-batch size = 64
-optimizer = Adam
-Max no. of epochs = 10

