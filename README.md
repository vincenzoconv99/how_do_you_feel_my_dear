# How do you feel my dear?

This project belongs to the Information Retrieval field and presents a solution for predicting the emotion of a tweet. 

In the file 'emotions_detection.ipybn' you can find all the experiments, while the files 'utils.py' and 'tuning.py' contain utility functions.

Aiming at experimenting both sequence learning techniques and classical ones we decided to use three different Machine Learning models.
Below you can find details about the employed architectures and hyperparameters.

### RNN based on static embeddings

#### Fixed Architecture:
<list>
  <li>Embedding Layer, 200 dimensions</li>
  <li>Dropout, 0.2 rate, ReLU</li>
  <li>LSTM, 50 units, tanh</li>
  <li>Output, 5 units, Softmax</li>
</list>

#### Fixed Hyperparamters:
<list>
<li>learning rate = 0.0001</li>
<li>batch size = 64</li>
<li>optimizer = Adam</li>
<li>Max no. of epochs = 10</li>
</list>

#### Meta Architecture:
<list>
<li>Embedding Layer, dimensions: min=100; max=400; step=100</li>
<li>Dropout, rate: min=0.0; max=0.9; step=0.1, ReLU</li>
<li>LSTM, units: min=50; max=200; step=50, tanh</li>
<li>Output, 5 units, Softmax</li>
</list>

#### Meta Hyperparameters:
<list>
<li>learning rate: min = 1e-5, max = 1e-2</li>
<li>batch size = 64</li>
<li>optimizer = Adam</li>
<li>Max no. of epochs = 10</li>
</list>


### MLP based on BERT encoder

#### Fixed Architecture:
<list>
  <li>Keras Layer, BERT encoder tiny variant, trainable=True</li>
  <li>Dense Layer, 64 units, linear</li>
  <li>Dropout, 0.1 rate</li>
  <li>Output, 5 units, Softmax</li>
</list>


#### Fixed Hyperparamters:
<list>
<li>learning rate = 0.0001</li>
<li>batch size = 64</li>
<li>optimizer = Adam</li>
<li>Max no. of epochs = 10</li>
</list>

### RF based on Tf-Idf Vectorizer

#### Fixed Hyperparameters:
<list>
<li>n_estimators = 100</li>
<li>min_samples_split = 2</li>
<li>min_samples_leaf = 1</li>
<li>max_features = 'sqrt'</li>
</list>


#### Meta Hyperparamters:
<list>
<li>n_estimators: min = 100, max = 200, step = 10</li>
<li>min_samples_split = [2, 4, 6] </li>
<li>min_samples_leaf = [1, 2]</li>
<li>max_features =  ['auto','sqrt','log2']</li>
</list>
