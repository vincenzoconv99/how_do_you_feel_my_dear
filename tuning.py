import keras_tuner as kt
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, Dropout, Bidirectional, LSTM, Dense, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import pandas as pd

class MetaRNN(kt.HyperModel):

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def build(self, hp: kt.HyperParameters)->Model:
        """Returns a Meta Feed-Forward Neural Network model for binary classification.
        
        Parameters
        -----------------------
        hp: kt.HyperParameters,
            Container for both a hyperparameter space, and current values.
            
        Returns
        -----------------------
        The MetaModel to search hyperparameters in.
        """
        encoder = TextVectorization()
        encoder.adapt(self.train_dataset.map(lambda text, label: text))
        model = Sequential()
        model.add(encoder)
        model.add(Embedding(len(encoder.get_vocabulary()), 
                            hp.Int('embedding_dim', min_value = 100, max_value = 400, default = 200, step=100), mask_zero=True))
        model.add(Dropout( hp.Float('drop_rate', min_value = 0.0, max_value = 0.9, step = 0.1, default = 0.2)))
        model.add(Bidirectional(LSTM(hp.Int('embedding_dim', min_value = 50, max_value = 200, default = 50, step=50), 
                                     return_sequences=False)))
        model.add(Dense(5, activation = 'softmax'))
        model.compile(loss = CategoricalCrossentropy(),
                      optimizer=Adam(hp.Float('lr', min_value = 1e-5, max_value = 1e-2, default = 1e-4)),
                      metrics=['accuracy'])
        return model
                  
def tune_model(
    meta_model: kt.HyperModel,
    model_name: str,
    train_dataset,
    val_dataset,
    test_dataset,
    holdout_number: int
):
    """Tune the provided MetaModel and returns the best_model found and a results_summary.
    
    Parameters
    ---------------------
    meta_model: kt.HyperModel,
        The metamodel to search parameter in.
    model_name: str,
        The meta model name.
    train_dataset,
        The training dataset.
    val_dataset,
        The validation dataset to optimize parameters on.
    test_dataset,
        The test dataset to evaluate the best model on.
    holdout_number: int,
        The number of the current external cv iteration.
        
    Returns
    ----------------------
    Tuple with best_model and model evaluations dataframe.
    """
    OBJECTIVE = kt.Objective('val_accuracy', direction = 'max')
    MAX_TRIALS = 20
    EXECUTIONS_PER_TRIAL = 1
    hp = kt.HyperParameters()

    callbacks = [EarlyStopping( 
                     monitor = 'accuracy',
                     min_delta = 0.001,
                     patience = 2,
                     mode = 'max',
                     restore_best_weights = True
                 )]
    
    tuner = kt.RandomSearch(
          hypermodel = meta_model,
          objective = OBJECTIVE,
          hyperparameters = hp,
          tune_new_entries = True,
          max_trials = MAX_TRIALS,
          executions_per_trial = EXECUTIONS_PER_TRIAL,
          seed = 42,
          directory = 'tuning/',
          project_name = model_name+'holdout'+str(holdout_number)
    )

    tuner.search(
      train_dataset,
      validation_data=val_dataset,
      epochs = 10,
      callbacks = callbacks,
      verbose = 1
    )
    
    #Creating evalutation dataframe
    best_model = tuner.get_best_models()[0]
    results_summary = pd.DataFrame(tuner.results_summary())
    
    return best_model, results_summary