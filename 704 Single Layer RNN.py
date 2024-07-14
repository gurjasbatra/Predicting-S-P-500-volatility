# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:08:58 2024

@author: hp
"""
#RNN Model

import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import HyperModel, BayesianOptimization
from keras_tuner.tuners import RandomSearch
import keras_tuner as kt

#%% Loading the merged Data
data = pd.read_pickle("Project2Data.pkl")
data.set_index(['date'], inplace=True)

#Scaling the data
data['rv5'] = data['rv5'] * (100**2)
data.dropna(inplace = True)
print()


#%% Calculating the moving averages
data['rv5_5'] = data['rv5'].rolling(5).mean()
data['rv5_21'] = data['rv5'].rolling(21).mean()

# Introducing lagged variable in the dataset
data['rv5_lag'] = data['rv5'].shift(-1)
data.dropna(inplace=True)

#%% Creating univariate and multivariate models
model_var_1 = ['rv5'] #Univariate

# Multivariate Model 2 with introducing interest rates and VIX Closing
model_var_2= ['rv5', 'dgs1', 'dgs10', 'dgs2', 'dgs30', 'dtb3', 'vixcls']

# Multivariate Model 3 with introducing just interest rates
model_var_3= ['rv5', 'dgs1', 'dgs10', 'dgs2', 'dgs30', 'dtb3']

# Multivariate Model 4 with introducing just VIX Closing
model_var_4 = ['rv5', 'vixcls']

# Multivariate Model 5 with introducing Moving Averages
model_var_5 = ['rv5', 'rv5_5', 'rv5_21']

# Model with just the lag of rv5
IR=['rv5_lag']

#%% Splitting and training the data for each model

train_stdscaler = StandardScaler()

train_stdscaler = StandardScaler()

#For Model 1
rv1_train = data.loc[:'2012', model_var_1].copy().dropna()
rv1_valid = data.loc['2013':'2016', model_var_1].copy().dropna()
rv1_test = data.loc['2017':, model_var_1].copy().dropna()

#For Model 2
rv2_train = data.loc[:'2012', model_var_2].copy().dropna()
rv2_valid = data.loc['2013':'2016', model_var_2].copy().dropna()
rv2_test = data.loc['2017':, model_var_2].copy().dropna()

#For Model 3
rv3_train = data.loc[:'2012', model_var_3].copy().dropna()
rv3_valid = data.loc['2013':'2016', model_var_3].copy().dropna()
rv3_test = data.loc['2017':, model_var_3].copy().dropna()

#For Model 4
rv4_train = data.loc[:'2012', model_var_4].copy().dropna()
rv4_valid = data.loc['2013':'2016', model_var_4].copy().dropna()
rv4_test = data.loc['2017':, model_var_4].copy().dropna()

#For Model 5
rv5_train = data.loc[:'2012', model_var_5].copy().dropna()
rv5_valid = data.loc['2013':'2016', model_var_5].copy().dropna()
rv5_test = data.loc['2017':, model_var_5].copy().dropna()

#For Model 6
IR_train = data.loc[:'2012', IR].copy().dropna()
IR_valid = data.loc['2013':'2016', IR].copy().dropna()
IR_test = data.loc['2017':, IR].copy().dropna()


#%% Normalizing the data for each model
train_stdscaler = StandardScaler()

rv1_train = pd.DataFrame(train_stdscaler.fit_transform(rv1_train),
                        columns=rv1_train.columns,
                        index=rv1_train.index)

rv1_valid = pd.DataFrame(train_stdscaler.transform(rv1_valid),
                        columns=rv1_valid.columns,
                        index=rv1_valid.index)

rv1_test = pd.DataFrame(train_stdscaler.transform(rv1_test),
                        columns=rv1_test.columns,
                        index=rv1_test.index)

rv2_train = pd.DataFrame(train_stdscaler.fit_transform(rv2_train),
                        columns=rv2_train.columns,
                        index=rv2_train.index)

rv2_valid = pd.DataFrame(train_stdscaler.transform(rv2_valid),
                        columns=rv2_valid.columns,
                        index=rv2_valid.index)

rv2_test = pd.DataFrame(train_stdscaler.transform(rv2_test),
                        columns=rv2_test.columns,
                        index=rv2_test.index)

rv3_train = pd.DataFrame(train_stdscaler.fit_transform(rv3_train),
                        columns=rv3_train.columns,
                        index=rv3_train.index)

rv3_valid = pd.DataFrame(train_stdscaler.transform(rv3_valid),
                        columns=rv3_valid.columns,
                        index=rv3_valid.index)

rv3_test = pd.DataFrame(train_stdscaler.transform(rv3_test),
                        columns=rv3_test.columns,
                        index=rv3_test.index)

rv4_train = pd.DataFrame(train_stdscaler.fit_transform(rv4_train),
                        columns=rv4_train.columns,
                        index=rv4_train.index)

rv4_valid = pd.DataFrame(train_stdscaler.transform(rv4_valid),
                        columns=rv4_valid.columns,
                        index=rv4_valid.index)

rv4_test = pd.DataFrame(train_stdscaler.transform(rv4_test),
                        columns=rv4_test.columns,
                        index=rv4_test.index)

rv5_train = pd.DataFrame(train_stdscaler.fit_transform(rv5_train),
                        columns=rv5_train.columns,
                        index=rv5_train.index)

rv5_valid = pd.DataFrame(train_stdscaler.transform(rv5_valid),
                        columns=rv5_valid.columns,
                        index=rv5_valid.index)

rv5_test = pd.DataFrame(train_stdscaler.transform(rv5_test),
                        columns=rv5_test.columns,
                        index=rv5_test.index)

IR_train = pd.DataFrame(train_stdscaler.fit_transform(IR_train),
                       columns=IR_train.columns,
                        index=rv1_train.index)

IR_valid = pd.DataFrame(train_stdscaler.transform(IR_valid),
                        columns=IR_valid.columns,
                        index=IR_valid.index)

IR_test = pd.DataFrame(train_stdscaler.transform(IR_test),
                        columns=IR_test.columns,
                        index=IR_test.index)



#%% Define Sequence for RNN for Model 2

seq_length = 21
tf.random.set_seed(42)  # extra code – ensures reproducibility

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv2_train.to_numpy(),
    targets=rv2_train['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=False,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv2_valid.to_numpy(),
    targets=rv2_valid['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv2_test.to_numpy(),
    targets=rv2_test['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

#%% Build and Train Model
tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, len(model_var_2)]),
])

model.output_shape

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse", patience=80, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt, metrics=["mse"])

history = model.fit(train_ds, validation_data=valid_ds, epochs=200,
                    callbacks=[early_stopping_cb])

#%% Plot History
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 250], ylim=[0, 1.5], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.show()

#%% Evaluate Model
def tx_helper_predict(pred, tx):
    mean = tx.mean_[0]
    var = tx.var_[0]
    return mean + np.sqrt(var) * pred

def tx_helper_mse(pred, tar, tx, seq_length):
    y_pred = tx_helper_predict(pred, tx)
    y_true = tar['rv5'][seq_length:].values
    return mean_squared_error(y_true, y_pred)

mse_train_scaled = tx_helper_mse(model.predict(train_ds).reshape(-1,1), rv1_train, train_stdscaler, seq_length)
print("Train Error: {}".format(mse_train_scaled))

mse_valid_scaled = tx_helper_mse(model.predict(valid_ds).reshape(-1,1), rv1_valid, train_stdscaler, seq_length)
print("Validation Error: {}".format(mse_valid_scaled))

mse_test_scaled = tx_helper_mse(model.predict(test_ds).reshape(-1,1), rv1_test, train_stdscaler, seq_length)
print("Test Error: {}".format(mse_test_scaled))

#%% Plot Results
plt.plot(data['rv5'])
plt.plot(rv1_train.index[seq_length:], tx_helper_predict(model.predict(train_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv1_valid.index[seq_length:], tx_helper_predict(model.predict(valid_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv1_test.index[seq_length:], tx_helper_predict(model.predict(test_ds).reshape(-1,1), train_stdscaler))
plt.title("Single Layer Multivariate RNN (with VIX and IR)")
#plt.xlim([pd.to_datetime("20080101"),pd.to_datetime("20090101")])
plt.show()

#%% Model 3

seq_length = 21
tf.random.set_seed(42)  # extra code – ensures reproducibility

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv3_train.to_numpy(),
    targets=rv3_train['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=False,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv3_valid.to_numpy(),
    targets=rv3_valid['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv3_test.to_numpy(),
    targets=rv3_test['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

#%% Build and Train Model
tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, len(model_var_3)]),
    tf.keras.layers.Dense(1)
])

model.output_shape

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse", patience=75, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt, metrics=["mse"])

history = model.fit(train_ds, validation_data=valid_ds, epochs=200,
                    callbacks=[early_stopping_cb])

#%% Plot History
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 250], ylim=[0, 1.5], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.show()

#%% Evaluate Model
def tx_helper_predict(pred, tx):
    mean = tx.mean_[0]
    var = tx.var_[0]
    return mean + np.sqrt(var) * pred

def tx_helper_mse(pred, tar, tx, seq_length):
    y_pred = tx_helper_predict(pred, tx)
    y_true = tar['rv5'][seq_length:].values
    return mean_squared_error(y_true, y_pred)

mse_train_scaled = tx_helper_mse(model.predict(train_ds).reshape(-1,1), rv3_train, train_stdscaler, seq_length)
print("Train Error: {}".format(mse_train_scaled))

mse_valid_scaled = tx_helper_mse(model.predict(valid_ds).reshape(-1,1), rv3_valid, train_stdscaler, seq_length)
print("Validation Error: {}".format(mse_valid_scaled))

mse_test_scaled = tx_helper_mse(model.predict(test_ds).reshape(-1,1), rv3_test, train_stdscaler, seq_length)
print("Test Error: {}".format(mse_test_scaled))

#%% Plot Results
plt.plot(data['rv5'])
plt.plot(rv3_train.index[seq_length:], tx_helper_predict(model.predict(train_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv3_valid.index[seq_length:], tx_helper_predict(model.predict(valid_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv3_test.index[seq_length:], tx_helper_predict(model.predict(test_ds).reshape(-1,1), train_stdscaler))
plt.title("Single Layer Multivariate RNN (with IR)")
#plt.xlim([pd.to_datetime("20080101"),pd.to_datetime("20090101")])
plt.show()

#%% Model 4

seq_length = 21
tf.random.set_seed(42)  # extra code – ensures reproducibility

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv4_train.to_numpy(),
    targets=rv4_train['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=False,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv4_valid.to_numpy(),
    targets=rv4_valid['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv4_test.to_numpy(),
    targets=rv4_test['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

#%% Build and Train Model
tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, len(model_var_4)]),
    tf.keras.layers.Dense(1)
])

model.output_shape

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse", patience=75, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt, metrics=["mse"])

history = model.fit(train_ds, validation_data=valid_ds, epochs=200,
                    callbacks=[early_stopping_cb])

#%% Plot History
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 250], ylim=[0, 1.5], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.show()

#%% Evaluate Model
def tx_helper_predict(pred, tx):
    mean = tx.mean_[0]
    var = tx.var_[0]
    return mean + np.sqrt(var) * pred

def tx_helper_mse(pred, tar, tx, seq_length):
    y_pred = tx_helper_predict(pred, tx)
    y_true = tar['rv5'][seq_length:].values
    return mean_squared_error(y_true, y_pred)

mse_train_scaled = tx_helper_mse(model.predict(train_ds).reshape(-1,1), rv4_train, train_stdscaler, seq_length)
print("Train Error: {}".format(mse_train_scaled))

mse_valid_scaled = tx_helper_mse(model.predict(valid_ds).reshape(-1,1), rv4_valid, train_stdscaler, seq_length)
print("Validation Error: {}".format(mse_valid_scaled))

mse_test_scaled = tx_helper_mse(model.predict(test_ds).reshape(-1,1), rv4_test, train_stdscaler, seq_length)
print("Test Error: {}".format(mse_test_scaled))

#%% Plot Results
plt.plot(data['rv5'])
plt.plot(rv4_train.index[seq_length:], tx_helper_predict(model.predict(train_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv4_valid.index[seq_length:], tx_helper_predict(model.predict(valid_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv4_test.index[seq_length:], tx_helper_predict(model.predict(test_ds).reshape(-1,1), train_stdscaler))
plt.title("Single Layer Multivariate LSTM (with VIX)")
#plt.xlim([pd.to_datetime("20080101"),pd.to_datetime("20090101")])
plt.show()

#%% Model 5
seq_length = 21
tf.random.set_seed(42)  # extra code – ensures reproducibility

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv5_train.to_numpy(),
    targets=rv5_train['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=False,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv5_valid.to_numpy(),
    targets=rv5_valid['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv5_test.to_numpy(),
    targets=rv5_test['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

#%% Build and Train Model
tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, len(model_var_5)]),
    tf.keras.layers.Dense(1)
])

model.output_shape

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse", patience=75, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt, metrics=["mse"])

history = model.fit(train_ds, validation_data=valid_ds, epochs=200,
                    callbacks=[early_stopping_cb])

#%% Plot History
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 250], ylim=[0, 1.5], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.show()

#%% Evaluate Model
def tx_helper_predict(pred, tx):
    mean = tx.mean_[0]
    var = tx.var_[0]
    return mean + np.sqrt(var) * pred

def tx_helper_mse(pred, tar, tx, seq_length):
    y_pred = tx_helper_predict(pred, tx)
    y_true = tar['rv5'][seq_length:].values
    return mean_squared_error(y_true, y_pred)

mse_train_scaled = tx_helper_mse(model.predict(train_ds).reshape(-1,1), rv5_train, train_stdscaler, seq_length)
print("Train Error: {}".format(mse_train_scaled))

mse_valid_scaled = tx_helper_mse(model.predict(valid_ds).reshape(-1,1), rv5_valid, train_stdscaler, seq_length)
print("Validation Error: {}".format(mse_valid_scaled))

mse_test_scaled = tx_helper_mse(model.predict(test_ds).reshape(-1,1), rv5_test, train_stdscaler, seq_length)
print("Test Error: {}".format(mse_test_scaled))

#%% Plotting Results
plt.plot(data['rv5'])
plt.plot(rv5_train.index[seq_length:], tx_helper_predict(model.predict(train_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv5_valid.index[seq_length:], tx_helper_predict(model.predict(valid_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv5_test.index[seq_length:], tx_helper_predict(model.predict(test_ds).reshape(-1,1), train_stdscaler))
plt.title("Single Layer Multivariate RNN (with Moving Averages)")
#plt.xlim([pd.to_datetime("20080101"),pd.to_datetime("20090101")])
plt.show()

#%% Hyperparameter Tuning for Model 4
# Define Tuner

def keras_code(hp_seq_length, hp_num_layers, hp_units, saving_path):

    seq_length = hp_seq_length
    tf.random.set_seed(42)  # extra code – ensures reproducibility

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        rv4_train.to_numpy(),
        targets=rv4_train['rv5'][seq_length:],
        sequence_length=seq_length,
        batch_size=32,
        shuffle=False,
        seed=42
    )
    
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        rv4_valid.to_numpy(),
        targets=rv4_valid['rv5'][seq_length:],
        sequence_length=seq_length,
        batch_size=32,
    )

    # Build model
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(seq_length, len(model_var_4))))
    model.add(tf.keras.layers.Reshape([1,seq_length * len(model_var_4)]))
    model.add(tf.keras.layers.Flatten())
    
    # Tune the number of layers and nodes
    for i in range(hp_num_layers):
        model.add(tf.keras.layers.Dense(hp_units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))  # Output layer
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mse", patience=25, restore_best_weights=True)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(loss="mse", optimizer=opt, metrics=["mse"])

    model.fit(train_ds, validation_data=valid_ds, epochs=500,
                        callbacks=[early_stopping_cb])

    # Save model
    model.save(saving_path)

    # Return a single float as the objective value.
    # You may also return a dictionary
    # of {metric_name: metric_value}.
    loss_valid, mse_valid = model.evaluate(valid_ds)
    return mse_valid


#%% Grid Search
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure consistent results
tf.random.set_seed(42)

# Data preparation function
def prepare_datasets(data, target_column, sequence_length, shuffle):
    return tf.keras.utils.timeseries_dataset_from_array(
        data=data.to_numpy()[:, :-1],  # Exclude target column from inputs
        targets=data[target_column].to_numpy()[sequence_length:],  # Labels are offset by the sequence length
        sequence_length=sequence_length,
        batch_size=32,
        shuffle=shuffle,
        seed=42 if shuffle else None
    )

# Assuming you have 'rv4_train', 'rv4_valid', 'rv4_test' as DataFrames
seq_length = 21
num_features = rv4_train.shape[1] - 1  # All columns except the target

# Prepare datasets with appropriate shuffling
train_ds = prepare_datasets(rv4_train, 'rv5', seq_length, shuffle=True)
valid_ds = prepare_datasets(rv4_valid, 'rv5', seq_length, shuffle=False)
test_ds = prepare_datasets(rv4_test, 'rv5', seq_length, shuffle=False)

# Build the RNN model
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=10, input_shape=(None, num_features), activation='relu'),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])

# Early stopping callback
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse", patience=75, restore_best_weights=True)

# Train the model
history = model.fit(
    train_ds,
    epochs=200,
    validation_data=valid_ds,
    callbacks=[early_stopping_cb]
)

# Plot training and validation loss
pd.DataFrame(history.history).plot(
    figsize=(8, 5),
    xlabel="Epochs",
    ylabel="MSE",
    title="Model Training and Validation Loss"
)
plt.grid(True)
plt.show()

# Evaluate the model
print("Training MSE:", model.evaluate(train_ds, verbose=0))
print("Validation MSE:", model.evaluate(valid_ds, verbose=0))
print("Test MSE:", model.evaluate(test_ds, verbose=0))



#%% Retrain model to capture history

def keras_code(hp_seq_length, hp_num_layers, hp_units, saving_path):
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)
    seq_length = hp_seq_length
    tf.random.set_seed(42)  # extra code – ensures reproducibility

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        rv4_train.to_numpy(),
        targets=rv4_train['rv5'][seq_length:],
        sequence_length=seq_length,
        batch_size=32,
        shuffle=False,
        seed=42
    )
    
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        rv4_valid.to_numpy(),
        targets=rv4_valid['rv5'][seq_length:],
        sequence_length=seq_length,
        batch_size=32,
    )

    # Build model
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(seq_length, len(model_var_4))))
    model.add(tf.keras.layers.Reshape([1,seq_length * len(model_var_4)]))
    model.add(tf.keras.layers.Flatten())
    
    # Tune the number of layers and nodes
    for i in range(hp_num_layers):
        model.add(tf.keras.layers.Dense(hp_units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))  # Output layer
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mse", patience=25, restore_best_weights=True)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(loss="mse", optimizer=opt, metrics=["mse"])

    model.fit(train_ds, validation_data=valid_ds, epochs=200,
                        callbacks=[early_stopping_cb])

    # Save model
    model.save(saving_path)

    # Return a single float as the objective value.
    # You may also return a dictionary
    # of {metric_name: metric_value}.
    loss_valid, mse_valid = model.evaluate(valid_ds)
    return mse_valid


#%% Bayesian Search
class MyTuner(kt.BayesianOptimization):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        return keras_code(
            hp_seq_length=hp.Int("hp_seq_length", 1, 150, 5),
            hp_num_layers=hp.Int("hp_num_layers", 1, 5, 1),
            hp_units=hp.Int("hp_units", 10, 30, 5),
            saving_path=os.path.join("tmp", f"{trial.trial_id}.keras")
        )


tuner = MyTuner(
    objective=kt.Objective("mse", direction="min"),
    max_trials=100,
    overwrite=True,
    directory="my_dir",
    project_name="keep_code_separate",
)
#%% Running the Tuner
tuner.search()
# Retraining the model
best_hp = tuner.get_best_hyperparameters()[0]
keras_code(**best_hp.values, saving_path="tmp/best_model_{}.keras".format(best_hp.values['hp_seq_length']))

#%% Loading the model
best_model = tf.keras.models.load_model("tmp/best_model_{}.keras".format(best_hp.values['hp_seq_length']))

#%% Retraining the model
seq_length = best_model.input_shape[1]
tf.random.set_seed(42)  

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv4_train.to_numpy(),
    targets=rv4_train['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=False,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv4_valid.to_numpy(),
    targets=rv4_valid['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    rv4_test.to_numpy(),
    targets=rv4_test['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

#%% Retraining the model to capture history

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse", patience=500, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

best_model.compile(loss="mse", optimizer=opt, metrics=["mse"])

history = best_model.fit(train_ds, validation_data=valid_ds, epochs=500,
                    callbacks=[early_stopping_cb])

#%% Plotting History
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 600], ylim=[0, 1.5], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.show()


#%% Evaluating the final Model
def tx_helper_predict(pred, tx):
    mean = tx.mean_[0]
    var = tx.var_[0]
    return mean + np.sqrt(var) * pred

def tx_helper_mse(pred, tar, tx, seq_length):
    y_pred = tx_helper_predict(pred, tx)
    y_true = tar['rv5'][seq_length:].values
    return mean_squared_error(y_true, y_pred)

mse_train_scaled = tx_helper_mse(best_model.predict(train_ds).reshape(-1,1), rv4_train, train_stdscaler, seq_length)
print("Train Error: {}".format(mse_train_scaled))

mse_valid_scaled = tx_helper_mse(best_model.predict(valid_ds).reshape(-1,1), rv4_valid, train_stdscaler, seq_length)
print("Validation Error: {}".format(mse_valid_scaled))

mse_test_scaled = tx_helper_mse(best_model.predict(test_ds).reshape(-1,1), rv4_test, train_stdscaler, seq_length)
print("Test Error: {}".format(mse_test_scaled))

#%% Plotting Results
plt.plot(data['rv5'])
plt.plot(rv4_train.index[seq_length:], tx_helper_predict(best_model.predict(train_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv4_valid.index[seq_length:], tx_helper_predict(best_model.predict(valid_ds).reshape(-1,1), train_stdscaler))
plt.plot(rv4_test.index[seq_length:], tx_helper_predict(best_model.predict(test_ds).reshape(-1,1), train_stdscaler))
plt.title(" Best Single Layer RNN Multivar Model (with VIX Closing Prices)")
plt.xlim([pd.to_datetime("20080101"),pd.to_datetime("20180101")])
plt.show()

