#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from matplotlib.dates import DateFormatter, YearLocator
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
!pip install keras-tuner
!pip install keras
from keras_tuner.tuners import RandomSearch
import os

#%% Load Data
data = pd.read_pickle("Project2Data.pkl")
data.set_index(['date'], inplace=True)

#%% Scale RV5 data by 100**2 for numerical stability

data['rv5'] = data['rv5'] * (100**2)

#%% Create Moving Average Variables

data['rv5_ma5'] = data['rv5'].rolling(5).mean()
data['rv5_ma21'] = data['rv5'].rolling(21).mean()

#%% Create the dependent variable
data['rv5_tp1'] = data['rv5'].shift(-1)

#%% Drop NAs
data.dropna(subset=['rv5_tp1', 'rv5_ma5', 'rv5_ma21','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3','vixcls'], inplace=True)

#%%Create the X Y varaible

#Univariate:lags of RV
X = data[['rv5','rv5_ma21','rv5_ma5']]

#Multivariate 1:lags of RV and interest rates
#X= data[['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3']]

#Multivariate 2:lags of RV and VIX
#X= data[['rv5','rv5_ma21','rv5_ma5','vixcls']]

#Multivariate 3:lags of RV,interest rate and VIX
#X = data[['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3','vixcls']]

y = data[['rv5_tp1']]

#%% Split Data Train-Validation-Test

# Training from start to end of 2012
# Validation 2013 to 2016
# Test 2017 to 2018 (only 6 months)

Xdata = ['rv5', 'rv5_ma5', 'rv5_ma21']
#Xdata = ['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3']
#Xdata = ['rv5','rv5_ma21','rv5_ma5','vixcls']
#Xdata = ['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3','vixcls']

Ydata = ['rv5_tp1']

X_train, y_train =  data.loc[:'2012', Xdata], data.loc[:'2012', Ydata]
X_valid, y_valid =  data.loc['2013':'2016', Xdata], data.loc['2013':'2016', Ydata]
X_test, y_test = data.loc['2017':, Xdata], data.loc['2017':, Ydata]


#%% Linear Model
lin_reg = LinearRegression()
lin_pipeline = make_pipeline(StandardScaler(), lin_reg)
lin_pipeline.fit(X_train, y_train)

# Training Error (MSE)
y_train_pred = lin_pipeline.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Training Error (MSE):", train_mse)

# Validation Error (MSE)
y_valid_pred = lin_pipeline.predict(X_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
print("Validation Error (MSE):", valid_mse)

# Test Error (MSE)
y_test_pred = lin_pipeline.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test Error (MSE):", test_mse)

# Plot results
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

plt.plot(data.index, data['rv5_tp1'], label='Actual')
plt.plot(X_train.index, y_train_pred, label='Train Pred')
plt.plot(X_valid.index, y_valid_pred, label='Valid Pred')
plt.plot(X_test.index, y_test_pred, label='Test Pred')
plt.title("Linear Model")
plt.legend()
plt.show()

#%% Normalize the data

train_stdscaler = StandardScaler()
X_train = pd.DataFrame(train_stdscaler.fit_transform(X_train),
                        columns=X_train.columns,
                        index=X_train.index)

X_valid = pd.DataFrame(train_stdscaler.transform(X_valid),
                        columns=X_valid.columns,
                        index=X_valid.index)

X_test = pd.DataFrame(train_stdscaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index)

#%% Define Sequence for DNN

seq_length = 21
tf.random.set_seed(42)  # extra code – ensures reproducibility

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_train.to_numpy(),
    targets=X_train[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=False,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_valid.to_numpy(),
    targets=X_valid[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_test.to_numpy(),
    targets=X_test[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

#%% Build the Deep Neural Network
tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_length, len(Xdata))),
    tf.keras.layers.Reshape([1,seq_length * len(Xdata)]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(1)
    
])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse", patience=25, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(loss="mse", optimizer=opt, metrics=["mse"])

history = model.fit(train_ds, validation_data=valid_ds, epochs=500,
                    callbacks=[early_stopping_cb])

#%% Plot history
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 250], ylim=[0, 2], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.title('DNN')
#plt.title('DNN with Interest Rate')
#plt.title('DNN with VIX')
#plt.title('DNN with Interest Rate and VIX')
plt.show()

#%%Evaluate the Model

def tx_helper_predict(pred, tx):
    mean = tx.mean_[0]
    var = tx.var_[0]
    return mean + np.sqrt(var) * pred

def tx_helper_mse(pred, tar, tx, seq_length):
    y_pred = tx_helper_predict(pred, tx)
    y_true = tar['rv5'][seq_length:].values
    return mean_squared_error(y_true, y_pred)

mse_train_scaled = tx_helper_mse(model.predict(train_ds).reshape(-1,1), X_train, train_stdscaler, seq_length)
print("Train Error: {}".format(mse_train_scaled))

mse_valid_scaled = tx_helper_mse(model.predict(valid_ds).reshape(-1,1), X_valid, train_stdscaler, seq_length)
print("Validation Error: {}".format(mse_valid_scaled))

mse_test_scaled = tx_helper_mse(model.predict(test_ds).reshape(-1,1), X_test, train_stdscaler, seq_length)
print("Test Error: {}".format(mse_test_scaled))

#%%Plot the result

plt.plot(data['rv5'])
plt.plot(X_train.index[seq_length:], tx_helper_predict(model.predict(train_ds).reshape(-1,1), train_stdscaler))
plt.plot(X_valid.index[seq_length:], tx_helper_predict(model.predict(valid_ds).reshape(-1,1), train_stdscaler))
plt.plot(X_test.index[seq_length:], tx_helper_predict(model.predict(test_ds).reshape(-1,1), train_stdscaler))
plt.title('DNN')
#plt.title('DNN with VIX')
#plt.title('DNN with Interest rate and VIX')
#plt.title('DNN with Interest Rate')
plt.show()





#%%Hypertuning

#Define tuner
def keras_code(hp_seq_length, hp_num_layers, hp_units, saving_path):

    seq_length = hp_seq_length
    tf.random.set_seed(42)  # extra code – ensures reproducibility

    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        X_train.to_numpy(),
        targets=X_train['rv5'][seq_length:],
        sequence_length=seq_length,
        batch_size=32,
        shuffle=False,
        seed=42
    )
    
    valid_ds = tf.keras.utils.timeseries_dataset_from_array(
        X_valid.to_numpy(),
        targets=X_valid['rv5'][seq_length:],
        sequence_length=seq_length,
        batch_size=32,
    )

    # Build model
   
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_length, len(Xdata))),
    tf.keras.layers.Reshape([1,seq_length * len(Xdata)]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
    
    ])
 
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

#%% Random Search
class MyTuner(RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        return keras_code(
            hp_seq_length=hp.Int("hp_seq_length", 1, 150, 5),
            hp_num_layers=hp.Int("hp_num_layers", 1, 5, 1),
            hp_units=hp.Int("hp_units", 10, 30, 5),
            saving_path=os.path.join("tmp", f"{trial.trial_id}.keras")
        )


tuner = MyTuner(
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="keep_code_separate",
)

#%% Run Tuner
tuner.search()
# Retraining the model
best_hp = tuner.get_best_hyperparameters()[0]
keras_code(**best_hp.values, saving_path="tmp/best_model_{}.keras".format(best_hp.values['hp_seq_length']))

#%% Load model
best_model = tf.keras.models.load_model("tmp/best_model_{}.keras".format(best_hp.values['hp_seq_length']))

#%% Retrain
seq_length = best_model.input_shape[1]
tf.random.set_seed(42)  # extra code – ensures reproducibility

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_train.to_numpy(),
    targets=X_train['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=False,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_valid.to_numpy(),
    targets=X_valid['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_test.to_numpy(),
    targets=X_test['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

#%% Retrain model to capture history

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse", patience=25, restore_best_weights=True)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

best_model.compile(loss="mse", optimizer=opt, metrics=["mse"])

history = best_model.fit(train_ds, validation_data=valid_ds, epochs=500,
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

mse_train_scaled = tx_helper_mse(best_model.predict(train_ds).reshape(-1,1), X_train, train_stdscaler, seq_length)
print("Train Error: {}".format(mse_train_scaled))

mse_valid_scaled = tx_helper_mse(best_model.predict(valid_ds).reshape(-1,1), X_valid, train_stdscaler, seq_length)
print("Validation Error: {}".format(mse_valid_scaled))

mse_test_scaled = tx_helper_mse(best_model.predict(test_ds).reshape(-1,1), X_test, train_stdscaler, seq_length)
print("Test Error: {}".format(mse_test_scaled))

#%% Plot Results
plt.plot(data['rv5'])
plt.plot(X_train.index[seq_length:], tx_helper_predict(best_model.predict(train_ds).reshape(-1,1), train_stdscaler))
plt.plot(X_valid.index[seq_length:], tx_helper_predict(best_model.predict(valid_ds).reshape(-1,1), train_stdscaler))
plt.plot(X_test.index[seq_length:], tx_helper_predict(best_model.predict(test_ds).reshape(-1,1), train_stdscaler))
plt.title('DNN')
#plt.title('DNN with VIX')
#plt.title('DNN with Interest rate and VIX')
#plt.title('DNN with Interest Rate')
plt.show()

# Print out the best hyperparameters
print("Best hyperparameters found:")
print(f"Sequence Length: {best_hp.get('hp_seq_length')}")
print(f"Number of Layers: {best_hp.get('hp_num_layers')}")
print(f"Units per Layer: {best_hp.get('hp_units')}")

#Sequence Length: 16
#Number of Layers: 5
#Units per Layer: 15





#%% After hypertuning, run the model again======================================================================

#Xdata = ['rv5', 'rv5_ma5', 'rv5_ma21']
#Xdata = ['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3']
#Xdata = ['rv5','rv5_ma21','rv5_ma5','vixcls']
Xdata = ['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3','vixcls']

Ydata = ['rv5_tp1']

X_train, y_train =  data.loc[:'2012', Xdata], data.loc[:'2012', Ydata]
X_valid, y_valid =  data.loc['2013':'2016', Xdata], data.loc['2013':'2016', Ydata]
X_test, y_test = data.loc['2017':, Xdata], data.loc['2017':, Ydata]



train_stdscaler = StandardScaler()
X_train = pd.DataFrame(train_stdscaler.fit_transform(X_train),
                        columns=X_train.columns,
                        index=X_train.index)

X_valid = pd.DataFrame(train_stdscaler.transform(X_valid),
                        columns=X_valid.columns,
                        index=X_valid.index)

X_test = pd.DataFrame(train_stdscaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index)



seq_length = 16
tf.random.set_seed(42)  # extra code – ensures reproducibility

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_train.to_numpy(),
    targets=X_train[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=False,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_valid.to_numpy(),
    targets=X_valid[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    X_test.to_numpy(),
    targets=X_test[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

# Build the Deep Neural Network
tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(seq_length, len(Xdata))),
    tf.keras.layers.Reshape([1,seq_length * len(Xdata)]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(1)
    
])

pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 250], ylim=[0, 1.5], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.show()

def tx_helper_predict(pred, tx):
    mean = tx.mean_[0]
    var = tx.var_[0]
    return mean + np.sqrt(var) * pred

def tx_helper_mse(pred, tar, tx, seq_length):
    y_pred = tx_helper_predict(pred, tx)
    y_true = tar['rv5'][seq_length:].values
    return mean_squared_error(y_true, y_pred)

mse_train_scaled = tx_helper_mse(model.predict(train_ds).reshape(-1,1), X_train, train_stdscaler, seq_length)
print("Train Error: {}".format(mse_train_scaled))

mse_valid_scaled = tx_helper_mse(model.predict(valid_ds).reshape(-1,1), X_valid, train_stdscaler, seq_length)
print("Validation Error: {}".format(mse_valid_scaled))

mse_test_scaled = tx_helper_mse(model.predict(test_ds).reshape(-1,1), X_test, train_stdscaler, seq_length)
print("Test Error: {}".format(mse_test_scaled))

# Plot Results
plt.plot(data['rv5'])
plt.plot(X_train.index[seq_length:], tx_helper_predict(model.predict(train_ds).reshape(-1,1), train_stdscaler))
plt.plot(X_valid.index[seq_length:], tx_helper_predict(model.predict(valid_ds).reshape(-1,1), train_stdscaler))
plt.plot(X_test.index[seq_length:], tx_helper_predict(model.predict(test_ds).reshape(-1,1), train_stdscaler))
#plt.title('DNN')
#plt.title('DNN with VIX')
plt.title('DNN with Interest rate and VIX')
#lt.title('DNN with Interest Rate')
plt.show()