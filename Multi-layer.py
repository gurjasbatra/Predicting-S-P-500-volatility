"""
Created on Fri Apr 19 22:05:43 2024

@author: Gurjas Singh Batra
"""
#%%Importing all libraries
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import keras_tuner as kt

#%%Loading the data
# Load Realized Volatility Data
rv_data = pd.read_csv("OxfordManRealizedVolatilityIndices.csv")
rv_data.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
rv_data['Date'] = pd.to_datetime(rv_data['Date'].str[:10])
rv_data = rv_data[rv_data['Symbol'] == ".SPX"]

# Load Additional CSV Files
csv_files = [pd.read_csv(file, parse_dates=[0], na_values=['.']) for file in os.listdir() if file[-3:] == "csv"]
data = rv_data[['Date', 'rv5']].copy()

#%%Merging data
# Merge CSV Files
for csv in csv_files:
    data = data.merge(csv.rename(columns={'DATE':'Date'}))
print(data.head())

#%%Cleaning the data

# Make column names lowercase and save data to pickle
data.columns = [col.lower() for col in data.columns]
data.to_pickle("Project2Data.pkl")

# Time Series Data Preparation
model_vars = ['rv5', 'vixcls']
train_data = data.loc[:'2012', model_vars].copy().dropna()
valid_data = data.loc['2013':'2016', model_vars].copy().dropna()
test_data = data.loc['2017':, model_vars].copy().dropna()

# Normalize Data
scaler = StandardScaler()
train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns, index=train_data.index)
valid_data = pd.DataFrame(scaler.transform(valid_data), columns=valid_data.columns, index=valid_data.index)
test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns, index=test_data.index)

# Define Sequence Length
seq_length = 21

# Create TensorFlow datasets
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    train_data.to_numpy(),
    targets=train_data['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=False,
    seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    valid_data.to_numpy(),
    targets=valid_data['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)
test_ds = tf.keras.utils.timeseries_dataset_from_array(
    test_data.to_numpy(),
    targets=test_data['rv5'][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
)

# Define and Compile Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, input_shape=[seq_length, len(model_vars)]),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.LSTM(1)
])
model.compile(loss="mse", optimizer='adam', metrics=["mse"])

# Train Model
history = model.fit(train_ds, validation_data=valid_ds, epochs=500,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_mse", patience=100, restore_best_weights=True)])

# Plot Training History
pd.DataFrame(history.history).plot(figsize=(8, 5), xlim=[0, 250], ylim=[0, 1.5], grid=True, xlabel="Epoch",
                                   style=["r--", "r--.", "b-", "b-*"])
plt.show()

# Evaluate Model
def calculate_mse(pred, target, scaler, seq_length):
    y_pred = scaler.inverse_transform(pred)
    y_true = scaler.inverse_transform(target['rv5'][seq_length:].values.reshape(-1, 1))
    return mean_squared_error(y_true, y_pred)

mse_train = calculate_mse(model.predict(train_ds), train_data, scaler, seq_length)
mse_valid = calculate_mse(model.predict(valid_ds), valid_data, scaler, seq_length)
mse_test = calculate_mse(model.predict(test_ds), test_data, scaler, seq_length)
print("Train Error: {}\nValidation Error: {}\nTest Error: {}".format(mse_train, mse_valid, mse_test))

# Plot Results
plt.plot(data['rv5'])
plt.plot(train_data.index[seq_length:], scaler.inverse_transform(model.predict(train_ds).reshape(-1,1)))
plt.plot(valid_data.index[seq_length:], scaler.inverse_transform(model.predict(valid_ds).reshape(-1,1)))
plt.plot(test_data.index[seq_length:], scaler.inverse_transform(model.predict(test_ds).reshape(-1,1)))
plt.title("LSTM Model Multivariate 4")
plt.show()

# Hyperparameter Tuning (Bayesian Optimization)
def build_and_compile_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(seq_length, len(model_vars))))
    model.add(tf.keras.layers.Reshape([1, seq_length * len(model_vars)]))
    model.add(tf.keras.layers.Flatten())
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(tf.keras.layers.Dense(hp.Int(f"units_{i}", 10, 30, step=5), activation='relu'))
        model.add(tf.keras.layers.Dropout(hp.Float(f"dropout_{i}", 0.1, 0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mse"])
    return model

tuner = kt.BayesianOptimization(
    build_and_compile_model,
    objective=kt.Objective("mse", direction="min"),
    max_trials=100,
    directory="my_dir",
    project_name="keep_code_separate"
)
tuner.search(train_ds, validation_data=valid_ds, epochs=200, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_mse", patience=25, restore_best_weights=True)])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
               
# Retraining the Best Model
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(train_ds, validation_data=valid_ds, epochs=200, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_mse", patience=25, restore_best_weights=True)])

# Evaluate the Retrained Model
mse_train = calculate_mse(best_model.predict(train_ds), train_data, scaler, seq_length)
mse_valid = calculate_mse(best_model.predict(valid_ds), valid_data, scaler, seq_length)
mse_test = calculate_mse(best_model.predict(test_ds), test_data, scaler, seq_length)
print("Train Error: {}\nValidation Error: {}\nTest Error: {}".format(mse_train, mse_valid, mse_test))

# Plot Results
plt.plot(data['rv5'])
plt.plot(train_data.index[seq_length:], scaler.inverse_transform(best_model.predict(train_ds).reshape(-1,1)))
plt.plot(valid_data.index[seq_length:], scaler.inverse_transform(best_model.predict(valid_ds).reshape(-1,1)))
plt.plot(test_data.index[seq_length:], scaler.inverse_transform(best_model.predict(test_ds).reshape(-1,1)))
plt.title("LSTM Model Multivariate 4")
plt.show()

               