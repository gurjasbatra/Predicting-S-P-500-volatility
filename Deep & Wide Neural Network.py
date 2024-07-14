import numpy as np
import pandas as pd
import os
print(os.getcwd())

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from sklearn.base import BaseEstimator, RegressorMixin
from matplotlib.dates import DateFormatter, YearLocator
import matplotlib.dates as mdates
from tensorflow.keras.optimizers import Adam, RMSprop

#Data Preprocessing
DGS1 = pd.read_csv("DGS1.csv")
DGS2 = pd.read_csv("DGS2.csv")
DGS10 = pd.read_csv("DGS10.csv")
DGS30 = pd.read_csv("DGS30.csv")
DTB3 = pd.read_csv("DTB3.csv")
VIXCLS = pd.read_csv("VIXCLS.csv")
DGS1['DATE'] = pd.to_datetime(DGS1['DATE'])
DGS1 = DGS1[(DGS1['DATE'] >= '2000-01-03') & (DGS1['DATE'] <= '2018-06-27')]
DGS1.set_index('DATE', inplace=True)
DGS2['DATE'] = pd.to_datetime(DGS2['DATE'])
DGS2 = DGS2[(DGS2['DATE'] >= '2000-01-03') & (DGS2['DATE'] <= '2018-06-27')]
DGS2.set_index('DATE', inplace=True)
DGS10['DATE'] = pd.to_datetime(DGS10['DATE'])
DGS10 = DGS10[(DGS10['DATE'] >= '2000-01-03') & (DGS10['DATE'] <= '2018-06-27')]
DGS10.set_index('DATE', inplace=True)
DGS30['DATE'] = pd.to_datetime(DGS30['DATE'])
DGS30 = DGS30[(DGS30['DATE'] >= '2000-01-03') & (DGS30['DATE'] <= '2018-06-27')]
DGS30.set_index('DATE', inplace=True)
DTB3['DATE'] = pd.to_datetime(DTB3['DATE'])
DTB3 = DTB3[(DTB3['DATE'] >= '2000-01-03') & (DTB3['DATE'] <= '2018-06-27')]
DTB3.set_index('DATE', inplace=True)
VIXCLS['DATE'] = pd.to_datetime(VIXCLS['DATE'])
VIXCLS = VIXCLS[(VIXCLS['DATE'] >= '2000-01-03') & (VIXCLS['DATE'] <= '2018-06-27')]
VIXCLS.set_index('DATE', inplace=True)
merged_df = DGS1.join([DGS2, DGS10, DGS30, DTB3, VIXCLS], how='outer')
print(merged_df)

#NA Fillin
def fillna(df):
  df = df.replace('.', pd.NA)
  f = df.fillna(method='ffill')
  b = df.fillna(method='bfill')
  df = (f.astype(float) + b.astype(float)) / 2
  return df
filled_df = fillna(merged_df)
print(filled_df)


#%% Data Preprocessing
df = pd.read_csv("oxfordmanrealizedvolatilityindices.csv")

df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], errors='coerce')
df.set_index('Unnamed: 0', inplace=True)
df.index = pd.to_datetime(df.index, utc=True).date
df.index.name = 'DATE'


merged_df = pd.merge(df, filled_df, left_index=True, right_index=True, how='left')
merged_df.drop(['open_time', 'close_time'], axis=1, inplace=True)

dummy_symbols = pd.get_dummies(df['Symbol'], prefix='Symbol')
df = pd.concat([df, dummy_symbols], axis=1)
df.drop('Symbol', axis=1, inplace=True)

Symbol = merged_df['Symbol']
selected_columns = ['rv5_ss', 'bv', 'rk_parzen', 'rk_twoscale', 'bv_ss', 'rk_th2', 'medrv', 'rv10_ss', 'rsv_ss', 'rv10', 'rsv', 'rv5']
Y = merged_df[selected_columns]
Y = pd.concat([Y, Symbol], axis=1)
merged_df.drop(selected_columns, axis=1, inplace=True)
X = merged_df
X

#%% Load Data
data = pd.read_pickle("Project2Data.pkl")
data.set_index(['date'], inplace=True)

print(data.head(),data.shape)

#%% Scale RV5 data by 100**2 for numerical stability
data['rv5'] = data['rv5'] * (100**2)

#%%Create Moving Average Variables

data['rv5_ma5'] = data['rv5'].rolling(5).mean()
data['rv5_ma21'] = data['rv5'].rolling(21).mean()
data['rv5_tp1'] = data['rv5'].shift(-1)

#%% Drop NAs
data.dropna(subset=['rv5_tp1', 'rv5_ma5', 'rv5_ma21'], inplace=True)

#%% Create the X Y varaible
# Without external data
X = data[['rv5','rv5_ma21','rv5_ma5']]
# With external data
X = data[['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3','vixcls']]
y = data['rv5_tp1']

#%% Split Data Train-Validation-Test
# Training from start to end of 2012
# Validation 2013 to 2016
# Test 2017 to 2018 (only 6 months)

Xdata = ['rv5','rv5_ma21','rv5_ma5']
# Xdata = ['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3','vixcls']

Ydata = ['rv5_tp1']

X_train, y_train =  data.loc[:'2012', Xdata], data.loc[:'2012', Ydata]
X_valid, y_valid =  data.loc['2013':'2016', Xdata], data.loc['2013':'2016', Ydata]
X_test, y_test = data.loc['2017':, Xdata], data.loc['2017':, Ydata]
print(X_train.shape, X_valid.shape, X_test.shape)
date= data.loc['2017':,Xdata].index
print(date.shape)

#%% Linear regression
lin_reg = LinearRegression()
lin_pipeline = make_pipeline(StandardScaler(), lin_reg)
lin_pipeline.fit(X_train, y_train)

#Training Error (MSE)
y_train_pred = lin_pipeline.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Training Error (MSE):", train_mse)

#Validation Error (MSE):
y_valid_pred = lin_pipeline.predict(X_valid)
valid_mse = mean_squared_error(y_valid, y_valid_pred)
print("Validation Error (MSE):", valid_mse)

#Test Error (MSE)
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
# First use the fit method to fit the training set
train_stdscaler.fit(X_train)

# Then use the transform method to transform all data sets
X_train = train_stdscaler.transform(X_train)                    
X_valid = train_stdscaler.transform(X_valid)              
X_test = train_stdscaler.transform(X_test)
print(X_train.shape)
print(X_train[:5,:])
# For y values, normalization is generally not needed unless you have a specific reason. The same approach can be used if needed:
# 1. Fit the scaler on the y_train
y_scaler = StandardScaler().fit(y_train)
# 2. Transform y datasets
y_train = y_scaler.transform(y_train)
y_valid = y_scaler.transform(y_valid)
y_test = y_scaler.transform(y_test)
print(y_train.shape)



#%% WideNNTimeSeries Model
#Training 
Xdata = ['rv5','rv5_ma21','rv5_ma5']
Ydata = ['rv5_tp1']

X_train, y_train =  data.loc[:'2012', Xdata], data.loc[:'2012', Ydata]
X_valid, y_valid =  data.loc['2013':'2016', Xdata], data.loc['2013':'2016', Ydata]
X_test, y_test = data.loc['2017':, Xdata], data.loc['2017':, Ydata]

#%% Define Sequence
seq_length = 21
tf.random.set_seed(42)

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
# Define the deep part
import torch
import torch.nn as nn
import torch.optim as optim

class DeepAndWideNN(nn.Module):
    def __init__(self, input_size, hidden_layers, wide_size, output_size):
        super(DeepAndWideNN, self).__init__()

        self.deep_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        input_layer_size = input_size
        for layer_size in hidden_layers:
            self.deep_layers.append(nn.Linear(input_layer_size, layer_size))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(self.dropout)
            input_layer_size = layer_size

        self.cross_layer = nn.Linear(input_size, wide_size)

        self.output_layer = nn.Linear(input_layer_size + wide_size, output_size)

    def forward(self, x):
        for layer in self.deep_layers:
            x = layer(x)

        cross_output = self.cross_layer(x)

        combined_output = torch.cat((x, cross_output), dim=1)

        output = self.output_layer(combined_output)
        return output
    
dropout_rate = 0.1
input_size = X_train.shape[1]
hidden_layers = [128, 64, 3]
wide_size = 64
output_size = 1

model = DeepAndWideNN(input_size, hidden_layers, wide_size, output_size)

criterion = nn.MSELoss()
l2_regularization = 0.01
optimizer = optim.Adam(model.parameters(), lr=1e-5,weight_decay=l2_regularization)

# Training
y_min = np.min([y_train.min(), y_valid.min(), y_test.min()])
y_max = np.max([y_train.max(), y_valid.max(), y_test.max()])
y_train = (y_train - y_min) / (y_max - y_min)
y_val = (y_valid - y_min) / (y_max - y_min)
y_test = (y_test - y_min) / (y_max - y_min)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.to_numpy())
y_train_tensor = torch.FloatTensor(y_train.to_numpy())
X_val_tensor = torch.FloatTensor(X_valid.to_numpy())
y_val_tensor = torch.FloatTensor(y_val.to_numpy())
X_test_tensor = torch.FloatTensor(X_test.to_numpy())
y_test_tensor = torch.FloatTensor(y_test.to_numpy())

print(X_train.shape, y_train.shape)


train_losses = []
num_epochs = 100
batch_size = 30
for epoch in range(num_epochs):
    epoch_losses = []
    
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]
      
        outputs  = model(inputs)
        

        loss = criterion(outputs, labels)
        epoch_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(epoch_loss)
  


    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")

import matplotlib.pyplot as plt
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, 'b', label='Training loss')
# plt.plot(epochs,val_losses, 'r', label='Validation loss')
plt.title('Train loss over epochs')

plt.xlabel('Epochs')

plt.ylabel('Loss')
plt.legend()
plt.show()

#Validation Loss
num_epochs = 100
batch_size = 30
val_losses = []
for epoch in range(num_epochs):
    epoch_losses1 = []
    for i in range(0, len(X_val_tensor), batch_size):
        inputs1 = X_val_tensor[i:i+batch_size]
        labels1 = y_val_tensor[i:i+batch_size]

        outputs1 = model(inputs1)

        loss = criterion(outputs1, labels1)
        epoch_losses1.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_losses1 = sum(epoch_losses1) / len(epoch_losses1)
    val_losses.append(epoch_losses1)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_losses1}")

import matplotlib.pyplot as plt
epochs = range(1, num_epochs + 1)
# plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.plot(epochs,val_losses, 'r', label='Validation loss')
plt.title('Validation loss over epochs')
plt.xlabel('Epochs')

plt.ylabel('Loss')
plt.legend()
plt.show()

# Test loss
num_epochs = 100
batch_size = 30
test_losses = []
for epoch in range(num_epochs):
    epoch_losses1 = []
    for i in range(0, len(X_test_tensor), batch_size):
        inputs1 = X_test_tensor[i:i+batch_size]
        labels1 = y_test_tensor[i:i+batch_size]

        outputs1 = model(inputs1)

        loss = criterion(outputs1, labels1)
        epoch_losses1.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_losses1 = sum(epoch_losses1) / len(epoch_losses1)
    test_losses.append(epoch_losses1)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_losses1}")
    
import matplotlib.pyplot as plt
epochs = range(1, num_epochs + 1)
# plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.plot(epochs,test_losses, 'r', label='Test loss')
plt.title('Test loss over epochs')
plt.xlabel('Epochs')

plt.ylabel('Loss')
plt.legend()
plt.show()

with torch.no_grad():
    predictions = model(X_test_tensor)

import matplotlib.pyplot as plt

date= data.loc['2017':,Xdata].index
# plt(predictions)
plt.plot(date, y_test_tensor, label='True Values', color='blue', alpha=0.2)
plt.plot(date, predictions, label='Predicted Values', color='red', alpha=0.2, linestyle='--')

plt.legend()
plt.title('True Values vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
#Evaluate the model
loss_function = torch.nn.MSELoss()

# Function to evaluate the model
def evaluate_model(model, X_data, y_data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Instructs PyTorch to not calculate gradients
        predictions = model(X_data)
        mse = loss_function(predictions, y_data)
        rmse = torch.sqrt(mse)
    return mse.item(), rmse.item()

mse_test, rmse_test = evaluate_model(model, X_test_tensor, y_test_tensor)
mse_train, rmse_train = evaluate_model(model, X_train_tensor, y_train_tensor)
mse_valid, rmse_valid = evaluate_model(model, X_val_tensor, y_val_tensor)

print("Test MSE:", mse_test)
print("Training MSE:", mse_train)
print("Validation MSE:", mse_valid)








# In[6]:


import matplotlib.pyplot as plt
epochs = range(1, num_epochs + 1)
# plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.plot(epochs,test_losses, 'r', label='Test loss')
plt.title('Test loss over epochs')
plt.xlabel('Epochs')

plt.ylabel('Loss')
plt.legend()
plt.show()

#%% WideNNTimeSeries with single regressor Vixcls Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data=pd.read_pickle('Project2Data.pkl')
data.set_index(['date'], inplace=True)
model_mutiple_var=['rv5','vixcls']
d_train = data.loc[:'2012', model_mutiple_var ].copy().dropna()
d_valid = data.loc['2013':'2016', model_mutiple_var ].copy().dropna()
d_test = data.loc['2017':, model_mutiple_var ].copy().dropna()
#%% Normalizing the data
train_stdscaler = StandardScaler()
d_train['rv5'] = train_stdscaler.fit_transform(d_train[['rv5']])

d_valid['rv5'] = train_stdscaler.transform(d_valid[['rv5']])
d_test['rv5'] = train_stdscaler.transform(d_test[['rv5']])


X_train = d_train[['rv5']].values  
X_valid = d_valid[['rv5']].values  
X_test = d_test[['rv5']].values    
y_train = d_train['vixcls'].values
y_valid = d_valid['vixcls'].values
y_test = d_test['vixcls'].values
print(d_train.shape, d_valid.shape, d_test.shape)
y_min = min(y_train.min(), y_valid.min(), y_test.min())
y_max = max(y_train.max(), y_valid.max(), y_test.max())
y_train = (y_train - y_min) / (y_max - y_min)
y_val = (y_valid - y_min) / (y_max - y_min)
y_test = (y_test - y_min) / (y_max - y_min)

x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
print(x_train_tensor.shape, y_train_tensor.shape)
print(f"x_train_tensor: {x_train_tensor.shape}")


# Check data shape
print(f"x_train shape: {x_train_tensor.shape}, Y_train shape: {y_train_tensor.shape}")
print(f"x_valid shape: {x_valid_tensor.shape}, Y_valid shape: {y_valid_tensor.shape}")
print(f"x_test shape: {x_test_tensor.shape}, Y_test shape: {y_test_tensor.shape}")

# Confirm that the data does not have NaN values
print(f"x_train NaNs: {torch.isnan(x_train_tensor).any()}")
print(f"y_train NaNs: {torch.isnan(y_train_tensor).any()}")


import torch
import torch.nn as nn
import torch.optim as optim

class DeepAndWideNN(nn.Module):
    def __init__(self, input_size, hidden_layers, wide_size, output_size):
        super(DeepAndWideNN, self).__init__()

        self.deep_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        input_layer_size = input_size
        for layer_size in hidden_layers:
            self.deep_layers.append(nn.Linear(input_layer_size, layer_size))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(self.dropout)
            input_layer_size = layer_size

        self.cross_layer = nn.Linear(input_size, wide_size)

        self.output_layer = nn.Linear(input_layer_size + wide_size, output_size)

    def forward(self, x):
        for layer in self.deep_layers:
            x = layer(x)

        cross_output = self.cross_layer(x)

        combined_output = torch.cat((x, cross_output), dim=1)

        output = self.output_layer(combined_output)
        return output

dropout_rate = 0.1
input_size = 1  # Since we have only one feature
hidden_layers = [64, 32,1]  # Simplified network
wide_size = 64
output_size = 1

model = DeepAndWideNN(input_size, hidden_layers, wide_size, output_size)
criterion = nn.MSELoss()
l2_regularization = 0.01
optimizer = optim.Adam(model.parameters(), lr=1e-5,weight_decay=l2_regularization)


num_epochs = 100
batch_size = 10

train_losses = []
for epoch in range(num_epochs):
    epoch_losses = []
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = x_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]


        outputs = model(inputs)

        loss = criterion(outputs, labels.unsqueeze(1))
        epoch_losses.append(loss.item())
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")

import matplotlib.pyplot as plt
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.title('Training loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

num_epochs = 100
batch_size = 10
val_losses = []
for epoch in range(num_epochs):
    epoch_losses1 = []
    for i in range(0, len(x_valid_tensor), batch_size):
        inputs1 = x_valid_tensor[i:i+batch_size]
        labels1 = y_val_tensor[i:i+batch_size]

        outputs1 = model(inputs1)

        loss = criterion(outputs1, labels1)
        epoch_losses1.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_losses1 = sum(epoch_losses1) / len(epoch_losses1)
    val_losses.append(epoch_losses1)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_losses1}")


import matplotlib.pyplot as plt
epochs = range(1, num_epochs + 1)
# plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.plot(epochs,val_losses, 'r', label='Validation loss')
plt.title('Validation loss over epochs')
plt.xlabel('Epochs')

plt.ylabel('Loss')
plt.legend()
plt.show()


num_epochs = 100
batch_size = 30

test_losses = []
for epoch in range(num_epochs):
    epoch_losses1 = []
    for i in range(0, len(x_test_tensor), batch_size):
        inputs1 = x_test_tensor[i:i+batch_size]
        labels1 = y_test_tensor[i:i+batch_size]

        outputs1 = model(inputs1)

        loss = criterion(outputs1, labels1)
        epoch_losses1.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_losses1 = sum(epoch_losses1) / len(epoch_losses1)
    test_losses.append(epoch_losses1)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_losses1}")


import matplotlib.pyplot as plt
epochs = range(1, num_epochs + 1)
# plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.plot(epochs,val_losses, 'r', label='Test loss')
plt.title('Test loss over epochs')
plt.xlabel('Epochs')

plt.ylabel('Loss')
plt.legend()
plt.show()



# Evaluate the model
print(model)
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(3, 128),  # Adjusting the input layer to accept 3 features
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

loss_function = torch.nn.MSELoss()

mse_test, rmse_test = evaluate_model(model, X_test_tensor, y_test_tensor)
mse_train, rmse_train = evaluate_model(model, X_train_tensor, y_train_tensor)
mse_valid, rmse_valid = evaluate_model(model, X_val_tensor, y_val_tensor)

print("Test MSE:", mse_test)
print("Training MSE:", mse_train)
print("Validation MSE:", mse_valid)

print("Train shape:", X_train_tensor.shape)
print("Validation shape:", X_val_tensor.shape)
print("Test shape:", X_test_tensor.shape)


with torch.no_grad():
    predictions = model(x_test_tensor)

import matplotlib.pyplot as plt

date= data.loc['2017':,model_mutiple_var].index
# plt(predictions)
plt.plot(date, y_test_tensor+0.1, label='True Values', color='blue', alpha=0.2)
plt.plot(date, predictions, label='Predicted Values', color='red', alpha=0.2, linestyle='--')

plt.legend()
plt.title('True Values vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Hypertuning Parameter
from tensorflow import keras
from keras_tuner import RandomSearch


data['rv5'] = data['rv5'] * (100**2)
#Create Moving Average Variables
data['rv5_ma5'] = data['rv5'].rolling(5).mean()
data['rv5_ma21'] = data['rv5'].rolling(21).mean()
data['rv5_tp1'] = data['rv5'].shift(-1)

data.dropna(subset=['rv5_tp1', 'rv5_ma5', 'rv5_ma21'], inplace=True)

# Without external data
X = data[['rv5','rv5_ma21','rv5_ma5']]
# With external data
X = data[['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3','vixcls']]
y = data['rv5_tp1']

# Training from start to end of 2012
# Validation 2013 to 2016
# Test 2017 to 2018 (only 6 months)

Xdata = ['rv5','rv5_ma21','rv5_ma5']
# Xdata = ['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3','vixcls']

Ydata = ['rv5_tp1']

X_train, y_train =  data.loc[:'2012', Xdata], data.loc[:'2012', Ydata]
X_valid, y_valid =  data.loc['2013':'2016', Xdata], data.loc['2013':'2016', Ydata]
X_test, y_test = data.loc['2017':, Xdata], data.loc['2017':, Ydata]

np.random.seed(42)
tf.random.set_seed(42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

def build_model(hp):
    input_shape = X_train_scaled.shape[1:]
    neurons = hp.Int('units', min_value=50, max_value=150, step=25)
    deep_layers = hp.Int('deep_layers', 1, 5)
    wide_layers = hp.Int('wide_layers', 1, 5)
    inputs = keras.layers.Input(shape=input_shape)
    deep_layer = inputs
    for _ in range(deep_layers):
        deep_layer = keras.layers.Dense(neurons, activation='relu')(deep_layer if _ else inputs)
    wide_layer = inputs
    for _ in range(wide_layers):
        wide_layer = keras.layers.Dense(neurons, activation='relu')(wide_layer if _ else inputs)
    concat_layer = keras.layers.concatenate([wide_layer, deep_layer])
    output = keras.layers.Dense(1)(concat_layer)
    model = keras.Model(inputs=inputs, outputs=output)
    optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'RootMeanSquaredError'])
    return model

tuner = RandomSearch(build_model, objective='val_loss', max_trials=10, directory='model_tuning', project_name='wdnn_tuning')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
tuner.search(X_train_scaled, y_train, epochs=100, validation_data=(X_valid_scaled, y_valid), callbacks=[early_stopping])
# Best hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print("Best hyperparameters:", best_hp.values)

#%% Using optimal parameter
Xdata = ['rv5','rv5_ma21','rv5_ma5']
# Xdata = ['rv5','rv5_ma21','rv5_ma5','dgs1', 'dgs2', 'dgs10', 'dgs30', 'dtb3','vixcls']

Ydata = ['rv5_tp1']

X_train, y_train =  data.loc[:'2012', Xdata], data.loc[:'2012', Ydata]
X_valid, y_valid =  data.loc['2013':'2016', Xdata], data.loc['2013':'2016', Ydata]
X_test, y_test = data.loc['2017':, Xdata], data.loc['2017':, Ydata]

np.random.seed(42)
tf.random.set_seed(42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Define the wide and deep neural network model
input_shape = X_train_scaled.shape[1:]
neurons = 75
deep_layers = 2
wide_layers = 4

inputs = tf.keras.layers.Input(shape=input_shape)
deep_layer = inputs
for _ in range(deep_layers):
    deep_layer = tf.keras.layers.Dense(neurons, activation='relu')(deep_layer)

# Define the wide part of the model
wide_layer = inputs
for _ in range(wide_layers):
    wide_layer = tf.keras.layers.Dense(neurons, activation='relu')(wide_layer)
    
concat_layer = tf.keras.layers.concatenate([wide_layer, deep_layer])
output_layer = tf.keras.layers.Dense(1)(concat_layer)
model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.1,callbacks=[early_stopping])
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 10], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.show()

mse_test, rmse_test = model.evaluate(X_test_scaled, y_test)
y_pred = model.predict(X_test_scaled)
y_pred_valid = model.predict(X_valid_scaled)

calculated_rmse = np.sqrt(mse_test)
print("Calculated RMSE:", calculated_rmse)
print("MSE:", mse_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

mse_test, rmse_test = model.evaluate(X_test_scaled, y_test)
mse_train, rmse_train = model.evaluate(X_train_scaled, y_train)
mse_valid, rmse_valid = model.evaluate(X_valid_scaled, y_valid)

print("Test MSE:", mse_test)
print("Training MSE:", mse_train)
print("Validation MSE:", mse_valid)

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

#plotting true vlaue & predicted value
plt.figure(figsize=(12, 7))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs. True')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.legend()
plt.show()

#scatter plot
plt.figure(figsize=(12, 7))
plt.scatter(y_valid, y_pred_valid, color='blue', label='Predicted vs. True Values (Validation)')
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. True Values (Validation)')
plt.legend()
plt.grid(True)
plt.show()

#Line Plot
plt.figure(figsize=(12, 7))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
