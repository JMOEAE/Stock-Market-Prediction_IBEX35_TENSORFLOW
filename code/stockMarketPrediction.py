# coding=utf-8

###########################################################################
############  Stock-Market-Prediction_IBEX35_TENSORFLOW ###################
#######################################################################
###############               Authors:                      ###############  
######			       Juan Manuel López Torralba                    ######
######                                                               ###### 
######  OpenSource with Creative Commons Attribution-NonCommercial-  ######
######  ShareAlike license ubicated in GitHub throughout:            ######
## https://github.com/jmlopezt/Stock-Market-Prediction_IBEX35_TENSORFLOW ##
######                                                               ######
######  This work is licensed under the Creative Commons             ######
######  Attribution-NonCommercial-ShareAlike CC BY-NC-SA License.    ######
######  To view a copy of the license, visit                         ######
######  https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode  ######
###############                                             ###############
###########################################################################
###########################################################################

# Import modules
import tensorflow as tf 						# Tensorflow
import numpy as np 								# Numpy arrays
import os
import math
import pandas as pd 							# Read data from CSV file
from datetime import datetime, timedelta 	
import matplotlib.pyplot as plt 				# plot
from sklearn.preprocessing import MinMaxScaler 	# Estimator

#Function def

# Remove NaN,inf values
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

# weight def 
def weight_variable(shape):
	weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=1)
	initial = weight_initializer(shape)
	return tf.Variable(initial)

# bias def 
def bias_variable(shape):
	bias_initializer = tf.zeros_initializer()
	initial = bias_initializer(shape)
	return tf.Variable(initial)

# reLU def 
def reLU(x, weight, bias):
	aux = tf.add(tf.matmul(x, weight), bias)
	return tf.nn.relu(aux)

# Csv data
dataRaw = pd.read_csv('..\csvfiles/ibex35/ibex35.csv')
MarketName = 'IBEX-35'
#dataRaw = pd.read_csv('..\csvfiles/ibex35/endesa.csv')
#MarketName = 'ENDESA'
#dataRaw = pd.read_csv('..\csvfiles/ibex35/bbva.csv')
#MarketName = 'BBVA'
#dataRaw = pd.read_csv('..\csvfiles/ibex35/mapfre.csv')
#MarketName = 'MAPRE'

# Parse data

dateRawAux = dataRaw['Date'].values
dateTimeAux = np.zeros(dateRawAux.shape)

	#date strings to numeric value
for i, j in enumerate(dateRawAux):

	dateTimeAux[i] = datetime.strptime(j, '%Y-%m-%d').timestamp()
    #Add the newly parsed column to the dataframe
	dataRaw['Timestamp'] = dateTimeAux
	Timestamp = dateTimeAux

# Remove undesired columns by columns
dataFrame_redux = dataRaw.drop(['Date','Timestamp','Volume'],axis=1)

data = clean_dataset(dataFrame_redux)
data = dataFrame_redux.values 	# te lo da como numpy

print(np.any(np.isnan(data)))
print(data)

# plot data
plt.plot(data)
plt.xlabel("Points", fontsize = 10)
plt.ylabel("EUR (€)", fontsize = 15)
plt.title(MarketName + " Market Price", fontsize = 20)
plt.legend( ('Open', 'High', 'Low', 'Close', 'Adj Close'), loc = 'upper left')
plt.show()

# dimensions of data [n,p]
n = data.shape[0]
p = data.shape[1]

print(n)
print(p)

## Set training data

train_start = 0
train_end = int(np.floor(0.3*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

plt.plot(data[:,1],'y--',data_train[:,1],'r--',data_test[:,1],'b--')
plt.xlabel("Points", fontsize = 10)
plt.ylabel("EUR (€)", fontsize = 15)
plt.title(MarketName + " Market Price", fontsize = 20)
plt.legend( ('Original Set', 'Training Set', 'Test Set'), loc = 'upper right')
plt.show()

print(data_train)

# Scale data

print(np.any(np.isnan(data_train)))  # meter bloque try catch

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

print(data_train)

# Build x and y test & training set
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

print(data_train[:, 1:])
print(data_train[:, 0])

# Number of stocks in training data
n_stocks = X_train.shape[1]
print(n_stocks)


## Building the Artifitial Neural Network (ANN)

# Neurons
n_neurons_1 = 1024
n_neurons_2 = int(n_neurons_1/2)  #512
n_neurons_3 = int(n_neurons_2/2)  #256
n_neurons_4 = int(n_neurons_3/2)  #128

# Session
ann = tf.InteractiveSession()

# Placeholder
x = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
y = tf.placeholder(dtype=tf.float32, shape=[None])

# Hidden layers weights
W_hidden_1 = weight_variable([n_stocks, n_neurons_1])
W_hidden_2 = weight_variable([n_neurons_1, n_neurons_2])
W_hidden_3 = weight_variable([n_neurons_2, n_neurons_3])
W_hidden_4 = weight_variable([n_neurons_3, n_neurons_4])

# Hidden layers biases
bias_hidden_1 = bias_variable([n_neurons_1])
bias_hidden_2 = bias_variable([n_neurons_2])
bias_hidden_3 = bias_variable([n_neurons_3])
bias_hidden_4 = bias_variable([n_neurons_4])

# Output layer weights & biases
W_out = weight_variable([n_neurons_4, 1])
bias_out = bias_variable([1])

# Hidden layer activation function application
hidden_1 = reLU(x, W_hidden_1, bias_hidden_1)
hidden_2 = reLU(hidden_1, W_hidden_2, bias_hidden_2)
hidden_3 = reLU(hidden_2, W_hidden_3, bias_hidden_3)
hidden_4 = reLU(hidden_3, W_hidden_4, bias_hidden_4)

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
ann.run(tf.global_variables_initializer())

# Setup plot
plt.ion()   			# Set interactive mode ON, so matplotlib will not be blocking the window
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()
fig.canvas.draw()

# Fit neural net
batch_size = 64
mse_train = []
mse_test = []

# Run
epochs = 20
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        ann.run(opt, feed_dict={x: batch_x, y: batch_y})

        # Show progress
        if np.mod(i, 10) == 0:
            # MSE train and test
            mse_train.append(ann.run(mse, feed_dict={x: X_train, y: y_train}))
            mse_test.append(ann.run(mse, feed_dict={x: X_test, y: y_test}))

            mse_train_aux = mse_train[-1]
            mse_test_aux = mse_test[-1]

            print('MSE Train: ', mse_train_aux)
            print('MSE Test: ', mse_test_aux)

            # Prediction

            pred = ann.run(out, feed_dict={x: X_test})
            line2.set_ydata(pred)
            plt.title('Test Performance: '+'Epoch ' + str(e) + ', Batch ' + str(i))
            plt.xlabel('points')
            textvar = plt.text(x = 1000, y = 1.0, s = 'MSE Train: ' + str(mse_train_aux), fontsize = 8)
            textvar2 = plt.text(x = 1000, y = 0.9, s = 'MSE Test: '  + str(mse_test_aux), fontsize = 8)
            plt.legend( ('Original Set', 'Prediction'), loc = 'upper right')
            plt.pause(0.1)
            textvar.remove()
            textvar2.remove()
          