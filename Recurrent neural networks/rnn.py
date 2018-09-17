#predicting the google's stock price for january 2017 with the past 5 years of data using RNN 

#importing libraries for pre-processing of the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the training dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range (60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train = np.array(X_train)
y_train = np.array(y_train)


#reshaping
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#Building the rnn

#importing the essntial libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

#adding the first LSTM layer and the dropout regularisation
regressor.add(LSTM(units=50,return_sequences=True,input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(rate=0.2))

#adding another LSTM layer and the dropout regularisation
regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(rate=0.2))

#adding another LSTM layer and the dropout regularisation 
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(rate=0.2))

#adding last LSTM layer and the dropout reglarisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

#adding the output layer
regressor.add(Dense(units=1))


#compiling 
regressor.compile(optimizer='adam',loss='mean_squared_error')

#fitting the rnn 
regressor.fit(X_train,y_train,batch_size=32,epochs=100)

#Making the predictions and visualising the results

#getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#getting the predicted dataset of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range (60,80):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualising the results
plt.plot(real_stock_price,color='red',label = 'real google stock price')
plt.plot(predicted_stock_price,color='green',label='predicted google stock price')
plt.title('Google real v/s Google predicted stock price')
plt.xlabel('time')
plt.ylabel('Stock price')
plt.legend()
plt.show()

