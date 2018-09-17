#Predicting which customers are likely to leave the bank using ANN

#importing essential libraries to pre-process the data 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataSet = pd.read_csv("Churn_Modelling.csv")

X = dataSet.iloc[:,3:13].values
y = dataSet.iloc[:,13].values

#handling categorical data 
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

#creating dummy matrix for the categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#splitting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

#applying feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing essential libraries for Artificial neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#adding the input layer and the first hidden layer
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu", input_dim=11))

#adding second hidden layer
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))

#adding output layer
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))

#compile the ann
classifier.compile(optimizer = 'adam',loss ='binary_crossentropy',metrics=['accuracy'])

#fitting the ann to training set
classifier.fit(X_train,y_train,batch_size = 10,epochs = 100)

#predicting the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)