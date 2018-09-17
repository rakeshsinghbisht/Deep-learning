#finding the best parameters for our ANN

#importing essential libraries to pre-process the data 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataSet = pd.read_csv('E:\Deep_Learning_A_Z\Artificial_Neural_Networks\Churn_Modelling.csv')

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
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

#splitting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#applying feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()   
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing essential libraries for Artificial neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def buildClassifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_dim=11))
    classifier.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=buildClassifier)
#choosing which parameters are good for our model
parameters = {'batch_size':[10,25],
              'epochs':[100,250],
              'optimizer':['adam','rmsprop']
              }
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',cv=10)
grid_search = grid_search.fit(X_train,y_train)

#finding the best parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_