# Multiple Linear Regression

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('50_Startups.csv')

# Importing the dataset

dataset = pd.read_csv('50_Startups.csv')
dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)

# Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


import pickle
file = open('multilinear.pkl','wb')
pickle.dump(regressor,file)

model = pickle.load(open('multilinear.pkl','rb'))
print(model.predict([[0,0,1,165346.3,136895.8,471784.9]]))
