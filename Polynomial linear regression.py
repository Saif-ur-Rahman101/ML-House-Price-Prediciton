# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:31:09 2024

@author: SAIF
"""

import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("Boston_Dataset.csv")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X = df[['RM']]
y = df[['MEDV']]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size= 0.2, random_state=20)

from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_regression_model(degree):
  "Creates a polynomial regression model for the given degree"
  poly_features = PolynomialFeatures(degree=degree)
  
  # transform the features to higher degree features.
  X_train_poly = poly_features.fit_transform(X_train)
  
  # fit the transformed features to Linear Regression
  poly_model = LinearRegression()
  poly_model.fit(X_train_poly, Y_train)
  
  # predicting on training data-set
  y_train_predicted = poly_model.predict(X_train_poly)
  
  # predicting on test data-set
  y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
  
  # evaluating the model on training dataset
  rmse_train = np.sqrt(metrics.mean_squared_error(Y_train, y_train_predicted))
  r2_train = metrics.r2_score(Y_train, y_train_predicted)
  
  # evaluating the model on test dataset
  rmse_test = np.sqrt(metrics.mean_squared_error(Y_test, y_test_predict))
  r2_test = metrics.r2_score(Y_test, y_test_predict)
  
  print("The model performance for the training set")
  print("-------------------------------------------")
  print("RMSE of training set is {}".format(rmse_train))
  print("R2 score of training set is {}".format(r2_train))
  
  print("\n")
  
  print("The model performance for the test set")
  print("-------------------------------------------")
  print("RMSE of test set is {}".format(rmse_test))
  print("R2 score of test set is {}".format(r2_test))
  

print(create_polynomial_regression_model(3))