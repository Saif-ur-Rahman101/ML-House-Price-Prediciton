# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:24:53 2024

@author: SAIF
"""

import pandas as pd
from sklearn import datasets
import seaborn as sns
# print(dir(datasets))

# boston_data = datasets.load_boston()

df = pd.read_csv("Boston_Dataset.csv")
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# sns.set(rc={'figure.figsize':(5,4)})
# sns.histplot(df['MEDV'], bins=30)

# sns.set(rc={'figure.figsize':(11,8)})
# corr_mat = df.corr().round(2)
# sns.heatmap(data=corr_mat, annot=True)

# # sns.set(rc={'figure.figsize':(5,4)})
# sns.scatterplot(x='RM', y='MEDV', data=df)

# sns.pairplot(df)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# X = df[['RM']]
# y = df[['MEDV']]

# X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# print(X_train.shape)

# model = LinearRegression()
# model.fit(X_train, y_train)
# y_test_predict =model.predict(X_test)


# print("MAE", metrics.mean_absolute_error(y_test, y_test_predict))
# print("MSE", metrics.mean_squared_error(y_test, y_test_predict))
# print("RMSE:", metrics.root_mean_squared_error(y_test, y_test_predict))
# print("R2:", metrics.r2_score(y_test, y_test_predict))

# X =df[['RM','LSTAT','CRIM']]
X = df
X = X.drop(['MEDV'], axis=1)

y = df[['MEDV']]

X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model = LinearRegression()
model.fit(X_train, y_train)
y_test_predict =model.predict(X_test)


print("MAE", metrics.mean_absolute_error(y_test, y_test_predict))
print("MSE", metrics.mean_squared_error(y_test, y_test_predict))
print("RMSE:", metrics.root_mean_squared_error(y_test, y_test_predict))
print("R2:", metrics.r2_score(y_test, y_test_predict))

model.intercept_
model.coef_

coeffcients = pd.DataFrame([X_train.columns,model.coef_.T]).T
coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
print(coeffcients)