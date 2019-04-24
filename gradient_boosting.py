import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
data = pd.read_csv('Admission.csv')

X = data.drop('Chance of Admit ',axis=1)
y = data['Chance of Admit ']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

gbrt = GradientBoostingRegressor(n_estimators=10,learning_rate=1.0)
gbrt.fit(X_train,y_train)
y_pred = gbrt.predict(X_test)

metric = metrics.mean_squared_error(y_test,y_pred)
print(metric)

