import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
data = pd.read_csv('Admission.csv')

X = data.drop('Chance of Admit ',axis=1)
y = data['Chance of Admit ']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.50)

# Создаем три регрессионных дерева
tree1 = DecisionTreeRegressor()
tree1.fit(X_train,y_train)
prediction1 = tree1.predict(X_test).reshape(-1,1)
tree2 = DecisionTreeRegressor()
tree2.fit(X_train,y_train)
prediction2 = tree2.predict(X_test).reshape(-1,1)
tree3 = DecisionTreeRegressor()
tree3.fit(X_train,y_train)
prediction3 = tree3.predict(X_test).reshape(-1,1)

# Создаем новый обучающий набор для смесителя, где
# входными признаками являются предсказания предыдущих деревьев
new_data = np.append(prediction1,prediction2,axis=1)
new_data = np.append(new_data,prediction3,axis=1)
new_data = np.append(new_data,y_test.values.reshape(-1,1),axis=1)
new_data_x = new_data[:,:-1]
new_data_y = new_data[:,-1]

# Обучаем смеситель на новых данных
smesitel = DecisionTreeRegressor()
smesitel.fit(new_data_x,new_data_y)
# Предсказываем только по трем признакам потому что у нас только 3
# прогнозатора первого уровня!
y_pred = smesitel.predict(X_train.values[:,0:3])

metric = metrics.mean_squared_error(y_train,y_pred)
print(metric)
