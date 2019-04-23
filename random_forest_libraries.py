import pandas as pd
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree	

'''
https://scikit-learn.org/stable/modules/generated/
sklearn.ensemble.RandomForestClassifier.html
'''

data = pd.read_csv('marks2.txt')
X = data.drop('0',axis=1)
y = data['0']
# Разделение данных на тренировочный и тестовый наборы
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
# Обучение и предсказание случайного леса
forest = RandomForestClassifier(n_estimators=500)
forest.fit(X_train,y_train)
y_pred = forest.predict(X_test)
# Обучение и предсказание дерева
tree = tree.DecisionTreeClassifier()
tree.fit(X_train,y_train)
y_pred1 = tree.predict(X_test)
# Точность дерева
metric1 = metrics.accuracy_score(y_test,y_pred1)
print('Tree - ',metric1)
# Точность случайного леса
metric = metrics.accuracy_score(y_test,y_pred)
print('Forest - ',metric)

# Создание случайного леса, используя другой класс(тоже самое!)
'''
forest = BaggingClassifier()
forest.fit(X_train,y_train)
y_pred = forest.predict(X_test)
'''
