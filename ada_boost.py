import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
data = pd.read_csv('marks2.txt')

X = data.drop('0',axis=1)
y = data['0']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

model = AdaBoostClassifier(n_estimators=150,learning_rate=1)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(metrics.accuracy_score(y_test)