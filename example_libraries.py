import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

data = pd.read_csv('marks2.txt')
X = data.drop('0',axis=1)
y = data['0']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("Actual: %s. Predicted: %s" %
	(y_test.values,prediction))
print('Accuracy:',metrics.accuracy_score(y_test,prediction))