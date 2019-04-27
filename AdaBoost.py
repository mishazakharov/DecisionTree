import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class AdaBoost():
	def __init__(self,M):
		# Number of base-learners!
		self.M = M

	def fit(self,X,Y):
		self.models = []
		self.alphas = []
		N = len(X)
		W = np.ones(N)/N

		for m in range(self.M):
			tree = DecisionTreeClassifier(max_depth=1)
			tree.fit(X,Y,sample_weight=W)
			P = tree.predict(X)

			err = W.dot(P != Y)
			alpha = 0.5 * (np.log(1-err) - np.log(err))
			# Vecotrized form
			W = W*np.exp(-alpha*Y*P)
			# Normalized
			W = W/W.sum()

			self.models.append(tree)
			self.alphas.append(alpha)

	def predict(self,X):
		N = len(X)
		FX = np.zeros(N)
		for alpha,tree in zip(self.alphas,self.models):
			FX += alpha*tree.predict(X)
		return np.sign(FX), FX

	def score(self,X,Y):
		P,FX = self.predict(X)
		L = np.exp(-Y*FX).mean()
		return np.mean(P == Y), L


if __name__ == '__main__':

	data = pd.read_csv('marks2.txt')
	X = data.drop('0',axis=1).values
	Y = data['0'].values

	Y[Y == 0] = -1 # make the targets -1,+1
	Ntrain = int(0.8*len(X))
	Xtrain, Ytrain = X[:Ntrain],Y[:Ntrain]
	Xtest, Ytest = X[Ntrain:],Y[Ntrain:]

	T = 120
	train_errors = np.empty(T)
	test_losses = np.empty(T)
	test_errors = np.empty(T)
	for num_trees in range(T):
		if num_trees == 0:
			train_errors[num_trees] = None
			test_losses[num_trees] = None
			test_errors[num_trees] = None
			continue
		if num_trees%20 == 0:
			print(num_trees)
		model = AdaBoost(num_trees)
		model.fit(Xtrain,Ytrain)
		acc, loss = model.score(Xtest,Ytest)
		acc_train,bb = model.score(Xtrain,Ytrain)
		train_errors[num_trees] = 1 - acc_train
		test_errors[num_trees] = 1 - acc
		test_losses[num_trees] = loss

		if num_trees == T - 1:
			print('final train error:',1 - acc_train)
			print('final test error:',1 - acc)
