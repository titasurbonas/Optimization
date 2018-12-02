import sys
import numpy as np


class Perceptron(object):
	#The constructor of our class.
	def __init__(self, learningRate, nb_epoch,MSE ):
		np.random.seed(3)
		self.learningRate = learningRate
		self.nb_epoch = nb_epoch
		self.MSE =MSE
	def train_weights(self,data,labels):		
		from sklearn.metrics import mean_squared_error
		self.weights = np.random.normal( size=data.shape[1])

		self.loss = []
		self.bias = 1

		for epoch in range(self.nb_epoch):
			if not self.MSE: 
				for d, l in zip(data , labels):
					error = l - self.predict(d)
					self.weights += self.learningRate*error*d
					self.bias += error 
					del error
			else:
				error = labels- self.predict(data)**2
				error = np.dot(error, data)
				error= np.mean(error, axis=0)
				self.weights += self.learningRate*error
				self.bias += np.sum(error) 
				del error
		
	def predict(self, data):
		#Heaviside function. Returns 1 or 0 
		return np.where((data.dot(self.weights)+self.bias)>=0.0 ,1 , 0)
