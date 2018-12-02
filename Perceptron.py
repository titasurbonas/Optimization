
import sys
import numpy as np


class Perceptron(object):
	#The constructor of our class.
	def __init__(self, learningRate, nb_epoch,MSE ):
		np.random.seed(2)
		self.learningRate = learningRate
		self.nb_epoch = nb_epoch
		self.MSE =MSE
	def train_weights(self,data,labels,stop_early=True,verbose=True):		
		from sklearn.metrics import mean_squared_error
		data_rows, data_columns = data.shape
		data_columns = data_columns+1
		self.weights = np.random.normal(loc=0.0, scale=0.001, size=data_columns)
		#print(self.weights)
		bias = np.ones(len(data))
		data = np.c_[ bias,data ]

		for epoch in range(self.nb_epoch):
			cur_acc = self.accuracy(data, labels)
			
			#if epoch==self.nb_epoch-1:
				#print(" \nWeights: ",self.weights)
				#print("Accuracy: ",cur_acc)
			
			if cur_acc==1.0 and stop_early: break 

			
			for i in range(len(data)):
		
				prediction = self.predict(data[i]) # get predicted classificaion
				if self.MSE:	error =mean_squared_error(labels[i], [prediction]) # get error from real classification
				else:			error = labels[i]-prediction         # get error from real classification
				for j in range(len(self.weights)):  # calculate new weight for each node
				
					self.weights[j] = self.weights[j]+(self.learningRate*error*data[i][j]) 
	
	def accuracy(self,data, labels):
		num_correct = 0.0
		preds       = []
		for i in range(len(data)):
			
			pred   = self.predict(data[i]) # get predicted classification
			preds.append(pred)
			if pred==labels[i]: num_correct+=1.0 
		return num_correct/float(len(data))

	def predict(self,inputs):
		activation=0.0
		for i,w in zip(inputs,self.weights):
			activation += i*w 
		return 1.0 if activation>=0.0 else 0.0