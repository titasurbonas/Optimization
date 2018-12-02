
import sys
import numpy as np


class Perceptron(object):
	#The constructor of our class.
	def __init__(self, learningRate, nb_epoch, ):
		np.random.seed(2)
		self.learningRate = learningRate
		self.nb_epoch = nb_epoch
		self.errors_ = []
	def train_weights(self,matrix,labels,stop_early=True,verbose=True):
		#random_generator = np.random.RandomState(self.random_state)
		
		matrix_rows, matrix_columns = matrix.shape
		matrix_columns = matrix_columns+1
		self.weights = np.random.normal(loc=0.0, scale=0.001, size=matrix_columns)
		bias = np.ones(len(matrix))
		matrix = np.c_[ bias,matrix ]
		#Step 1 - Initialize all weights to 0 or a small random number  
		#weight[0] = the weight of the Bias Term
		for epoch in range(self.nb_epoch):
			cur_acc = self.accuracy(matrix, labels)
			
			#if epoch==self.nb_epoch-1:
				#print(" \nWeights: ",self.weights)
				#print("Accuracy: ",cur_acc)
			
			if cur_acc==1.0 and stop_early: break 

			
			for i in range(len(matrix)):
		
				prediction = self.predict(matrix[i]) # get predicted classificaion
				error = labels[i]-prediction         # get error from real classification
				
			
				for j in range(len(self.weights)):  # calculate new weight for each node
				
					self.weights[j] = self.weights[j]+(self.learningRate*error*matrix[i][j]) 
	
	def accuracy(self,matrix, labels):
		num_correct = 0.0
		preds       = []
		for i in range(len(matrix)):
			
			pred   = self.predict(matrix[i]) # get predicted classification
			preds.append(pred)
			if pred==labels[i]: num_correct+=1.0 
		return num_correct/float(len(matrix))

	def predict(self,inputs):
		activation=0.0
		for i,w in zip(inputs,self.weights):
			activation += i*w 
		return 1.0 if activation>=0.0 else 0.0