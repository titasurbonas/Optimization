import numpy as np
from sklearn.metrics import mean_squared_error
class Perceptron(object):
	#The constructor of our class.
	def __init__(self, learningRate, nb_epoch,MSE ):
		np.random.seed(3)
		self.learningRate = learningRate
		self.nb_epoch = nb_epoch
		self.MSE =MSE
		self.error_= []
		self.acc_=[]
		self.bias = 1.0
		self.weights =np.array([])
	def train_weights(self,data,labels):		
		from sklearn.metrics import mean_squared_error
		if self.weights.size == 0:
			self.weights = np.random.normal( size=data.shape[1])
		BestWeights= np.array([])
		bestBias = 0
		for epoch in range(self.nb_epoch):
			if not self.MSE: 
				error = labels - np.array(self.predict(data)).reshape(-1,1)
				gradia= np.dot(data.T,error)
				self.weights += self.learningRate*gradia[:, 0].T
				self.bias += np.sum(error)*self.learningRate
				error = np.sum(error)
				"""
				error =0
				for d, l in zip(data , labels):
					SingelSampelerror = l - self.predict(d)
					self.weights += self.learningRate*SingelSampelerror*d
					self.bias += float(SingelSampelerror) 
					error += SingelSampelerror
				error *= -1
				"""
			#Can not be tested for MNIST data: MemoryError  error = np.dot(error, data)
			else:
				#predicts = np.array(self.predict(data))
				error= mean_squared_error(labels, np.array(self.predict(data)).reshape(-1,1) )
			#	predicts = predicts.reshape(-1,1)
				gradient = np.dot(data.T,error )
				gradient /= len(data)
				gradient *= self.learningRate
			
				self.weights  += gradient[:, 0].T
				self.bias -=  error*self.learningRate#(np.sum(-2*error)/len(data))* self.learningRate
				
				#self.error_=labels- self.predict(data)**2
				
			#Log learning progress
			#accurecy = np.mean(np.where(labels==self.predict(data),1,0))
		#	self.acc_.append(accurecy)
			self.error_.append(np.mean(error))
			#if error ==0 : break
			"""
			if accurecy ==1: break;
			if accurecy >= max(self.acc_):
				BestWeights = np.array(self.weights)
				bestBias = float(self.bias)
				SmallestError = float(error)
				BiggestAccuracy = float(accurecy)
			if epoch == self.nb_epoch-1 and BestWeights.size != 0:
				self.weights = BestWeights
				self.bias = bestBias
				self.acc_.append(BiggestAccuracy)
				self.error_.append(SmallestError)
			"""
			del error
	def predict(self, data):
		#Heaviside function. Returns 1 or 0 
		return np.where((data.dot(self.weights)+self.bias)>=0.0 ,1 , 0)
