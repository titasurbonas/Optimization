#!/usr/bin/env python

import pandas as pd
import numpy as np
import Neural_Network
from sklearn.model_selection import train_test_split

####################################################################
# Read the data from files 
orl_data = np.genfromtxt('./ORL_txt/orl_data.txt',delimiter='	')
orl_data = np.delete(orl_data, [400], axis=1).T
orl_data_labels = np.genfromtxt('./ORL_txt/orl_lbls.txt')
orl_data_labels = np.reshape(orl_data_labels, (-1, 1))

#MNIST_train_data =np.genfromtxt('./MNIST/mnist_train.csv',delimiter=',')
#MNIST_train_labels= MNIST_train_data[:,:1]
#MNIST_train_data = np.delete(MNIST_train_data, [0], axis=1) 

#MNIST_test_data = np.genfromtxt('./MNIST/mnist_test.csv',delimiter=',')
#MNIST_test_labels = MNIST_test_data[:,:1]
#MNIST_test_data = np.delete(MNIST_test_data, [0], axis=1) 

####################################################################
# Fit a PCA
def applyPCA(data):
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler
	f_transform = StandardScaler().fit_transform(data)
	pca = PCA(n_components=2 ,  whiten=True ).fit_transform(data)
	#pca.fit(data)
	# Project the data in 2D
	#pca.transform(data)
	#print(pca)
	return pca

####################################################################
# Display training data with labels and test data with predictions 
def displayData(data_2D, labels, test_data ,predicted,title):	#TODO fix displaying data
	from matplotlib import pyplot as plt
	labels = labels.transpose
	#predicted = predicted.values
	plt.figure(figsize=(10, 10)) 
	#plt.scatter(data_2D, data_2D[:1],marker="v",  c=labels[:])
	#plt.scatter(test_data[:,0], test_data[:,1],marker="x",  c=predicted[:])
	#plt.title(title)
	#plt.show()

####################################################################
# Nearest Centroid Classifier
def NearestCentroidClassifier(data, label):
	from sklearn.neighbors.nearest_centroid import NearestCentroid
	clf = NearestCentroid().fit(data, label) # label expected array gets 1D matrix future update
	return clf

####################################################################
# Nearest Neighbors Classifier
def  NearestNeighborsClassifier(data, label):
	from sklearn.neighbors import KNeighborsClassifier
	nnc = KNeighborsClassifier(n_neighbors=5)
	nnc.fit(data,label)# label expected array gets 1D matrix future update
	return nnc

####################################################################
# Data conversion to 2D
orl_data_2D = applyPCA(orl_data)
orlD_Train , orlD_Test , orlL_Train, orlL_Test  = train_test_split(orl_data_2D, orl_data_labels, test_size=0.3, random_state=0)

#MNIST_train_2D = applyPCA(MNIST_train_data)
#MNIST_test_2D = applyPCA(MNIST_test_data)
"""
####################################################################
# orl Data Nearest Centroid Classifier
classifierORL = NearestCentroidClassifier(orlD_Train,orlL_Train)

print("Nearest Centroid Classifier score for orl Data:")
print (classifierORL.score(orlD_Test,orlL_Test))

displayData(orlD_Train,orlL_Train,orlD_Test, classifierORL.predict(orlD_Test), "NearestCentroidClassifier")

####################################################################
#MNIST Data Nearest Neighbors Classifier
classifierMNIST = NearestCentroidClassifier(MNIST_train_2D,MNIST_train_labels)

print("Nearest Centroid Classifier score for MNIST Data:")
print (classifierMNIST.score(MNIST_test_2D,MNIST_test_labels))

displayData(MNIST_train_2D,MNIST_train_labels,MNIST_test_2D, classifierMNIST.predict(MNIST_test_2D), "NearestCentroidClassifier")

####################################################################
# orl Data Nearest Neighbors Classifier
classifierORL = NearestNeighborsClassifier(orlD_Train,orlL_Train)

print("Nearest Neighbors Classifier score for orl Data:")
print (classifierORL.score(orlD_Test,orlL_Test))

displayData(orlD_Train,orlL_Train,orlD_Test, classifierORL.predict(orlD_Test), "NearestNeighborsClassifier")

####################################################################
#MNIST Data Nearest Neighbors Classifier
classifierMNIST = NearestNeighborsClassifier(MNIST_train_2D,MNIST_train_labels)

print ("Nearest Neighbors Classifier score for MNIST Data:" )
print (classifierMNIST.score(MNIST_test_2D,MNIST_test_labels))

displayData(MNIST_train_2D,MNIST_train_labels,MNIST_test_2D, classifierMNIST.predict(MNIST_test_2D), "NearestNeighborsClassifier")
"""
####################################################################
#orl Data Nearest Neural_Network Backpropagation
NN = Neural_Network.Neural_Network(0.07,2,10,40)
data = orlD_Train/np.amax(orlD_Train, axis=0) # sacale data form -1 to 1
data_test = orlD_Test/np.amax(orlD_Test, axis=0)
label = np.zeros((np.size(orlL_Train,0),40))
for l in range(np.size(orlL_Train,0)):  
    index = int(orlL_Train[l])-1 # we dont have callas 0
    label[l,index] = 1 
for i in range(1000):
	NN.train(data,label)

print ("Neural_Network Backpropagation for orl_data: "+ str(NN.score(data_test,orlL_Test)) )

"""
####################################################################
#MNIST Data Nearest Neural_Network Backpropagation
NN = Neural_Network.Neural_Network(1,2,4,10)
data = MNIST_train_2D/np.amax(MNIST_train_2D, axis=0) # sacale data form -1 to 1
data_test = MNIST_test_2D/np.amax(MNIST_test_2D, axis=0)
label = np.zeros((np.size(MNIST_train_labels,0),10))
for l in range(np.size(MNIST_train_labels,0)):  
    index = int(MNIST_train_labels[l])
    label[l,index] = 1 
print(data)
print(np.size(label,0))

NN.train(data,label)

print (" Neural_Network Backpropagation for MNIST Data:" )
print(NN.guess(data_test))
print(NN.score(data_test,MNIST_test_labels))
"""