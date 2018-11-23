#!/usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split

###########################################################
# Read the data fromm files 

orl_data = pd.read_csv('./ORL_txt/orl_data.txt' ,sep="	", header=None).transpose()
orl_data.drop(orl_data.index[400], axis=0, inplace=True)
orl_data_labels = pd.read_csv('./ORL_txt/orl_lbls.txt ', header=None)
orl_data_labels = orl_data_labels.iloc[:,0]
#print (orl_data_labels)
#print (orl_data)
"""
MNIST_train_data = pd.read_csv('./MNIST/mnist_train.csv', header=None)
MNIST_train_labels = MNIST_train_data.iloc[:,0]
MNIST_train_data.drop(MNIST_train_data.index[0], axis=1, inplace=True)


MNIST_test_data = pd.read_csv('./MNIST/mnist_test.csv', header=None)
MNIST_test_labels = MNIST_test_data.iloc[:,0]
MNIST_test_data.drop(MNIST_test_data.index[0], axis=1, inplace=True)
"""
	

#print(MNIST_train_data)

############################################################
# Fit a PCA
def applyPCA(data):
	from sklearn.decomposition import PCA
	from sklearn.preprocessing import StandardScaler
	f_transform = StandardScaler().fit_transform(data)
	pca = PCA(n_components=2, whiten=True).fit_transform(data)
	#pca.fit(data)
	# Project the data in 2D
	#pca.transform(data)
	#print(pca)
	return pca
def displayData(data_2D, labels, test_data ,predicted,title):	
	from matplotlib import pyplot as plt
	labels = labels.values
	#predicted = predicted.values
	plt.figure(figsize=(20, 20)) 
	plt.scatter(data_2D[:,0], data_2D[:,1],marker="v",  c=labels[:],)
	plt.scatter(test_data[:,0], test_data[:,1],marker="x",  c=predicted[:])
	plt.title(title)
	plt.show()
def NearestCentroidClassifier(data, label):
	from sklearn.neighbors.nearest_centroid import NearestCentroid
	clf = NearestCentroid().fit(data, label) # label expected array gets 1D matrix future update
	return clf
def  NearestNeighborsClassifier(data, label):
	from sklearn.neighbors import KNeighborsClassifier
	nnc = KNeighborsClassifier(n_neighbors=5)
	nnc.fit(data,label)# label expected array gets 1D matrix future update
	return nnc

"""
MNIST_2D_train = applyPCA(MNIST_train_data)
MNIST_2D_test = applyPCA(MNIST_test_data)
"""

#orl data
orl_data_2D = applyPCA(orl_data)
orlD_Train , orlD_Test , orlL_Train, orlL_Test  = train_test_split(orl_data_2D, orl_data_labels, test_size=0.3, random_state=0)

classifierORL = NearestNeighborsClassifier(orlD_Train,orlL_Train)

print (classifierORL.score(orlD_Test,orlL_Test))

displayData(orlD_Train,orlL_Train,orlD_Test, classifierORL.predict(orlD_Test), NearestNeighborsClassifier)
"""
#MNIST
MNIST_train_2D = applyPCA(MNIST_train_data)
MNIST_test_2D = applyPCA(MNIST_test_data)

classifierMNIST = NearestNeighborsClassifier(MNIST_train_2D,MNIST_train_labels)

print (classifierMNIST.score(MNIST_test_2D,MNIST_test_labels))

displayData(MNIST_train_2D,MNIST_train_labels,MNIST_test_2D, classifierMNIST.predict(MNIST_test_2D), NearestNeighborsClassifier)
"""

"""
print (classifierORL.score(orlD_Test,orlL_Test))
displayData(orl_data_2D,orl_data_labels )
"""
"""
print("hear")
classifierMNIST = NearestNeighborsClassifier(MNIST_train_data,MNIST_train_labels)
print (classifierMNIST.score(MNIST_test_data,MNIST_test_labels))
"""
#nccc_orl = NearestCentroidclassifier(orlD_Train,orlL_Train)
#print (nccc_orl.score(orlD_Test,orlL_Test))

#displayData(orl_data_2D)
#
#MNIST_train_data_2D = applyPCA(MNIST_train_data)
#displayData(MNIST_train_data_2D)