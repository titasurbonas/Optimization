#!/usr/bin/env python


import numpy as np
import NearestSubCentroidClassifier, Neural_Network , Perceptron
from sklearn.model_selection import train_test_split

####################################################################
# Read the data from files 
orl_data = np.genfromtxt('./ORL_txt/orl_data.txt',delimiter='	')	

orl_data = np.delete(orl_data, [400], axis=1).T


orl_data_labels = np.genfromtxt('./ORL_txt/orl_lbls.txt')
orl_data_labels = np.reshape(orl_data_labels, (-1, 1))

MNIST_train_data =np.genfromtxt('./MNIST/mnist_train.csv',delimiter=',')
MNIST_train_labels= MNIST_train_data[:,:1]
MNIST_train_data = np.delete(MNIST_train_data, [0], axis=1) 

MNIST_test_data = np.genfromtxt('./MNIST/mnist_test.csv',delimiter=',')
MNIST_test_labels = MNIST_test_data[:,:1]
MNIST_test_data = np.delete(MNIST_test_data, [0], axis=1) 

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
def displayData(data_2D, labels, test_data ,test_labels,algorithm):	
	from matplotlib import pyplot as plt
	from matplotlib.colors import ListedColormap
	h = .02 
	camp = ListedColormap(['#ee80ff', '#ffae80', '#73a4e6', '#bf6960', '#81468c', 
							'#5100ff', '#00ff2f', '#ff8000', '#d500ff', '#0063cc', 
						   	'#b20012', '#007fa6', '#00993b', '#8c2f00', '#008068',
						   	'#004b73', '#66002e', '#14004d', '#4c2b00', '#003321', 
						   	'#262200', '#0a001a', '#40ffa9', '#ff6940', '#3dddf2', 
						   	'#f2ce3d', '#e53967', '#95d936', '#bf3083', '#b27d2d', 
						   	'#432080', '#7b8020', '#4c1c13', '#0d2833', '#260a10', 
						   	'#336366', '#48592d', '#4d2642', '#ffbff4', '#ffc5bf'])
	x_min, x_max = data_2D[:, 0].min() - 1, data_2D[:, 0].max() + 1
	y_min, y_max = data_2D[:, 1].min() - 1, data_2D[:, 1].max() + 1

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	
	Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

	Z = Z.reshape(xx.shape)


	plt.figure(figsize=(10,10))
	plt.pcolormesh(xx, yy, Z,cmap=camp)
	#print(len(test_data))
	if len(test_data) > 400:
		test_data = test_data[::50]
		test_labels= test_labels[::50]

	plt.scatter(test_data[:,0], test_data[:,1],edgecolor='white', c=test_labels[:,0],cmap=camp)
	#plt.show()

####################################################################
# Nearest Centroid Classifier
def NearestCentroidClassifier(data, label):
	from sklearn.neighbors.nearest_centroid import NearestCentroid
	clf = NearestCentroid().fit(data, label[:,0]) 
	return clf

####################################################################
# Nearest Neighbors Classifier
def  NearestNeighborsClassifier(data, label):
	from sklearn.neighbors import KNeighborsClassifier
	nnc = KNeighborsClassifier(n_neighbors=5)
	nnc.fit(data,label[:,0])
	return nnc

####################################################################
# Data conversion to 2D
orl_data_2D = applyPCA(orl_data)

MNIST_2D = applyPCA(np.vstack((MNIST_train_data,MNIST_test_data)))

MNIST_train_2D = MNIST_2D[:60000]
MNIST_test_2D =  MNIST_2D[60000:]

#orlD_Train , orlD_Test , orlL_Train, orlL_Test  = train_test_split(orl_data_2D, orl_data_labels, test_size=0.3, random_state=2)

orlD_Train=	orl_data_2D[0:7]
orlD_Test= orl_data_2D[7:10]
orlL_Train =orl_data_labels[0:7]
orlL_Test =	orl_data_labels[7:10]

for i in range(2, 46):
	orlD_Train = np.vstack((orlD_Train,orl_data_2D[i*10-10:i*10-3]))
	orlD_Test= np.vstack((orlD_Test, orl_data_2D[i*10-3:i*10]))
	orlL_Train =np.vstack((orlL_Train, orl_data_labels[i*10-10:i*10-3]))
	orlL_Test =	np.vstack((orlL_Test,orl_data_labels[i*10-3:i*10]))


del orl_data_2D
del MNIST_2D

####################################################################
# orl data Nearest Centroid Classifier

classifierORL = NearestCentroidClassifier(orlD_Train,orlL_Train)

print("Nearest Centroid Classifier score for orl Data: "+ str(classifierORL.score(orlD_Test, orlL_Test)))
displayData(orlD_Train,orlL_Train,orlD_Test, orlL_Test, classifierORL)
del classifierORL

nscc = NearestSubCentroidClassifier.CentroidClassifier(orlD_Train,orlL_Train, False ,1	)
nscc.centroids(0)
print("Nearest Centroid Classifier mine score for orl Data:  " + str(nscc.score(orlD_Test, orlL_Test)))
del nscc

####################################################################
# orl data Nearest Centroid Classifier sub 2

nscc2 = NearestSubCentroidClassifier.CentroidClassifier(orlD_Train,orlL_Train, True ,1	)
nscc2.centroids(2)
print("Nearest Centroid Classifier sub2 mine score for orl Data:  " + str(nscc2.score(orlD_Test, orlL_Test)))
del nscc2

####################################################################
#  orl data Nearest Centroid Classifier sub 3
nscc3 = NearestSubCentroidClassifier.CentroidClassifier(orlD_Train,orlL_Train, True ,1	)
nscc3.centroids(3)
print("Nearest Centroid Classifier sub3 mine score for orl Data:  " + str(nscc3.score(orlD_Test, orlL_Test)))
del nscc3

####################################################################
# orl data Nearest Centroid Classifier sub 5
nscc5 = NearestSubCentroidClassifier.CentroidClassifier(orlD_Train,orlL_Train, True ,1	)
nscc5.centroids(5)
print("Nearest Centroid Classifier sub5 mine score for orl Data:  " + str(nscc5.score(orlD_Test, orlL_Test)))
del nscc5

####################################################################
#  orl data Nearest Neighbors Classifier
classifierORL = NearestNeighborsClassifier(orlD_Train,orlL_Train)

print("Nearest Neighbors Classifier score for orl Data:  "+ str(classifierORL.score(orlD_Test,orlL_Test)))
displayData(orlD_Train,orlL_Train,orlD_Test, orlL_Test, classifierORL)
del classifierORL

####################################################################
#orl Data Perceptron Backpropagation

perceptron = []
for i in range(int(np.amax(orlL_Train))):
	#print("Perceptron nr:" + str(i))
	ppn = Perceptron.Perceptron(learningRate=0.05, nb_epoch=16)
	labels = np.where(orlL_Train == i+1, 1, 0)
	ppn.train_weights(orlD_Train, labels) 
	perceptron.append(ppn)
#	print("Finished perceptron training nr."+ str(i))
	del ppn,labels
bias = np.ones(len(orlD_Test))
orlDpB_Test = np.c_[ bias,orlD_Test ]
score= 0
for d in range(orlDpB_Test.shape[0]):
	for p in range(len(perceptron)):	
		if perceptron[p].predict(orlDpB_Test[d]) == 1 and  p+1 == orlL_Test[d]:
			score += 1
print("Perceptron Backpropagation score for orl data: "+str(score/orlDpB_Test.shape[0]))
del  perceptron, bias, orlDpB_Test

####################################################################
# MNIST Nearest Centroid Classifier

classifierMNIST = NearestCentroidClassifier(MNIST_train_2D,MNIST_train_labels)

print("Nearest Centroid Classifier score for MNIST Data: " + str(classifierMNIST.score(MNIST_test_2D,MNIST_test_labels)))
displayData(MNIST_train_2D,MNIST_train_labels,MNIST_test_2D, MNIST_test_labels, classifierMNIST)
del classifierMNIST



nsccM = NearestSubCentroidClassifier.CentroidClassifier(MNIST_train_2D,MNIST_train_labels, False, 0)
nsccM.centroids(0)
print("Nearest Centroid Classifier mine score for MNIST Data:  " + str(nsccM.score(MNIST_test_2D, MNIST_test_labels)))
del nsccM

####################################################################
# MNIST data Nearest Centroid Classifier sub 2

nsccM2 = NearestSubCentroidClassifier.CentroidClassifier(MNIST_train_2D,MNIST_train_labels, True, 0)
nsccM2.centroids(2)
print("Nearest Centroid Classifier sub2 mine score for MNIST Data:  " + str(nsccM2.score(MNIST_test_2D, MNIST_test_labels)))
del nsccM2

####################################################################
#  MNIST data Nearest Centroid Classifier sub 3
nsccM3 = NearestSubCentroidClassifier.CentroidClassifier(MNIST_train_2D,MNIST_train_labels, True, 0)
nsccM3.centroids(3)
print("Nearest Centroid Classifier sub3 mine score for MNIST Data:  " + str(nsccM3.score(MNIST_test_2D, MNIST_test_labels)))
del nsccM3

####################################################################
#  MNIST data Nearest Centroid Classifier sub 5
nsccM5 = NearestSubCentroidClassifier.CentroidClassifier(MNIST_train_2D,MNIST_train_labels, True, 0)
nsccM5.centroids(5)
print("Nearest Centroid Classifier sub5 mine score for MNIST Data:  " + str(nsccM5.score(MNIST_test_2D, MNIST_test_labels)))
del nsccM5

####################################################################
#  MNIST data Nearest Neighbors Classifier

classifierMNIST = NearestNeighborsClassifier(MNIST_train_2D,MNIST_train_labels)

print ("Nearest Neighbors Classifier score for MNIST Data:  " + str(classifierMNIST.score(MNIST_test_2D,MNIST_test_labels) ))
displayData(orlD_Train,orlL_Train,orlD_Test, orlL_Test, classifierMNIST)
del classifierMNIST

####################################################################
#MNIST Data Perceptron Backpropagation

perceptron = []

for i in range(int(np.amax(MNIST_train_labels))+1):
	#print("Perceptron nr:" + str(i))
	ppn = Perceptron.Perceptron(learningRate=0.01, nb_epoch=10)
	labels = np.where(MNIST_train_labels == i, 1, 0)
	ppn.train_weights(MNIST_train_2D, labels) 
	perceptron.append(ppn)
#	print("Finished perceptron training nr."+ str(i))
	del ppn,labels
#print("Finish training")
bias = np.ones(len(MNIST_test_2D))
MNIST_test_bias_data =np.c_[ bias,MNIST_test_2D ]
score= 0
for d in range(MNIST_test_bias_data.shape[0]):
	for p in range(len(perceptron)):	
		if perceptron[p].predict(MNIST_test_bias_data[d]) == 1 and  p == MNIST_test_labels[d]:
			score += 1
print("Perceptron Backpropagation score for MNIST data: "+str(score/MNIST_test_bias_data.shape[0]))
# MNIST_test_2D  MNIST_train_2D  MNIST_train_labels  MNIST_test_labels

"""
ppn = Perceptron.Perceptron(learningRate=0.5, n_iter=20)
labels = np.where(orlL_Train == 1, 1, 0)
	#print(labels.flatten())

ppn.train_weights(orlD_Train, labels) 
"""
    #print(orlL_Train)
	#orlD_Train  orlD_Test orlL_Train  orlL_Test 
"""
####################################################################
#orl Data Nearest Neural_Network Backpropagation
NN = Neural_Network.Neural_Network(0.05,2,20,40)
data = orlD_Train/np.amax(orlD_Train, axis=0) # sacale data form -1 to 1
data_test = orlD_Test/np.amax(orlD_Test, axis=0)
label = np.zeros((np.size(orlL_Train,0),40))
for l in range(np.size(orlL_Train,0)):  
	index = int(orlL_Train[l])-1 # we dont have class 0
	label[l,index] = 1 
for i in range(1000):
	NN.train(data,label)

print ("Neural_Network Backpropagation for orl_data: "+ str(NN.score(data_test,orlL_Test)) )
del NN , data_test,label


####################################################################
#MNIST Data Nearest Neural_Network Backpropagation
NN = Neural_Network.Neural_Network(0.05,2,5,10)
data = MNIST_train_2D/np.amax(MNIST_train_2D, axis=0) # sacale data form -1 to 1
data_test = MNIST_test_2D/np.amax(MNIST_test_2D, axis=0)
label = np.zeros((np.size(MNIST_train_labels,0),10))
for l in range(np.size(MNIST_train_labels,0)):  
	index = int(MNIST_train_labels[l])
	label[l,index] = 1 
NN.train(data,label)

print (" Neural_Network Backpropagation for MNIST Data: " + str(NN.score(data_test,MNIST_test_labels)) )

del NN , data_test,label
"""
