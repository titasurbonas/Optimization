import numpy as np
import csv, Neural_Network
from sklearn.model_selection import train_test_split

data =np.genfromtxt('cleveland_data.csv',delimiter=',')
labels = data[:,-1]

data= data[:,:-1]
data = data/np.amax(data, axis=0)
#print(labels)
#print(data)
#labels = np.zeros((10, range(0, labels.shape[0])))[MNIST_test_labels] = 1
Data_Train , Data_Test , Labels_Train, Labels_Test  = train_test_split(data, labels, test_size=0.3, random_state=2)

label = np.zeros((
	,5))
for l in range(np.size(Labels_Train,0)):
    index = int(Labels_Train[l])
    label[l,index] =  1 




print(Data_Train.shape)
print(Data_Train[1])
print(label.shape)
print(label[1])
print(np.size(Data_Test,0))
for i in range(50000):
	NN = Neural_Network.Neural_Network(0.05,13,8,5)

NN.train(Data_Train,label)
 
print(NN.score(Data_Test,Labels_Test))
