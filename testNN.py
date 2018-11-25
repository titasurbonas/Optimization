import numpy as np
import csv, Neural_Network

data =np.genfromtxt('cleveland_data.csv',delimiter=',')
labels = data[:,-1]

data= data[:,:-1]
#print(labels)
#print(data)
#labels = np.zeros((10, range(0, labels.shape[0])))[MNIST_test_labels] = 1
label = np.zeros((np.size(labels,0),5))
print(label)
for l in range(np.size(labels,0)):
    index = int(labels[l])
    label[l,index] =  1 


data = data/np.amax(data, axis=0) # sacale data form 0 to 1
print(data.shape)
print(label.shape)

NN = Neural_Network.Neural_Network(0.1,13,8,5)
for i in range(1000):
    NN.train(data,label)

print(NN.score(data,labels))
