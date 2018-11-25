import numpy as np
import csv

data =np.genfromtxt('cleveland_data.csv',delimiter=',')
labels = data[:,-1]

data= data[:,:-1]
#print(labels)
#print(data)

data = data/np.amax(data, axis=0) # sacale data form 0 to 1

class Neural_Network(object):
    def __init__(self, learning_rate, input_size, output_size):
        np.random.seed(5)
        self.learning_rate = learning_rate
        hidden_size = int(input_size / 2)
        self.weights1 = np.random.randn(input_size, hidden_size) # random wirghts for (input --> hidden)
        self.weights2 = np.random.randn(hidden_size, output_size)# random wirghts for (hidden --> output)

    def forward(self, data):
        #forward propagation through our network

        hidden_layer_sum = np.dot(data, self.weights1) # dot product of X (input) and set of weights1
        self.hidden_layer_result = self.activation(hidden_layer_sum) # activation function
        output_layer_sym = np.dot(self.hidden_layer_result, self.weights2) # dot product of hidden layer and  weights2
        output_layer_result = self.activation(output_layer_sym) # final activation function
        return output_layer_result


    def activation(self, s):
        return 1 / (1 + np.exp(-s))


    def activation_derivative(self, s):
        return s * (1 - s)


    def backward(self, data, label, prediction):
        output_delta = (label - prediction) * self.activation_derivative(prediction) # applying derivative of sigmoid to error
        hidden_delta = output_delta.dot(self.weights2.T) * self.activation_derivative(self.hidden_layer_result)# applying derivative of sigmoid to output_delta error
        self.weights1 += data.T.dot(hidden_delta) * self.learning_rate #adjusting first set (input --> hidden) weights
        self.weights2 += self.hidden_layer_result.T.dot(output_delta) * self.learning_rate #adjusting second set (hidden --> output) weights


    def train(self, data, label):
        o = self.forward(data)
        self.backward(data, label, o)
    def guess(self, data):
        return self.forward(data)
    def score(self, data, labels):
        score = 0
        for row in range(0, data.shape[0]):
            if np.argmax(self.guess(data[row])) == labels[row]:
                score +=1
        score = score/row
        return score

NN = Neural_Network(0.1,13,5)
LabelMatrix = np.empty(shape=[0,5])
for x in np.nditer(labels):
    if x == 0:
        LabelMatrix = np.append(LabelMatrix,[[1,0,0,0,0]],axis=0)
    elif x==1:
        LabelMatrix = np.append(LabelMatrix,[[0,1,0,0,0]],axis=0)
    elif x==2:
        LabelMatrix = np.append(LabelMatrix,[[0,0,1,0,0]],axis=0)
    elif x==3:
        LabelMatrix = np.append(LabelMatrix,[[0,0,0,1,0]],axis=0)
    elif x==4:
        LabelMatrix = np.append(LabelMatrix,[[0,0,0,0,1]],axis=0)
for i in range(1000):
    NN.train(data,LabelMatrix)

print(NN.score(data,labels))
