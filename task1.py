import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training

#input data
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
#output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])

#NeuralNetwork class
class NeuralNetwork:

    #intialize 
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    #sigmoid function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return self.sigmoid(x) * self.sigmoid(1 - x)
        return 1 / (1 + np.exp(-x))

    #feedforward
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    #backpropagation
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    #train net for 25,000 iterations
    def train(self, epochs=25000):
        for epoch in range(epochs):
            self.feed_forward()
            self.backpropagation()
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    #function to predict                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

#create neural network   
NN = NeuralNetwork(inputs, outputs)
#train neural network
NN.train()

#metrics of performance                                   
example_3 = np.array([[1, 0, 1]])

#print the metrics of performance                                
print(NN.predict(example_3), ' - Correct: ', example_3[0][0])

#plot graph of the error over train 
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()