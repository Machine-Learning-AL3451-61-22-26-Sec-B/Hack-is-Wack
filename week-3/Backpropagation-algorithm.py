import numpy as np
import streamlit as st

class NeuralNetwork:
    def __init__(self):
        # Parameters
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3
        
        # Weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        # Forward propagation through the network
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o 

    def backward(self, X, y, o):
        # Backward propagate through the network
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoid_prime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        # Training function
        o = self.forward(X)
        self.backward(X, y, o)

    def sigmoid(self, s):
        # Activation function
        return 1 / (1 + np.exp(-s))

    def sigmoid_prime(self, s):
        # Derivative of sigmoid
        return s * (1 - s)

    def evaluate(self, X):
        # Evaluation function
        return self.forward(X)

if __name__ == "__main__":
    # Sample data
    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
    y = np.array(([92], [86], [89]), dtype=float)

    # Scale units
    X = X / np.amax(X, axis=0)
    y = y / 100  # Max test score is 100

    # Create and train neural network
    NN = NeuralNetwork()
    print("\nInput:\n", X)
    print("\nActual Output:\n", y)
    print("\nPredicted Output:\n", NN.evaluate(X))
    print("\nLoss:\n", np.mean(np.square(y - NN.evaluate(X))))  # Mean squared loss
    NN.train(X, y)
