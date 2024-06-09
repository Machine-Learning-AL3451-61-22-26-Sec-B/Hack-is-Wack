import streamlit as st
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o 

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def sigmoid_prime(self, s):
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoid_prime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

def scale_data(X, y):
    X_scaled = X / np.amax(X, axis=0)
    y_scaled = y / 100  # Scaling the output by dividing by 100
    return X_scaled, y_scaled

def main():
    st.title('Neural Network with Streamlit')

    # Original data
    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)     
    y = np.array(([92], [86], [89]), dtype=float)           

    # Scale the data
    X_scaled, y_scaled = scale_data(X, y)

    # Neural network initialization
    input_size = X_scaled.shape[1]
    output_size = y_scaled.shape[1]
    hidden_size = 3
    NN = NeuralNetwork(input_size, output_size, hidden_size)

    # Display original and scaled data
    st.subheader('Original Data:')
    st.write('Input:')
    st.write(X)
    st.write('Actual Output:')
    st.write(y)

    st.subheader('Scaled Data:')
    st.write('Scaled Input:')
    st.write(X_scaled)
    st.write('Scaled Output:')
    st.write(y_scaled)

    # Training the neural network
    NN.train(X_scaled, y_scaled)

    # Display predicted output and loss
    st.subheader('Prediction and Loss:')
    predicted_output = NN.forward(X_scaled)
    loss = np.mean(np.square(y_scaled - predicted_output))
    st.write('Predicted Output:')
    st.write(predicted_output)
    st.write('Loss:')
    st.write(loss)

    # Output the final weights
    st.subheader('Final Weights:')
    st.write('Weights from input to hidden layer:')
    st.write(NN.W1)
    st.write('Weights from hidden to output layer:')
    st.write(NN.W2)

if __name__ == "__main__":
    main()
