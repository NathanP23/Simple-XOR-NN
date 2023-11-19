"""
main.py sets up the input data, creates a neural network instance, and initiates the training process to teach the network to approximate the XOR operation.
Adjusting the input data or network architecture here can demonstrate how the neural network learns and adapts to different problems or configurations.
"""

from neural_network import NeuralNetwork
import numpy as np

if __name__ == "__main__":

    """
    Defines a NumPy array inputs containing input data. In this case, it represents the logical XOR operation with input pairs [0, 0], [0, 1], [1, 0], [1, 1].
    """
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    """
    Defines another NumPy array targets containing the corresponding target outputs for the XOR operation, which are [0], [1], [1], [0] respectively.
    """
    targets = np.array([[0], [1], [1], [0]])

    """
    Creates an instance of the NeuralNetwork class with specific sizes for the input, hidden, and output layers (input_size=2, hidden_size=4, output_size=1).
    This initializes a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron.
    """
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    """
    Invokes the train() method of the neural network instance nn. It trains the neural network using the provided inputs and targets data over 1000 epochs with a learning rate of 0.1. 
    This initiates the training process for the neural network, which includes forward and backward passes to update the weights and biases based on the provided data.
    """
    nn.train(inputs, targets, epochs=1000, learning_rate=0.1)