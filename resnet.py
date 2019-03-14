import numpy as np

class Resnet:

    # This is a simple resnet with a skip connection from the first layer to the output layer.

    def __init__(self, data, layers_count, node_count):
        '''
        data: numpy ndarray. Last column in array is targets
        weights: numpy ndarray. Contains the weights for each layer in each column.
        output: numpy ndarray. Contains the output of each layer in each column.
                Let first column in output be the input vector.
        error: numpy ndarray. Contains the error for each layer in each column.
        bias: numpy ndarray. Contains the bias for each layer in each column.
        '''
        self.data = data
        # TODO: Determine right size of weights vector
        self.weights = np.ones((layers_count, node_count)) # Initialize to one
        self.output = np.zeros((layers_count, node_count))  # Initialize to zero
        self.error = np.zeros((layers_count, node_count))   # Initialize to zero
        self.bias = np.ones((layers_count, node_count))     # Initialize to one
        self.layers_count = layers_count
        self.node_count = node_count

    def forward(self):
        '''
        output[layer] = activation(W[layer - 1][layer] * output[layer - 1] + bias[layer])
        '''

        # For each layer
        for layer in range(1, self.layers_count):
            # Check if layer is the skip layer
            if layer != self.skip_connection:
                net = self.weights[layer - 1]*self.output[layer - 1]# Compute the net of the layer
                self.output[layer] = self.activation() # Get the output of the next layer
            else:
                # Add the output of first layer
        pass

    def backprop(self):
        pass

    def activation(self, net):
        '''
        Computes a nonlinear activation function on the net to return the output
        :return:
        '''
        pass

    def train(self):
        '''
        Train the resnet
        :return:
        '''
        for input in self.data:
            self.forward(input)
            self.backprop()
