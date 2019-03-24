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
        self.skip_connection = False
        self.learning_rate = .0001

    def forward(self, input):
        '''
        output[layer] = activation(W[layer - 1][layer] * output[layer - 1] + bias[layer])
        '''

        # initialize input
        self.output[0] = input

        # For each layer
        for layer in range(1, self.layers_count):
            # Check if layer is the skip layer
            if layer != self.skip_connection:
                net = np.matmul(self.weights[layer - 1], self.output[layer - 1])# Compute the net of the layer
                self.output[layer] = self.activation(net) # Get the output of the next layer
            else:
                # Add the output of first layer and apply activation
                skip_output = self.activation(self.ouput[layer - 1] + self.data[0])
                # multiply output with weights (not sure if nessisary)
                net = np.matmul(self.weights[layer - 1], skip_output)
                # append to output
                self.output[layer] = self.activation(net)


    def backprop(self, target):

        for back_layer in reversed(range(0, self.layers_count)):
            fprime_net = self.output[back_layer] * (1 - self.output[back_layer])

            # output layer
            if back_layer == 0:
                # TODO: check the target is in correct np array format
                self.error[back_layer] = ((target - self.output[back_layer]) * fprime_net)

            # hidden layer
            else:
                self.error[back_layer] = (np.matmul(self.weights[back_layer].T, self.error[back_layer] * fprime_net))
            new_weights = self.learning_rate * (np.outer(self.error[back_layer], self.output[back_layer]))
            updated_weights = self.weights[back_layer] + new_weights
            self.weights[back_layer] = updated_weights

    def activation(self, net):
        '''
        Computes a nonlinear activation function on the net to return the output
        :return:
        '''
        return 1 / (1 + np.e ** -net)

    def train(self):
        '''
        Train the resnet
        :return:
        '''
        for input in self.data:
            self.forward(input)
            self.backprop([0, 1, 0])
