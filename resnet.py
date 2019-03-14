import numpy as np

class Resnet:
    def __init__(self, data, layers_count, node_count):
        '''
        data: list of tuples
        output: numpy array. Contains the output of each layer in each column.
        weights

        :return:
        '''
        self.data = data
        self.weights = np.zeros((layers_count, node_count))
        self.output = np.zeros((layers_count, node_count))
        self.error = np.zeros((layers_count, node_count))
        pass

    def forward(self):
        '''
        output: numpy array. Contains the output of each layer in each column.
        weights

        :return:
        '''
        pass

    def backprop(self):
        pass