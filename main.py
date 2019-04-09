# Main
from resnet import Resnet
import numpy as np

data = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]]) # Example test dataset
layers_count = 8    # The number of layers in the resnet
node_count = 3  # The number of nodes in each layer
skip_length = 3
skip_connection = [1, 4, 7]
resnet = Resnet(data, layers_count, node_count, skip_connection) # Initialize a resnet
resnet.train()