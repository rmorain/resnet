# Main
from resnet import Resnet
data = [[0, 1, 1], [1, 0, 0], [1, 1, 1]] # Example test dataset
layers_count = 3    # The number of layers in the resnet
node_count = 3  # The number of nodes in each layer
resnet = Resnet(data, layers_count, node_count) # Initialize a resnet