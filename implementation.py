import numpy as np


class Node:

    def __init__(self, weights, layer_number, type):
        self.weights = weights
        self.layer_number = layer_number
        self.type = type

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_pass(self, f):
        return node.sigmoid(np.dot(self.weights * f))

    def backward_pass(self):
        # TODO: adjust and apply gradients!
        pass

if __name__ == "__main__":
    n = Node()