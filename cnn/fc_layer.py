# Fully connected layer

import numpy as np


class FullyConnectedLayer:
    """
    Fully Connected Layer with Softmax activation
    """
    def __init__(self, flatten_filters_amount, class_amount):
        self.weights = np.random.rand(flatten_filters_amount, class_amount)
        self.biases = np.random.rand(class_amount)

    def feedforward(self, filters):
        flat_filters = filters.flatten()
        import ipdb; ipdb.set_trace()
        totals = np.dot(flat_filters, self.weights) + self.biases
        e_xs = np.exp(totals)
        return e_xs / np.sum(e_xs)
