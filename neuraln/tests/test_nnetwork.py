import numpy as np

from nn import NeuralNetwork1HiddenLayer


def test_nn_feedforward():
    # suppose, weight and bias are the same for every neuron
    inputs = np.array([2, 3])
    nn = NeuralNetwork1HiddenLayer()
    ffd_result = nn.feedforward(inputs)
    assert np.around(ffd_result, decimals=4) == 0.5
