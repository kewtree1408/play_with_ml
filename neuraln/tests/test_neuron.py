import numpy as np

from nn import Neuron


def test_feedforward_without_hidden_layers():
    # output layer: x1, x2, w1, w2, b
    # f - sigmoid by default

    # suppose, weight and bias are the same for every neuron
    inputs = np.array([2, 3])
    weights = np.array([0, 1])
    bias = 0

    n = Neuron(weights, bias)
    ffd_result = n.feedforward(inputs)
    assert np.around(ffd_result, decimals=4) == 0.9526


def test_feedforward_with_hidden_layers():
    # hidden layer and output layer
    # f - sigmoid by default

    # suppose, weight and bias are the same for every neuron
    inputs = np.array([2, 3])
    weights = np.array([0, 1])
    bias = 0

    n = Neuron(weights, bias)
    hidden_res = n.feedforward(inputs)
    hidden_inputs = np.full(2, hidden_res)
    ffd_result = n.feedforward(hidden_inputs)

    assert np.around(ffd_result, decimals=4) == 0.7216


def test_feedforward_with_different_weights():
    # hidden layer and output layer
    # f - sigmoid by default

    # suppose, weight and bias are not the same
    inputs = np.array([1, 2, 3])
    weights = np.array([4, 5, 6])
    bias = 7
    sh = inputs.shape

    n_hidden = Neuron(weights, bias)
    hidden_res = n_hidden.feedforward(inputs)
    hidden_inputs = np.full(sh, hidden_res)
    hidden_weights = np.array([8, 9, 10])
    hidden_bias = 11

    n_output = Neuron(hidden_weights, hidden_bias)
    output_result = n_output.feedforward(hidden_inputs)

    print(output_result)
    assert np.around(output_result, decimals=4) == 1.0
