from typing import Callable

import numpy as np
from activations import sigmoid


class Neuron:
    def __init__(
        self, weights: np.ndarray, bias: float, activation_func: Callable = sigmoid
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func

    def feedforward(self, inputs: np.ndarray):
        assert inputs.shape == self.weights.shape
        return self.activation_func(np.dot(inputs, self.weights) + self.bias)


class NeuralNetwork1HiddenLayer:
    def __init__(self):
        self.weights = np.array([0, 1])
        self.bias = 0

        self.hidden_neuron_1 = Neuron(self.weights, self.bias)
        self.hidden_neuron_2 = Neuron(self.weights, self.bias)
        self.output_neuron = Neuron(self.weights, self.bias)

    def feedforward(self, inputs):
        assert inputs.shape == self.weights.shape

        ffd_hidden_1 = self.hidden_neuron_1.feedforward(inputs)
        ffd_hidden_2 = self.hidden_neuron_2.feedforward(inputs)
        hidden_inputs = np.array([ffd_hidden_1, ffd_hidden_2])
        return self.output_neuron.feedforward(hidden_inputs)


def main():
    pass


if __name__ == "__main__":
    main()
