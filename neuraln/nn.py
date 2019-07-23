from typing import Callable

import numpy as np
from activations import sigmoid
from derivatives import sigmoid_deriv
from losses import mse


class Neuron:
    def __init__(
        self, weights: np.ndarray, bias: float, activation_func: Callable = sigmoid
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func

    def act_sum(self, inputs):
        # sum for activation function or for backpropagation derivative
        return np.dot(inputs, self.weights) + self.bias

    def feedforward(self, inputs: np.ndarray):
        assert inputs.shape == self.weights.shape
        return self.activation_func(self.act_sum(inputs))


class NeuralNetwork1HiddenLayer:
    def __init__(self):
        """
        A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
        """
        weights_size = 6
        bias_size = 3

        # all zeroes for simplier testing
        self.weights = np.array([0.0 for _ in range(weights_size)])
        self.biases = np.array([0.0 for _ in range(bias_size)])

        self.learning_rate = 0.1
        self.epochs = 1000

        self.setup_neurons()

    def setup_neurons(self):
        # Reassign values, because biases and weights might be changed
        hiddenn_weights_1 = self.weights[:2]
        hiddenn_weights_2 = self.weights[2:4]
        outputn_weights = self.weights[4:]

        self.hidden_neuron_1 = Neuron(hiddenn_weights_1, self.biases[0])
        self.hidden_neuron_2 = Neuron(hiddenn_weights_2, self.biases[1])
        self.output_neuron = Neuron(outputn_weights, self.biases[2])

    def feedforward(self, inputs):
        self.setup_neurons()
        ffd_hidden_1 = self.hidden_neuron_1.feedforward(inputs)
        ffd_hidden_2 = self.hidden_neuron_2.feedforward(inputs)
        hidden_inputs = np.array([ffd_hidden_1, ffd_hidden_2])
        return self.output_neuron.feedforward(hidden_inputs)

    def get_all_partial_derivatives(self, train_data, y_true):
        y_pred = self.feedforward(train_data)
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Update weights for Neuron
        self.setup_neurons()
        sum_h1 = self.hidden_neuron_1.act_sum(train_data)
        sum_h2 = self.hidden_neuron_2.act_sum(train_data)
        sum_o1 = self.output_neuron.act_sum(train_data)

        # Output neuron
        d_ypred_d_w5 = self.hidden_neuron_1.feedforward(train_data) * sigmoid_deriv(sum_o1)
        d_ypred_d_w6 = self.hidden_neuron_2.feedforward(train_data) * sigmoid_deriv(sum_o1)
        d_ypred_d_b3 = sigmoid_deriv(sum_o1)

        d_ypred_d_h1 = self.weights[4] * sigmoid_deriv(sum_o1)
        d_ypred_d_h2 = self.weights[5] * sigmoid_deriv(sum_o1)

        # Neuron h1
        d_h1_d_w1 = train_data[0] * sigmoid_deriv(sum_h1)
        d_h1_d_w2 = train_data[1] * sigmoid_deriv(sum_h1)
        d_h1_d_b1 = sigmoid_deriv(sum_h1)

        # Neuron h2
        d_h2_d_w3 = train_data[0] * sigmoid_deriv(sum_h2)
        d_h2_d_w4 = train_data[1] * sigmoid_deriv(sum_h2)
        d_h2_d_b2 = sigmoid_deriv(sum_h2)

        return (
            d_L_d_ypred,
            d_h1_d_w1,
            d_h1_d_w2,
            d_h1_d_b1,
            d_h2_d_w3,
            d_h2_d_w4,
            d_h2_d_b2,
            d_ypred_d_h1,
            d_ypred_d_h2,
            d_ypred_d_w5,
            d_ypred_d_w6,
            d_ypred_d_b3
        )

    def backpropagation(self, train_data, y_true):
        assert train_data.shape[0] == 2
        (
            d_L_d_ypred,
            d_h1_d_w1,
            d_h1_d_w2,
            d_h1_d_b1,
            d_h2_d_w3,
            d_h2_d_w4,
            d_h2_d_b2,
            d_ypred_d_h1,
            d_ypred_d_h2,
            d_ypred_d_w5,
            d_ypred_d_w6,
            d_ypred_d_b3
        ) = self.get_all_partial_derivatives(train_data, y_true)

        backprop_weights = np.array([
            self.learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1,
            self.learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2,
            self.learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3,
            self.learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4,
            self.learning_rate * d_L_d_ypred * d_ypred_d_w5,
            self.learning_rate * d_L_d_ypred * d_ypred_d_w6,
        ])
        backprop_biases = np.array([
            self.learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1,
            self.learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2,
            self.learning_rate * d_L_d_ypred * d_ypred_d_b3,
        ])
        # import ipdb; ipdb.set_trace()
        return backprop_weights, backprop_biases

    def train(self, train_data, y_trues):
        for epoch in range(self.epochs):
            for input_data, y_true in zip(train_data, y_trues):
                y_pred = self.feedforward(input_data)
                backprop_weights, backprop_biases = self.backpropagation(input_data, y_true)
                # import ipdb; ipdb.set_trace()
                self.weights -= backprop_weights
                self.biases -= backprop_biases
                # import ipdb; ipdb.set_trace()
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, train_data)
                loss = mse(y_trues, y_preds)
                print(f"Epoch {epoch} loss: {loss}")

    def prediction(self, inputs):
        # Return value which is predicted by network
        return self.feedforward(inputs)


def main():
    # Define dataset
    input_data = np.array([
        [-2, -1],  # Alice
        [25, 6],   # Bob
        [17, 4],   # Charlie
        [-15, -6], # Diana
    ])
    all_y_trues = np.array([
        1, # Alice
        0, # Bob
        0, # Charlie
        1, # Diana
    ])

    # Train our neural network!
    network = NeuralNetwork1HiddenLayer()
    network.train(input_data, all_y_trues)

    # Make some predictions
    emily = np.array([-7, -3]) # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches

    # TODO: add this check into test
    print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
    print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M

if __name__ == "__main__":
    main()
