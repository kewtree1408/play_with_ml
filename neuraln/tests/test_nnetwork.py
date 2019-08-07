from unittest.mock import MagicMock

import numpy as np

from neuraln.nn import NeuralNetwork1HiddenLayer


def test_nn_feedforward():
    # suppose, weight and bias are the same for every neuron
    inputs = np.array([2, 3])
    nn = NeuralNetwork1HiddenLayer()
    ffd_result = nn.feedforward(inputs)
    assert np.around(ffd_result, decimals=4) == 0.5


def test_nn_setup_neurons():
    # try set up neurons for a few times
    nn = NeuralNetwork1HiddenLayer()
    deafult_weights = np.array([0.0 for _ in range(6)])
    expected_weights = np.array([0.0, 0.0])
    np.testing.assert_array_equal(nn.hidden_neuron_1.weights, expected_weights)
    np.testing.assert_array_equal(nn.hidden_neuron_2.weights, expected_weights)
    np.testing.assert_array_equal(nn.output_neuron.weights, expected_weights)

    second_inputs = np.array([10.0 for _ in range(6)])
    nn.weights = second_inputs
    expected_weights = np.array([10.0, 10.0])
    nn.setup_neurons()
    np.testing.assert_array_equal(nn.hidden_neuron_1.weights, expected_weights)
    np.testing.assert_array_equal(nn.hidden_neuron_2.weights, expected_weights)
    np.testing.assert_array_equal(nn.output_neuron.weights, expected_weights)


def test_nn_setup_backpropagartion():
    nn_mock = NeuralNetwork1HiddenLayer()
    nn_mock.hidden_inputs = [0, 0]
    nn_mock.setup_neurons = MagicMock(return_value=None)
    nn_mock.backpropagation(train_data=np.array([1, 1]), y_true=1, y_pred=1)
    nn_mock.backpropagation(train_data=np.array([2, 2]), y_true=2, y_pred=2)
    assert nn_mock.setup_neurons.call_count == 0


def test_train():
    input_data = np.array(
        [[-2, -1], [25, 6], [17, 4], [-15, -6]]  # Alice  # Bob  # Charlie  # Diana
    )
    all_y_trues = np.array([1, 0, 0, 1])  # Alice  # Bob  # Charlie  # Diana

    # Train our neural network!
    network = NeuralNetwork1HiddenLayer()
    loss_value = network.train(input_data, all_y_trues)

    expected_loss = 0.0024
    assert np.around(loss_value, decimals=4) == expected_loss

    # Make some predictions
    emily = np.array([-7, -3])  # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches

    emily_prob = 0.967
    frank_prob = 0.056
    assert np.around(network.feedforward(emily), decimals=3) == emily_prob
    assert np.around(network.feedforward(frank), decimals=3) == frank_prob
