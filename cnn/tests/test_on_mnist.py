import pytest

import mnist
import numpy as np

from cnn.cnn import CNN
from cnn.conv_layer import ConvLayer3x3
from cnn.fc_layer import FullyConnectedLayer
from cnn.maxpool_layer import MaxPoolLayer2


def test_mnist_shape():
    # The mnist package handles the MNIST dataset for us!
    # Learn more at https://github.com/datapythonista/mnist

    # Download MNIST dataset
    FIRST_N = 100
    train_images = mnist.train_images()[:FIRST_N]

    conv = ConvLayer3x3(8)
    output = conv.feedforward(train_images[0])
    assert output.shape == (26, 26, 8)

    p = MaxPoolLayer2()
    output = p.feedforward(output)
    assert output.shape == (13, 13, 8)

    amount_of_classes = 20
    fc = FullyConnectedLayer(np.prod(output.shape), amount_of_classes)
    output = fc.feedforward(output)
    assert output.shape == (amount_of_classes,)


@pytest.mark.slow
def test_mnist_train_one_epoch():
    FIRST_N = 1000
    train_images = mnist.train_images()[:FIRST_N]
    train_labels = mnist.train_labels()[:FIRST_N]

    np.random.seed(0)
    conv_net = CNN()
    accuracy, loss = conv_net._train_one_epoch(train_images, train_labels)
    assert accuracy == 83
    assert np.around(loss, decimals=2) == 69.8


@pytest.mark.slow
def test_mnist_train_and_test():
    # Download MNIST dataset
    FIRST_N = 1000
    train_images = mnist.train_images()[:FIRST_N]
    train_labels = mnist.train_labels()[:FIRST_N]
    test_images = mnist.test_images()[:FIRST_N]
    test_labels = mnist.test_labels()[:FIRST_N]

    np.random.seed(0)
    conv_net = CNN(epoch_amount=1)
    conv_net.train(train_images, train_labels)
    acc, loss = conv_net.test(test_images, test_labels)
    assert np.around(acc, decimals=2) == 0.83
    assert np.around(loss, decimals=2) == 0.71
