import mnist
import numpy as np

from cnn.cnn import CNN
from cnn.conv_layer import ConvLayer3x3
from cnn.fc_layer import FullyConnectedLayer
from cnn.maxpool_layer import MaxPoolLayer2

# Download MNIST dataset
FIRST_N = 1000
TRAIN_IMAGES = mnist.train_images()[:FIRST_N]
TRAIN_LABELS = mnist.train_labels()[:FIRST_N]


def test_mnist_shape():
    # The mnist package handles the MNIST dataset for us!
    # Learn more at https://github.com/datapythonista/mnist

    conv = ConvLayer3x3(8)
    output = conv.feedforward(TRAIN_IMAGES[0])
    assert output.shape == (26, 26, 8)

    p = MaxPoolLayer2()
    output = p.feedforward(output)
    assert output.shape == (13, 13, 8)

    amount_of_classes = 20
    fc = FullyConnectedLayer(np.prod(output.shape), amount_of_classes)
    output = fc.feedforward(output)
    assert output.shape == (amount_of_classes,)


def test_mnist_train():
    np.random.seed(0)
    conv_net = CNN()
    acc, loss = conv_net.train(TRAIN_IMAGES, TRAIN_LABELS)
    assert acc == 79
    assert np.around(loss, decimals=2) == 1.15
