import numpy as np

from conv import Conv3x3
from pooling import Pooling2
from fc_layer import FullyConnectedLayer

import mnist

# Download MNIST dataset
TRAIN_IMAGES = mnist.train_images()
TRAIN_LABELS = mnist.train_labels()


def test_mnist_shape():
    # The mnist package handles the MNIST dataset for us!
    # Learn more at https://github.com/datapythonista/mnist

    conv = Conv3x3(8)
    output = conv.convolve(TRAIN_IMAGES[0])
    assert output.shape == (26, 26, 8)

    p = Pooling2()
    output = p.pool(output)
    assert output.shape == (13, 13, 8)

    amount_of_classes = 20
    fc = FullyConnectedLayer(np.prod(output.shape), amount_of_classes)
    output = fc.feedforward(output)
    assert output.shape == (amount_of_classes,)
