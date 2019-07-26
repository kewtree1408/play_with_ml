import numpy as np

from conv import Conv3x3
import mnist

# Download MNIST dataset
TRAIN_IMAGES = mnist.train_images()
TRAIN_LABELS = mnist.train_labels()


def test_mnist_shape():
    # The mnist package handles the MNIST dataset for us!
    # Learn more at https://github.com/datapythonista/mnist

    conv = Conv3x3(8)
    output = conv.convolve(TRAIN_IMAGES[0])
    expected_shape = (26, 26, 8)
    assert output.shape == expected_shape
