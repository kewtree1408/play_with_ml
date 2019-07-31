import numpy as np

from conv_layer import ConvLayer3x3
from maxpool_layer import MaxPoolLayer2
from fc_layer import FullyConnectedLayer


class CNN:
    def feedforward(self, image):
        amount_of_filters = 8
        self.conv_layer = ConvLayer3x3(amount_of_filters)
        self.maxpool_layer = MaxPoolLayer2()
        self.fc_layer = FullyConnectedLayer(13 * 13 * 8, 10)
