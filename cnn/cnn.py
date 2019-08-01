import numpy as np

from conv_layer import ConvLayer3x3
from maxpool_layer import MaxPoolLayer2
from fc_layer import FullyConnectedLayer


class CNN:
    def __init__(self):
        self.epoches = 100
        self.softmax_loss = lambda x: -np.log(x)

    def feedforward(self, image):
        # Transform the image from [0, 255] to [-0.5, 0.5]
        image = (image / 255) - 0.5
        amount_of_filters = 8
        image_hight, image_width = image.shape
        amount_of_classes = 10

        self.conv_layer = ConvLayer3x3(amount_of_filters)
        # -2 because the conv layer will shift 2 elements
        conv_hight = image_hight-2
        conv_width = image_width-2
        self.maxpool_layer = MaxPoolLayer2()

        pool_hight = int(conv_hight/2)
        pool_width = int(conv_width/2)
        self.fc_layer = FullyConnectedLayer(pool_hight * pool_width * amount_of_filters, amount_of_classes)

        output = self.conv_layer.convolve(image)
        output = self.maxpool_layer.pool(output)
        predictions = self.fc_layer.feedforward(output)
        return predictions

    def train(self, train_data, train_labels):
        loss = 0
        accuracy = 0
        for idx, (tdata, tlabel) in enumerate(zip(train_data, train_labels)):
            predictions = self.feedforward(tdata)
            # Find the class (0-10) with the higest prebability
            pred_label = np.argmax(predictions)
            accuracy += int(pred_label == tlabel)
            loss += self.softmax_loss(predictions[tlabel])
            # TODO: backprop

        return accuracy, loss