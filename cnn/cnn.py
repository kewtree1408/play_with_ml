import numpy as np

from .conv_layer import ConvLayer3x3
from .fc_layer import FullyConnectedLayer
from .maxpool_layer import MaxPoolLayer2


class CNN:
    def __init__(self):
        self.epoches = 100
        self.softmax_loss = lambda x: -np.log(x)
        self.amount_of_classes = 10

        amount_of_filters = 8
        self.expected_image_shape = (28, 28)
        image_hight, image_width = self.expected_image_shape

        self.conv_layer = ConvLayer3x3(amount_of_filters)
        # -2 because the conv layer will shift 2 elements
        conv_hight = image_hight - 2
        conv_width = image_width - 2
        self.maxpool_layer = MaxPoolLayer2()

        pool_hight = int(conv_hight / 2)
        pool_width = int(conv_width / 2)
        self.fc_layer = FullyConnectedLayer(
            pool_hight * pool_width * amount_of_filters, self.amount_of_classes
        )

    def feedforward(self, image):
        # Transform the image from [0, 255] to [-0.5, 0.5]
        image = (image / 255) - 0.5
        assert image.shape == self.expected_image_shape

        output = self.conv_layer.feedforward(image)
        output = self.maxpool_layer.feedforward(output)
        predictions = self.fc_layer.feedforward(output)
        return predictions

    def backprop(self, out, correct_label):
        gradient = np.zeros(self.amount_of_classes)
        # Loss = -ln(x)
        gradient[correct_label] = -1 / out[correct_label]
        gradient = self.fc_layer.backprop(gradient)
        # import ipdb; ipdb.set_trace()
        # gradient = self.maxpool_layer.backprop(gradient)
        # gradient = self.conv_layer.backprop(gradient)

        return gradient

    def train(self, train_data, train_labels):
        loss = 0
        accuracy = 0
        for idx, (tdata, tlabel) in enumerate(zip(train_data, train_labels)):
            if idx % 100 == 0:
                _print_loss_and_acc(idx, loss, accuracy)
                loss = 0
                accuracy = 0
            predictions = self.feedforward(tdata)
            # Find the class (0-10) with the higest prebability
            pred_label = np.argmax(predictions)
            accuracy += int(pred_label == tlabel)
            loss += self.softmax_loss(predictions[tlabel])

            self.backprop(predictions, tlabel)

        _print_loss_and_acc(idx, loss, accuracy)
        return accuracy, loss / 100


def _print_loss_and_acc(idx, loss, accuracy):
    print(
        f"[Step {idx+1}] Past 100 steps: Average Loss {loss/100} | Accuracy: {accuracy}"
    )
