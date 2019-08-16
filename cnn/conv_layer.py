from typing import Generator, Tuple

import numpy as np


class ConvLayer3x3:
    def __init__(self, filters_amount: int):
        self.filters_amount = filters_amount
        self.filter_size = 3  # because ConvLayer3x3, but can be any number
        # 9 - magic number for Xavier - ? TODO!
        self.filters = (
            np.random.rand(filters_amount, self.filter_size, self.filter_size) / 9
        )
        self.learning_rate = 0.005

    def divide_input(
        self, input_image: np.ndarray
    ) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """input_image is 2D"""
        height, weight = input_image.shape
        boundary = self.filter_size - 1
        for i in range(height - boundary):
            for j in range(weight - boundary):
                yield input_image[
                    i : i + self.filter_size, j : j + self.filter_size
                ], i, j

    @staticmethod
    def dot_sum(image_part, conv_filter):
        assert image_part.shape == conv_filter.shape
        return np.sum(np.multiply(image_part, conv_filter))

    def feedforward(self, input_image: np.ndarray) -> np.ndarray:
        self.last_image = input_image
        height, weight = input_image.shape
        boundary = self.filter_size - 1
        output_data = np.zeros(
            (height - boundary, weight - boundary, self.filters_amount)
        )
        for img_part, i, j in self.divide_input(input_image):
            # Next line is the same as, but faster:
            # for conv_filter in self.filters:
            #     output_data[i, j] = self.dot_sum(img_part, conv_filter)
            output_data[i, j] = np.sum(img_part * self.filters, axis=(1, 2))
        return output_data

    def backprop(self, d_L_d_out):
        # The gradient: d_L_d_filters = d_L_d_out * d_out_d_filter
        # d_out_d_filter = part_of_image from self.divide_input()
        d_L_d_filters = np.zeros(self.filters.shape)
        for img_part, i, j in self.divide_input(self.last_image):
            for f in range(self.filters_amount):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * img_part

        self.filters -= self.learning_rate * d_L_d_filters
        return d_L_d_filters
