from typing import Generator, Tuple

import numpy as np


class MaxPoolLayer2:
    # Pooling with size 2
    def __init__(self):
        self.size = 2
        self.pooling_method = np.max

    @staticmethod
    def is_even(input_filters):
        height, width, num_filters = input_filters.shape
        if height % 2 == 0 and width % 2 == 0:
            return True
        return False

    @staticmethod
    def zero_padding(input_filters):
        height, width, num_filters = input_filters.shape
        zero_padded_filters = input_filters
        if height % 2 != 0:
            zero_padded_filters = np.insert(zero_padded_filters, height, 0, axis=0)
        if width % 2 != 0:
            zero_padded_filters = np.insert(zero_padded_filters, width, 0, axis=1)
        return zero_padded_filters

    def divide_input(
        self, input_filters: np.ndarray
    ) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """input_image is 2D"""

        even = self.is_even(input_filters)
        if not even:
            input_filters = self.zero_padding(input_filters)
        height, width, num_filters = input_filters.shape

        half_height = height // self.size
        half_length = width // self.size

        jump = self.size
        for i in range(half_height):
            for j in range(half_length):
                i_jump = i * jump
                j_jump = j * jump
                yield input_filters[
                    i_jump : i_jump + jump, j_jump : j_jump + jump
                ], i, j

    def backprop(self, d_L_d_out):
        """
        :param d_l_d_out: the loss gradient from the previous step
        """
        d_L_d_input = np.zeros(self.last_input.shape)

        for img_part, idx, jdx in self.divide_input(self.last_input):
            height, weight, depth = img_part.shape
            amax = np.amax(img_part, axis=(0, 1))

            for i in range(height):
                for j in range(weight):
                    for f in range(depth):
                        # If this pixel was the max value, copy the gradient to it.
                        if img_part[i, j, f] == amax[f]:
                            d_L_d_input[idx * 2 + i, jdx * 2 + j, f] = d_L_d_out[
                                idx, jdx, f
                            ]

        return d_L_d_input

    def feedforward(self, input_filters):
        # input_filters is a 3D array from Conv layer
        self.last_input = input_filters

        height, width, num_filters = input_filters.shape
        half_height = height // self.size
        half_width = width // self.size

        output = np.zeros((half_height, half_width, num_filters))
        for part, i, j in self.divide_input(input_filters):
            # import ipdb; ipdb.set_trace()
            output[i, j] = np.max(part, axis=(0, 1))
        return output
