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

        h = height // self.size
        l = width // self.size

        jump = self.size
        for i in range(h):
            for j in range(l):
                i_jump = i * jump
                j_jump = j * jump
                yield input_filters[
                    i_jump : i_jump + jump, j_jump : j_jump + jump
                ], i, j

    def backprop(self, d_L_d_out):
        """
        :param d_l_d_out: gradient from the previous layer
        """
        d_L_d_input = np.zeros(self.last_input.shape)
        height, width, num_filters = d_L_d_input.shape
        for i in range(height):
            for j in range(width):
                idx = int(i / self.size)
                jdx = int(j / self.size)
                if i % self.size == 0 and j % self.size == 0:
                    d_L_d_input[i, j] = d_L_d_out[idx, jdx]
        return d_L_d_input

    # def backprop2(self, d_L_d_out):
    #     '''
    #     Performs a backward pass of the maxpool layer.
    #     Returns the loss gradient for this layer's inputs.
    #     - d_L_d_out is the loss gradient for this layer's outputs.
    #     '''
    #     d_L_d_input = np.zeros(self.last_input.shape)

    #     for im_region, i, j in self.divide_input(self.last_input):
    #         h, w, f = im_region.shape
    #         amax = np.amax(im_region, axis=(0, 1))

    #         for i2 in range(h):
    #             for j2 in range(w):
    #                 for f2 in range(f):
    #                     # If this pixel was the max value, copy the gradient to it.
    #                     if im_region[i2, j2, f2] == amax[f2]:
    #                         d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    #     return d_L_d_input

    def feedforward(self, input_filters):
        # input_filters is a 3D array from Conv layer
        self.last_input = input_filters

        height, width, num_filters = input_filters.shape
        h = height // self.size
        l = width // self.size

        output = np.zeros((h, l, num_filters))
        for part, i, j in self.divide_input(input_filters):
            # import ipdb; ipdb.set_trace()
            output[i, j] = np.max(part, axis=(0, 1))
        return output
