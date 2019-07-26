import numpy as np
from typing import Generator, Tuple


class Pooling2:
    # Pooling with size 2
    def __init__(self):
        self.size = 2
        self.pooling_method = np.max

    @staticmethod
    def even_size(input_filters):
        height, length, num_filters = input_filters.shape
        if height % 2 == 0 and length % 2 == 0:
            return True
        return False

    @staticmethod
    def zero_padding(input_filters):
        height, length, num_filters = input_filters.shape
        if height % 2 != 0:
            ...

    def divide_input(self, input_filters: np.ndarray) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """input_image is 2D"""
        even = self.even_size(input_filters)
        if not even:
            input_filters = self.zero_padding(input_filters)
        height, length, num_filters = input_filters.shape

        h = height // self.size
        l = length // self.size

        jump = self.size
        for i in range(h):
            for j in range(l):
                i_jump = i*jump
                j_jump = j*jump
                yield input_filters[
                    i_jump:i_jump+jump,
                    j_jump:j_jump+jump
                ], i, j

    def pool(self, input_filters):
        # input_filters
        height, length, num_filters = input_image.shape
        h = height // self.size
        l = length // self.size

        output = np.zeros((h, l, num_filters))
        for part, i, j in self.divide_input(input_filters):
            output[i, j] = np.max(part, axis=(0,1))
        return output
