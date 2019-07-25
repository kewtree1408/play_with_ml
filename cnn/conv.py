import numpy as np
from typing import Generator, Tuple


class Conv3x3:
    def __init__(self, filters_amount: int):
        self.filters_amount = filters_amount
        self.filter_size = 3 # because Conv3x3, but can be any number
        # 9 - magic number for Xavier - ? TODO!
        self.filters = np.random.rand(
            filters_amount, self.filter_size, self.filter_size
        ) / 9

    @staticmethod
    def divide_input(input_image: np.ndarray) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """input_image is 2D"""
        height, length = input_image.shape
        for i in range(height-2):
            for j in range(length-2):
                yield input_image[i:i+3, j:j+3], i, j

    @staticmethod
    def dot_sum(image_part, conv_filter):
        assert image_part.shape == conv_filter.shape
        return np.sum(np.multiply(image_part, conv_filter))

    def forward(self, input_image: np.ndarray) -> np.ndarray:
        height, length = input_image.shape
        output_data = np.zeros((height-2, length-2, self.filters_amount))
        for img_part, i, j in self.divide_input(input_image):
            output_data[i, j] = np.sum(img_part * self.filters, axis=(1, 2))
            # for conv_filter in self.filters:
            #     output_data[i, j] = self.dot_sum(img_part, conv_filter)
        return output_data

