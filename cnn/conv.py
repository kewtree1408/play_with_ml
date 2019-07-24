import numpy as np


class Conv3x3:
    def __init__(self, filter_num):
        self.filter_num = filter_num

        self.filters = np.random.rand(filter_num, 3, 3) / 9

