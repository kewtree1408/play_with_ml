import numpy as np

from pooling import Pooling2


# Tests for this example:
# https://victorzhou.com/pool-ac441205fd06dc037b3db2dbf05660f7.gif

def test_divide_input_3x3():
    p = Pooling2()
    input_data = np.array([
        [0, 50, 0],
        [0, 80, 31],
        [33, 90, 0],
    ])
    expected_shape = (2, 2)
    expected_parts = [
        (np.array([
            [0, 50],
            [0, 80],
        ]), 0, 0),
        (np.array([
            [0, 29],
            [31, 2],
        ]), 0, 1),
        (np.array([
            [33, 90],
            [0, 9],
        ]), 1, 0),
        (np.array([
            [0, 75],
            [0, 95],
        ]), 1, 1),
    ]
    input_data_3d = input_data[:, :, np.newaxis]
    divided_parts = list(p.divide_input(input_data_3d))
    # import ipdb; ipdb.set_trace()
    assert len(divided_parts) == expected_shape
    for res, expected in zip(divided_parts, expected_parts):
        part_result, i_result, j_result = res
        part_expected, i_expected, j_expected = expected
        assert i_result == i_expected
        assert j_result == j_expected
        np.testing.assert_array_equal(part_result, part_expected)


def test_divide_input_4x4():
    p = Pooling2()
    input_data = np.array([
        [0, 50, 0, 29],
        [0, 80, 31, 2],
        [33, 90, 0, 75],
        [0, 9, 0, 95],
    ])
    input_data_3d = input_data[:, :, np.newaxis]
    expected_parts = [
        (np.array([
            [0, 50],
            [0, 80],
        ]), 0, 0),
        (np.array([
            [0, 29],
            [31, 2],
        ]), 0, 1),
        (np.array([
            [33, 90],
            [0, 9],
        ]), 1, 0),
        (np.array([
            [0, 75],
            [0, 95],
        ]), 1, 1),
    ]
    apply_3d = lambda m: m[:, :, np.newaxis]

    divided_parts = list(p.divide_input(input_data_3d))
    assert len(divided_parts) == 4
    for res, expected in zip(divided_parts, expected_parts):
        part_result, i_result, j_result = res
        part_expected, i_expected, j_expected = expected
        flat_res = part_result.flatten()
        flat_expected = part_expected.flatten()
        np.testing.assert_array_equal(flat_res, flat_expected)


def test_even():
    ...

def test_pool():
    ...

def test_zero_padding():
    ...