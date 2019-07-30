import numpy as np

from pooling import Pooling2


# Tests for this example:
# https://victorzhou.com/pool-ac441205fd06dc037b3db2dbf05660f7.gif
# https://victorzhou.com/media/cnn-post/maxpool-forward.svg


def _assert_divided_parts(res, expected):
    part_result, i_result, j_result = res
    part_expected, i_expected, j_expected = expected
    flat_res = part_result.flatten()
    flat_expected = part_expected.flatten()
    np.testing.assert_array_equal(flat_res, flat_expected)


def test_is_even():
    input_data = np.array([
        [0, 50, 0, 29],
        [0, 80, 31, 2],
        [33, 90, 0, 75],
        [0, 9, 0, 95],
    ])
    input_data_3d = input_data[:, :, np.newaxis]
    assert Pooling2.is_even(input_data_3d) is True

    input_data = np.array([
        [0, 50, 0],
        [0, 80, 31],
        [33, 90, 0],
    ])
    input_data_3d = input_data[:, :, np.newaxis]
    assert Pooling2.is_even(input_data_3d) is False

    input_data = np.array([
        [0, 50, 0],
        [0, 80, 31],
    ])
    input_data_3d = input_data[:, :, np.newaxis]
    assert Pooling2.is_even(input_data_3d) is False


def test_zero_padding_3x3():
    input_data = np.array([
        [0, 50, 0],
        [0, 80, 31],
        [33, 90, 0],
    ])
    input_data_3d = input_data[:, :, np.newaxis]

    res = Pooling2.zero_padding(input_data_3d)
    expected = np.array([
        [[0], [50], [0], [0]],
        [[0], [80], [31], [0]],
        [[33], [90], [0], [0]],
        [[0], [0], [0], [0]]
    ])
    np.testing.assert_array_equal(res, expected)


def test_divide_input_3x3():
    input_data = np.array([
        [0, 50, 0],
        [0, 80, 31],
        [33, 90, 0],
    ])
    expected_shape = 4
    expected_parts = [
        (np.array([
            [[0], [50]],
            [[0], [80]]
        ]), 0, 0),
        (np.array([
            [[0], [0]],
            [[31], [0]],
        ]), 0, 1),
        (np.array([
            [[33], [90]],
            [[0], [0]],
        ]), 1, 0),
        (np.array([
            [[0], [0]],
            [[0], [0]],
        ]), 1, 1),
    ]
    input_data_3d = input_data[:, :, np.newaxis]

    p = Pooling2()
    divided_parts = list(p.divide_input(input_data_3d))

    assert len(divided_parts) == expected_shape
    for res, expected in zip(divided_parts, expected_parts):
        _assert_divided_parts(res, expected)


def test_divide_input_4x4():
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

    p = Pooling2()
    divided_parts = list(p.divide_input(input_data_3d))

    assert len(divided_parts) == 4
    for res, expected in zip(divided_parts, expected_parts):
        _assert_divided_parts(res, expected)


def test_divide_2_filters():
    filter1 = np.array([
        [0, 50, 0, 29],
        [0, 80, 31, 2],
        [33, 90, 0, 75],
        [0, 9, 0, 95],
    ])
    filter1 = filter1[:, :, np.newaxis]
    filter2 = np.array([
        [0, 55, 0, 29],
        [20, 0, 41, 33],
        [0, 90, 0, 0],
        [0, 57, 0, 95],
    ])
    filter2 = filter2[:, :, np.newaxis]
    filters = np.concatenate((filter1, filter2), axis=2)
    expected_parts = [
        (np.array([
            [[ 0,  0],
            [50, 55]],
            [[ 0, 20],
            [80,  0]]]), 0, 0
        ),
        (np.array([
            [[ 0,  0],
            [29, 29]],
            [[31, 41],
            [ 2, 33]]]), 0, 1
        ),
        (np.array([
            [[33,  0],
            [90, 90]],
            [[ 0,  0],
            [ 9, 57]]]), 1, 0
        ),
        (np.array([
            [[ 0,  0],
            [75,  0]],
            [[ 0,  0],
            [95, 95]]]), 1, 1
        )
    ]

    p = Pooling2()
    divided_parts = list(p.divide_input(filters))
    assert len(divided_parts) == 4
    for res, expected in zip(divided_parts, expected_parts):
        _assert_divided_parts(res, expected)


def test_pool():
    filter1 = np.array([
        [0, 50, 0, 29],
        [0, 80, 31, 2],
        [33, 90, 0, 75],
        [0, 9, 0, 95],
    ])
    filter1 = filter1[:, :, np.newaxis]
    filter2 = np.array([
        [0, 55, 0, 29],
        [20, 0, 41, 33],
        [0, 90, 0, 0],
        [0, 57, 0, 95],
    ])
    filter2 = filter2[:, :, np.newaxis]
    filters = np.concatenate((filter1, filter2), axis=2)

    p = Pooling2()
    res = p.pool(filters)
    expected = np.array([
        [
            [80, 55],
            [31, 41]
        ],
        [
            [90, 90],
            [95, 95]
        ]
    ])
    np.testing.assert_array_equal(res, expected)
