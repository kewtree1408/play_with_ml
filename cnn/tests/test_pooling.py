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
            [33, 90],
            [0, 9],
        ]), 1, 0),
        (np.array([
            [0, 29],
            [31, 2],
        ]), 0, 1),
        (np.array([
            [0, 75],
            [0, 95],
        ]), 1, 1),
    ]

    divided_parts = list(p.divide_input(input_data))
    import ipdb; ipdb.set_trace()
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
    expected_shape = (2, 2)
    expected_parts = [
        (np.array([
            [0, 50],
            [0, 80],
        ]), 0, 0),
        (np.array([
            [33, 90],
            [0, 9],
        ]), 1, 0),
        (np.array([
            [0, 29],
            [31, 2],
        ]), 0, 1),
        (np.array([
            [0, 75],
            [0, 95],
        ]), 1, 1),
    ]

    divided_parts = list(p.divide_input(input_data))
    assert len(divided_parts) == expected_shape
    for res, expected in zip(divided_parts, expected_parts):
        part_result, i_result, j_result = res
        part_expected, i_expected, j_expected = expected
        assert i_result == i_expected
        assert j_result == j_expected
        np.testing.assert_array_equal(part_result, part_expected)


def test_convolve_1_filter():
    filters_amount = 1 # depth
    cn = Conv3x3(filters_amount)
    input_data = np.array([
        [3, 3, 2, 1, 0],
        [0, 0, 1, 3, 1],
        [3, 1, 2, 2, 3],
        [2, 0, 0, 2, 2],
        [2, 0, 0, 0, 1],
    ])
    # reassign filter for easier testing
    cn.filters = np.array([[
        [0, 1, 2],
        [2, 2, 0],
        [0, 1, 2]
    ]])
    expected_output = np.array([
        [
            [12],
            [12],
            [17],
        ],
        [
            [10],
            [17],
            [19],
        ],
        [
            [9],
            [6],
            [14],
        ]
    ])
    output_result = cn.convolve(input_data)
    np.testing.assert_array_equal(output_result, expected_output)


def test_convolve_3_filters():
    filters_amount = 3 # depth
    cn = Conv3x3(filters_amount)
    # reassign filter for easier testing
    cn.filters = np.array([
        [
            [0, 1, 2],
            [2, 2, 0],
            [0, 1, 2]
        ],
        [
            [0, 1, 2],
            [2, 2, 0],
            [0, 1, 2]
        ],
        [
            [0, 1, 2],
            [2, 2, 0],
            [0, 1, 2]
        ],
    ])
    input_data = np.array([
        [3, 3, 2, 1, 0],
        [0, 0, 1, 3, 1],
        [3, 1, 2, 2, 3],
        [2, 0, 0, 2, 2],
        [2, 0, 0, 0, 1],
    ])
    output_result = cn.convolve(input_data)
    expected_output = np.array([
        [
            [12, 12, 12],
            [12, 12, 12],
            [17, 17, 17]
        ],
       [
            [10, 10, 10],
            [17, 17, 17],
            [19, 19, 19]
        ],
       [
            [ 9,  9,  9],
            [ 6,  6,  6],
            [14, 14, 14]
        ]
    ])
    np.testing.assert_array_equal(output_result, expected_output)
