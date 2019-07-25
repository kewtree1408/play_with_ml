import numpy as np

from conv import Conv3x3


# Tests for this example:
# http://deeplearning.net/software/theano/_images/numerical_no_padding_no_strides.gif

def test_dot_sum():
    cnn_filter = np.array([
        [0, 1, 2],
        [2, 2, 0],
        [0, 1, 2]
    ])
    input_data = np.array([
        [3, 3, 2],
        [0, 0, 1],
        [3, 1, 2],
    ])
    assert Conv3x3.dot_sum(input_data, cnn_filter) == 12

def test_divide_input():
    input_data = np.array([
        [3, 3, 2, 1, 0],
        [0, 0, 1, 3, 1],
        [3, 1, 2, 2, 3],
        [2, 0, 0, 2, 2],
        [2, 0, 0, 0, 1],
    ])
    expected_shape = 9
    expected_parts = [
        (np.array([
            [3, 3, 2],
            [0, 0, 1],
            [3, 1, 2],
        ]), 0, 0),
        (np.array([
            [3, 2, 1],
            [0, 1, 3],
            [1, 2, 2],
        ]), 0, 1),
        (np.array([
            [2, 1, 0],
            [1, 3, 1],
            [2, 2, 3],
        ]), 0, 2),
        (np.array([
            [0, 0, 1],
            [3, 1, 2],
            [2, 0, 0],
        ]), 1, 0),
        (np.array([
            [0, 1, 3],
            [1, 2, 2],
            [0, 0, 2],
        ]), 1, 1),
        (np.array([
            [1, 3, 1],
            [2, 2, 3],
            [0, 2, 2],
        ]), 1, 2),
        (np.array([
            [3, 1, 2],
            [2, 0, 0],
            [2, 0, 0],
        ]), 2, 0),
        (np.array([
            [1, 2, 2],
            [0, 0, 2],
            [0, 0, 0],
        ]), 2, 1),
        (np.array([
            [2, 2, 3],
            [0, 2, 2],
            [0, 0, 1],
        ]), 2, 2),
    ]

    divided_parts = list(Conv3x3.divide_input(input_data))
    assert len(divided_parts) == expected_shape
    for res, expected in zip(divided_parts, expected_parts):
        part_result, i_result, j_result = res
        part_expected, i_expected, j_expected = expected
        assert i_result == i_expected
        assert j_result == j_expected
        np.testing.assert_array_equal(part_result, part_expected)


def test_forward_1_filter():
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
    output_result = cn.forward(input_data)
    np.testing.assert_array_equal(output_result, expected_output)


def test_forward_3_filters():
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
    output_result = cn.forward(input_data)
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
