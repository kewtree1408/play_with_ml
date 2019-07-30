import numpy as np

from fc_layer import FullyConnectedLayer


def test_feedworward():
    filter1 = np.array([[0, 50, 0, 29], [0, 80, 31, 2], [33, 90, 0, 75], [0, 9, 0, 95]])
    filter1 = filter1[:, :, np.newaxis]
    filter2 = np.array([[0, 55, 0, 29], [20, 0, 41, 33], [0, 90, 0, 0], [0, 57, 0, 95]])
    filter2 = filter2[:, :, np.newaxis]
    filters = np.concatenate((filter1, filter2), axis=2)

    filters_amount = np.prod(filters.shape)
    pr_classes_amount = 10

    np.random.seed(0)
    fc = FullyConnectedLayer(filters_amount, pr_classes_amount)
    res = fc.feedforward(filters)
    expected = np.array(
        [
            0.00000000e000,
            1.92947847e-192,
            8.36569318e-304,
            0.00000000e000,
            3.82303629e-256,
            3.69844747e-257,
            9.69376399e-272,
            3.68103332e-241,
            1.15259295e-108,
            1.00000000e000,
        ]
    )
    np.testing.assert_allclose(res, expected)
