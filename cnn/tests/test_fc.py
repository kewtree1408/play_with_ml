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
            2.878495e-12,
            9.043105e-07,
            3.226238e-10,
            4.065223e-14,
            9.022273e-09,
            8.888899e-09,
            3.146241e-09,
            3.353428e-08,
            3.780579e-04,
            9.996210e-01,
        ]
    )
    np.testing.assert_allclose(res, expected, rtol=1e-05)
