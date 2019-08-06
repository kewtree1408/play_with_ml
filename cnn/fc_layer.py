# Fully connected layer

import numpy as np


class FullyConnectedLayer:
    """
    Fully Connected Layer with Softmax activation
    """

    def __init__(self, flatten_filters_amount, class_amount):
        self.weights = (
            np.random.randn(flatten_filters_amount, class_amount)
            / flatten_filters_amount
        )
        self.biases = np.random.randn(class_amount) / class_amount
        self.learn_rate = 0.005

        # variables for derivative calculus
        self.last_shape = None
        self.last_flat_input = None
        self.last_totals = None

    def feedforward(self, filters):
        self.last_shape = filters.shape

        flat_filters = filters.flatten()
        self.last_flat_input = flat_filters

        totals = np.dot(flat_filters.T, self.weights) + self.biases
        self.last_totals = totals

        e_xs = np.exp(totals)
        return e_xs / np.sum(e_xs)

    def _caclulate_gradients(self, idx, d_L_d_out):
        # the goal is to calculate:
        # d_L_d_w = d_L_d_out * d_out_d_t * d_t_d_w
        # d_L_d_b = d_L_d_out * d_out_d_t * d_t_d_b
        # d_L_d_input = d_L_d_out * d_out_d_t * d_t_d_input
        # Can replace d_L_d_t = d_L_d_out * d_out_d_t

        # t_exp = exp^totals
        t_exp = np.exp(self.last_totals)
        # Sum of all exp totals
        S = np.sum(t_exp)
        # tc_exp is total exp of the correct_class (i)
        tc_exp = t_exp[idx]

        # calculate gradients of out[i] against totals
        d_out_d_t = -tc_exp * t_exp / (S ** 2)
        # case when k == correct_class
        d_out_d_t[idx] = tc_exp * (S - tc_exp) / (S ** 2)

        # caclucate gradients of weights and biases
        # base on this equation: totals = w * input + b
        d_t_d_w = self.last_flat_input
        d_t_d_b = 1
        d_t_d_input = self.weights

        # calculate gradients of loss against totals
        d_L_d_t = d_L_d_out * d_out_d_t

        # caclulate gradients of loss against weights/biases/input
        d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
        d_L_d_b = d_L_d_t * d_t_d_b
        d_L_d_input = d_t_d_input @ d_L_d_t

        return d_L_d_w, d_L_d_b, d_L_d_input

    def backprop(self, gradient):
        # gradient is the one from previous layer (from class probability inputs)

        # d_L_d_out is derivative(Loss) / derivative(out)
        for i, d_L_d_out in enumerate(gradient):
            if d_L_d_out == 0:
                continue
            d_L_d_w, d_L_d_b, d_L_d_input = self._caclulate_gradients(i, d_L_d_out)
            # Update weights / biases
            self.weights -= self.learn_rate * d_L_d_w
            self.biases -= self.learn_rate * d_L_d_b
            # import ipdb; ipdb.set_trace()

        return d_L_d_input.reshape(self.last_shape)
