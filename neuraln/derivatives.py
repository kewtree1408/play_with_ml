import numpy as np
from activations import sigmoid


def sigmoid_deriv(x: float) -> float:
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    return sigmoid(x) * (1 - sigmoid(x))

