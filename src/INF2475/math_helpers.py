import numpy as np

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: float) -> float:
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x: float) -> float:
    return np.tanh(x)

def tanh_derivative(x: float) -> float:
    return 1 - (np.tanh(x) ** 2)

def scaled_tanh(x: float) -> float:
    return (np.tanh(x) + 1) / 2

def scaled_tanh_derivative(x: float) -> float:
    return 0.5 * (1 - np.tanh(x) ** 2)