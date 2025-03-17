import numpy as np
from INF2475.math_helpers import scaled_tanh, scaled_tanh_derivative, tanh, tanh_derivative
from INF2475.neuron import Neuron
from INF2475.perceptron import Perceptron
from INF2475.nn import SimpleNN

def test_neuron_forward():
    neuron = Neuron(input_size=2, lr=0.1, activation_fn=tanh, activation_derivative_fn=tanh_derivative)
    inputs = [0.5, -0.5]
    output = neuron.forward(inputs)
    
    # Check that the forward pass sets last_input and last_z correctly
    assert neuron.last_input == inputs, "Forward pass did not store last_input correctly."
    assert neuron.last_z is not None, "Forward pass did not store last_z."
    
    # Verify that the output equals tanh(last_z)
    expected_output = tanh(neuron.last_z)
    assert abs(output - expected_output) < 1e-6, "Forward pass output does not match expected activation."
    
    print("test_neuron_forward passed.")

def test_perceptron_xor_failure():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])
    p = Perceptron(input_size=2, lr=0.1, epochs=10)
    p.train(X, y)
    predictions = [p.predict(xi) for xi in X]
    # We expect the perceptron to fail at learning XOR
    # XOR is not linearly separable and needs more than one perceptron to be solved (neural network)
    assert predictions != [0, 1, 1, 0], f"Unexpectedly, the perceptron solved XOR! Predictions: {predictions}"

def test_perceptron_and():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1])
    p = Perceptron(input_size=2, lr=0.1, epochs=10)
    p.train(X, y)
    predictions = [p.predict(xi) for xi in X]
    assert predictions == [0, 0, 0, 1], f"Expected [0, 0, 0, 1] but got {predictions}"

def test_perceptron_or():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 1])
    p = Perceptron(input_size=2, lr=0.1, epochs=10)
    p.train(X, y)
    predictions = [p.predict(xi) for xi in X]
    assert predictions == [0, 1, 1, 1], f"Expected [0, 1, 1, 1] but got {predictions}"    

def test_nn_and():
    # Define the dataset for the AND function with floats
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    y = [0.0, 0.0, 0.0, 1.0]
    
    nn = SimpleNN(input_size=2, hidden_size=2, output_size=1, lr=0.1, epochs=1000)
    nn.train(X, y)
    predictions = nn.predict(X)
    # predictions is already a list of integers, so we can assert directly
    assert predictions == [0, 0, 0, 1], f"Expected [0, 0, 0, 1] but got {predictions}"

def test_nn_or():
    # Define the dataset for the OR function with floats
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    y = [0.0, 1.0, 1.0, 1.0]
    
    nn = SimpleNN(input_size=2, hidden_size=2, output_size=1, lr=0.1, epochs=1000)
    nn.train(X, y)
    predictions = nn.predict(X)
    # predictions is already a list of integers, so we can assert directly
    assert predictions == [0, 1, 1, 1], f"Expected [0, 1, 1, 1] but got {predictions}"

def test_nn_xor():
    # Define the dataset for the XOR function with floats
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    y = [0.0, 1.0, 1.0, 0.0]
    
    nn = SimpleNN(input_size=2, hidden_size=3, lr=0.1, epochs=1000,
              neuron_factory=lambda size, lr: Neuron(size, lr, tanh, tanh_derivative))
    nn.train(X, y)
    predictions = nn.predict(X)
    # predictions is already a list of integers, so we can assert directly
    assert predictions == [0, 1, 1, 0], f"Expected [0, 1, 1, 0] but got {predictions}"    