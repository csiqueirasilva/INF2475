import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        # Initialize weights (including bias as weight[0])
        self.weights = np.random.randn(input_size + 1) * 0.1
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        # Step activation function: 1 if x >= 0 else 0
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Compute weighted sum (bias + dot product)
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(z)

    def train(self, X, y):
        for epoch in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                # Update weights and bias
                self.weights[1:] += self.lr * error * xi
                self.weights[0] += self.lr * error
            print(f"Epoch {epoch+1}/{self.epochs} - Weights: {self.weights}")

def run_perceptron_example():
    print("Running Perceptron Example (AND Function)")
    # AND function dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1])
    perceptron = Perceptron(input_size=2, lr=0.1, epochs=10)
    perceptron.train(X, y)
    for xi in X:
        print(f"Input: {xi} -> Prediction: {perceptron.predict(xi)}")