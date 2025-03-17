from typing import Callable, List, Optional
import numpy as np
from INF2475.math_helpers import sigmoid, sigmoid_derivative
from INF2475.neuron import Neuron

# A simple neural network built using our Neuron class.
class SimpleNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, 
                 lr: float = 0.1, epochs: int = 2000,
                 neuron_factory: Optional[Callable[[int, float], Neuron]] = None):
        self.lr = lr
        self.epochs = epochs
        # Use the provided neuron_factory to create neurons
        # If none is provided, default to sigmoid activation
        if neuron_factory is None:
            neuron_factory = lambda size, lr: Neuron(size, lr, sigmoid, sigmoid_derivative)
        self.hidden_layer: List[Neuron] = [neuron_factory(input_size, lr) for _ in range(hidden_size)]
        self.output_layer: List[Neuron] = [neuron_factory(hidden_size, lr) for _ in range(output_size)]
    
    def forward(self, inputs: List[float]):
        # Forward pass through hidden layer
        hidden_outputs = [neuron.forward(inputs) for neuron in self.hidden_layer]
        # Forward pass through output layer
        outputs = [neuron.forward(hidden_outputs) for neuron in self.output_layer]
        return hidden_outputs, outputs

    def train(self, X: List[List[float]], y: List[float]) -> None:
        for epoch in range(self.epochs):
            total_loss = 0.0
            for inputs, target in zip(X, y):
                # Forward pass
                hidden_outputs, outputs = self.forward(inputs)
                # For simplicity, we assume one output neuron.
                predicted = outputs[0]
                error = target - predicted
                total_loss += error ** 2

                # Backpropagation for the output neuron.
                output_neuron = self.output_layer[0]
                # Here, we pass the raw error; the neuron multiplies by the derivative internally.
                output_neuron.backward_update(error)
                
                # Backpropagate error to hidden layer:
                # Each hidden neuron's delta is influenced by the output neuron's weights.
                for i, hidden_neuron in enumerate(self.hidden_layer):
                    # Error contribution from the output neuron:
                    delta_hidden = error * output_neuron.weights[i+1]
                    hidden_neuron.backward_update(delta_hidden)
            
            # Optionally, print average loss every 10% of epochs.
            if epoch % (self.epochs // 10) == 0:
                avg_loss = total_loss / len(X)
                print(f"Epoch {epoch} Average Loss: {avg_loss}")

    def predict(self, X: List[List[float]]) -> List[int]:
        predictions = []
        for inputs in X:
            _, outputs = self.forward(inputs)
            # For one output neuron: threshold at 0.5
            predictions.append(1 if outputs[0] > 0.5 else 0)
        return predictions