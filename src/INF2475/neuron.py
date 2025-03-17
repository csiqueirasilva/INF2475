from typing import Callable, List, Optional
import numpy as np
from INF2475.perceptron import Perceptron

class Neuron(Perceptron):
    def __init__(self, input_size: int, lr: float, 
                 activation_fn: Callable[[float], float],
                 activation_derivative_fn: Callable[[float], float]):
        # No epochs needed for per-neuron update
        super().__init__(input_size, lr, epochs=0)
        self.activation_fn = activation_fn
        self.activation_derivative_fn = activation_derivative_fn
        self.last_input: Optional[List[float]] = None
        self.last_z: Optional[float] = None

    def forward(self, inputs: List[float]) -> float:
        self.last_input = inputs
        z = np.dot(inputs, self.weights[1:]) + self.weights[0]
        self.last_z = z
        output = self.activation_fn(z)
        print(f"Forward: inputs={inputs}, z={z}, output={output}")
        return output
    
    def backward_update(self, delta: float) -> None:
        assert self.last_input is not None, "Call forward() before backward_update()"
        assert self.last_z is not None, "last_z"
        deriv = self.activation_derivative_fn(self.last_z)
        delta_local = delta * deriv
        print(f"Backward: delta={delta}, last_z={self.last_z}, derivative={deriv}, delta_local={delta_local}")
        for i, input_val in enumerate(self.last_input):
            self.weights[i+1] += self.lr * delta_local * input_val
        self.weights[0] += self.lr * delta_local