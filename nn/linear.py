import numpy as np
from ..core.tensor import Tensor
from layer import Layer

class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.rand(n_inputs, n_outputs)*np.sqrt(2.0/(n_inputs))
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, x):
        return x.mm(self.weight) + self.bias.expand(0, len(x.data))Commit on 2026-02-02T12:00:00
Commit on 2026-02-20T12:00:00
