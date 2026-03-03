import numpy as np
from ..core.tensor import Tensor
from layer import Layer

class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        weight = (np.random.rand(vocab_size, dim) - 0.5)/self.dim
        self.weight = Tensor(weight, autograd=True)

        self.parameters.append(self.weight)
    def forward(self, input):
        return self.weight.index_select(input)