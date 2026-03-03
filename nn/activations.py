from layer import Layer


class Tanh(Layer):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.tanh()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.sigmoid()