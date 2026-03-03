class SGD():
    def __init__(self, parameters, alpha = 0.1):
        self.parameters = parameters
        self.alpha = alpha
    def zero(self):
        for params in self.parameters:
            params.grad.data *= 0
    def step(self, zero=True):
        for params in self.parameters:
            params.data -= params.grad.data*self.alpha
            if(zero):
                params.grad.data *= zero