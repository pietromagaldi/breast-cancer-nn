class SGD:
    def __init__(self, parameters: list, lr: float = 0.01, momentum: float = 0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum_factor = momentum
        self.momentum_buffer = [0.0 for _ in range(len(parameters))]

    def step(self):
        for i, param in enumerate(self.parameters.values()):
            if param.grad is not None:
                momentum = self.momentum_factor * self.momentum_buffer[i] + param.grad 
                self.momentum_buffer[i] = momentum
                param._data -= self.lr * momentum
    
    def zero_grad(self):
        for param in self.parameters.values():
            param.grad = None
