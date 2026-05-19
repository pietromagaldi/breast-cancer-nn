from collections import OrderedDict

import numpy as np
from ..tensor.tensor import Tensor, zeros
from .functional import dropout, relu, softmax

class Parameter(Tensor):
    pass

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value

        elif isinstance(value, Module):
            self._modules[name] = value

        object.__setattr__(self, name, value)

    def add_module(self, name: str, module: Module):
        if not isinstance(module, Module):
            raise ValueError(f"Expected Module instance, got {type(module)}")
        elif name in self._modules:
            raise ValueError(f"Module with name '{name}' already exists")
        elif name in self._parameters:
            raise ValueError(f"Parameter with name '{name}' already exists")
        setattr(self, name, module)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method not implemented")
    
    def parameters(self):
        params = dict(self._parameters)

        for module_name, module in self._modules.items():
            for param_name, param in module.parameters().items():
                full_name = f"{module_name}.{param_name}"
                params[full_name] = param

        return params
    
    def train(self):
        self._training = True
    
    def eval(self):
        self._training = False

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, init: str = 'he'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if init in ('xavier', 'xavier_normal'):
            self._xavier_normal_init()
        elif init == 'xavier_uniform':
            self._xavier_uniform_init()
        elif init in ('lecun', 'lecun_normal'):
            self._lecun_normal_init()
        elif init == 'lecun_uniform':
            self._lecun_uniform_init()
        elif init in ('he', 'he_normal'):
            self._he_normal_init()
        elif init == 'he_uniform':
            self._he_uniform_init()
        elif init == 'zeros':
            self._zeros_init()
        else:
            raise ValueError(f"Unsupported initialization method: {init}")
        

    def _zeros_init(self):
        self.weight = Parameter(np.zeros((self.in_features, self.out_features)), requires_grad=True)
        self.bias = Parameter(np.zeros(self.out_features), requires_grad=True)

    def _lecun_normal_init(self):
        std = np.sqrt(1 / self.in_features)
        self.weight = Parameter(np.random.normal(0, std, (self.in_features, self.out_features)), requires_grad=True)
        self.bias = Parameter(np.zeros(self.out_features), requires_grad=True)

    def _lecun_uniform_init(self):
        limit = np.sqrt(3 / self.in_features)
        self.weight = Parameter(np.random.uniform(-limit, limit, (self.in_features, self.out_features)), requires_grad=True)
        self.bias = Parameter(np.zeros(self.out_features), requires_grad=True)

    def _xavier_normal_init(self):
        std = np.sqrt(2 / (self.in_features + self.out_features))
        self.weight = Parameter(np.random.normal(0, std, (self.in_features, self.out_features)), requires_grad=True)
        self.bias = Parameter(np.zeros(self.out_features), requires_grad=True)

    def _xavier_uniform_init(self):
        limit = np.sqrt(6 / (self.in_features + self.out_features))
        self.weight = Parameter(np.random.uniform(-limit, limit, (self.in_features, self.out_features)), requires_grad=True)
        self.bias = Parameter(np.zeros(self.out_features), requires_grad=True)

    def _he_normal_init(self):
        std = np.sqrt(2 / self.in_features)
        self.weight = Parameter(np.random.normal(0, std, (self.in_features, self.out_features)), requires_grad=True)
        self.bias = Parameter(np.zeros(self.out_features), requires_grad=True)

    def _he_uniform_init(self):
        limit = np.sqrt(6 / self.in_features)
        self.weight = Parameter(np.random.uniform(-limit, limit, (self.in_features, self.out_features)), requires_grad=True)
        self.bias = Parameter(np.zeros(self.out_features), requires_grad=True)

    def forward(self, x):
        return x @ self.weight + self.bias
    
class Sequential(Module):

    def __init__(self, *args):
        super().__init__()

        if len(args) == 1 and isinstance(args[0], dict):
            for name, module in args[0].items():
                self.add_module(name, module)
        else:
            for i, module in enumerate(args):
                self.add_module(f"layer_{i}", module)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class Softmax(Module):
    def forward(self, x):
        return softmax(x)


class Dropout(Module):
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x):
        return dropout(x, self.p, self.training)
    