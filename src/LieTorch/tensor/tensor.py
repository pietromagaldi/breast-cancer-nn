from __future__ import annotations

import numpy as np

from ..autograd.grad_state import grad_enabled


class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False, retain_grad: bool = False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self._data: np.ndarray = data
        self.dtype: np.dtype = data.dtype
        self.shape: tuple = data.shape
        self.requires_grad: bool = requires_grad
        self.retain_grad: bool = retain_grad
        self.grad: np.ndarray = None
        self.grad_fn = None

    def __repr__(self):
        return f"T([{','.join(str(s) for s in self.shape)}], grad_fn={self.grad_fn})"

    def __add__(self, other) -> Tensor:
        return add(self, other)
        
    def __sub__(self, other) -> Tensor:
        return sub(self, other)
    
    def __mul__(self, other) -> Tensor:
        return mul(self, other)

    def __truediv__(self, other) -> Tensor:
        return div(self, other)

    def __matmul__(self, other) -> Tensor:
        return matmul(self, other)

    def backward(self, retain_graph=False):
        from ..autograd.backpropagation import run_backward
        run_backward(self)

    def detach(self) -> Tensor:
        return Tensor(self._data.copy(), requires_grad=False)
    
    def numpy(self) -> np.ndarray:
        if self.requires_grad:
            raise ValueError("Cannot convert a tensor that requires grad to numpy array. Detach it first.")
        return self._data.copy()
        
    def item(self):
        if self._data.size != 1:
            raise ValueError("Only one element tensors can be converted to Python scalars")
        return self._data.item()

def add(a: Tensor, b: Tensor) -> Tensor:
    from ..autograd.grad_function import AddBackward

    requires_grad = grad_enabled() and (a.requires_grad or b.requires_grad)
    t = Tensor(a._data + b._data, requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = AddBackward(a, b)
    return t

def sub(a: Tensor, b: Tensor) -> Tensor:
    from ..autograd.grad_function import SubBackward

    requires_grad = grad_enabled() and (a.requires_grad or b.requires_grad)
    t = Tensor(a._data - b._data, requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = SubBackward(a, b)
    return t

def mul(a: Tensor, b: Tensor) -> Tensor:
    from ..autograd.grad_function import MulBackward

    requires_grad = grad_enabled() and (a.requires_grad or b.requires_grad)
    t = Tensor(a._data * b._data, requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = MulBackward(a, b)    
    return t

def div(a: Tensor, b: Tensor) -> Tensor:
    from ..autograd.grad_function import DivBackward

    requires_grad = grad_enabled() and (a.requires_grad or b.requires_grad)
    t = Tensor(a._data / b._data, requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = DivBackward(a, b)    
    return t

def matmul(a: Tensor, b: Tensor) -> Tensor:
    from ..autograd.grad_function import MatMulBackward

    requires_grad = grad_enabled() and (a.requires_grad or b.requires_grad)
    t = Tensor(a._data @ b._data, requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = MatMulBackward(a, b)
    return t

def exp(a: Tensor) -> Tensor:
    from ..autograd.grad_function import ExpBackward

    requires_grad = grad_enabled() and a.requires_grad
    t = Tensor(np.exp(a._data), requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = ExpBackward(a)
    return t

def sum(a: Tensor, axis=None, keepdims=False) -> Tensor:
    from ..autograd.grad_function import SumBackward

    requires_grad = grad_enabled() and a.requires_grad
    t = Tensor(np.sum(a._data, axis=axis, keepdims=keepdims), requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = SumBackward(a, axis, keepdims)
    return t

def mean(a: Tensor, axis=None, keepdims=False) -> Tensor:
    from ..autograd.grad_function import MeanBackward

    requires_grad = grad_enabled() and a.requires_grad
    t = Tensor(np.mean(a._data, axis=axis, keepdims=keepdims), requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = MeanBackward(a, axis=axis, keepdims=keepdims)
    return t

def zeros(shape, dtype=np.float32, requires_grad=False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

def ones(shape, dtype=np.float32, requires_grad=False) -> Tensor:
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

def zeros_like(tensor: Tensor | np.ndarray) -> Tensor:
    if isinstance(tensor, Tensor):
        return Tensor(np.zeros_like(tensor._data), requires_grad=tensor.requires_grad)
    else:
        return Tensor(np.zeros_like(tensor), requires_grad=False)

def one_like(tensor: Tensor | np.ndarray) -> Tensor:
    if isinstance(tensor, Tensor):
        return Tensor(np.ones_like(tensor._data), requires_grad=tensor.requires_grad)
    else:
        return Tensor(np.ones_like(tensor), requires_grad=False)