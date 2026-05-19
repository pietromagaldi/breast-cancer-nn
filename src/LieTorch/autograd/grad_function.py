import numpy as np
from  ..tensor.tensor import Tensor

def _reduce_to_shape(grad, shape):
    # 1. Sum out extra leading dimensions
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # 2. Sum along broadcasted dimensions
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

class GradFunction:    
    def __call__(self, *args, **kwargs):
        return self.grad(*args, **kwargs)
    
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def grad(self, grad_output: np.ndarray):
        raise NotImplementedError("Backward method not implemented")


class AddBackward(GradFunction):
    def __init__(self, a: Tensor, b: Tensor):
        super().__init__()
        self.parents = (a, b)
        self.a_shape = a.shape
        self.b_shape = b.shape

    def grad(self, grad_output: np.ndarray):
        grad_a = _reduce_to_shape(grad_output, self.a_shape)
        grad_b = _reduce_to_shape(grad_output, self.b_shape)
        return (grad_a, grad_b)

class SubBackward(GradFunction):
    def __init__(self, a: Tensor, b: Tensor):
        super().__init__()
        self.parents = (a, b)
        self.a_shape = a.shape
        self.b_shape = b.shape

    def grad(self, grad_output: np.ndarray):
        grad_a = _reduce_to_shape(grad_output, self.a_shape)
        grad_b = _reduce_to_shape(-grad_output, self.b_shape)
        return (grad_a, grad_b)

class MulBackward(GradFunction):
    def __init__(self, a: Tensor, b: Tensor):
        self.parents = (a, b)
        self.a_data = a._data
        self.b_data = b._data

    def grad(self, grad_output: np.ndarray):
        grad_a = _reduce_to_shape(grad_output * self.b_data, self.a_shape)
        grad_b = _reduce_to_shape(grad_output * self.a_data, self.b_shape)
        return (grad_a, grad_b)


class DivBackward(GradFunction):
    def __init__(self, a: Tensor, b: Tensor):
        self.parents = (a, b)
        self.a_data = a._data
        self.b_data = b._data

    def grad(self, grad_output: np.ndarray):
        grad_a = _reduce_to_shape(grad_output / self.b_data, self.a_shape)
        grad_b = _reduce_to_shape(-grad_output * self.a_data / (self.b_data ** 2), self.b_shape)
        return (grad_a, grad_b)

class ExpBackward(GradFunction):
    def __init__(self, a: Tensor):
        self.parents = (a,)
        self.a_data = a._data

    def grad(self, grad_output: np.ndarray):
        grad_a = grad_output * np.exp(self.a_data), self.a_data.shape
        return (grad_a,)

class MatMulBackward(GradFunction):
    def __init__(self, a: Tensor, b: Tensor):
        self.parents = (a, b)
        self.a_data = a._data
        self.b_data = b._data

    def grad(self, grad_output: np.ndarray):
        a_t = np.swapaxes(self.a_data, -1, -2)
        b_t = np.swapaxes(self.b_data, -1, -2)
        
        grad_a = _reduce_to_shape(grad_output @ b_t, self.a_data.shape)
        grad_b = _reduce_to_shape(a_t @ grad_output, self.b_data.shape)
        return (grad_a, grad_b)


class SumBackward(GradFunction):
    def __init__(self, a: Tensor, axis, keepdims):
        self.parents = (a,)
        self.shape = a.shape
        self.axis = axis
        self.keepdims = keepdims

    def grad(self, grad_output: np.ndarray):
        grad = grad_output

        # If keepdims=False, we need to reinsert the reduced axes
        if not self.keepdims and self.axis is not None:
            axes = self.axis
            if not isinstance(axes, tuple):
                axes = (axes,)

            for ax in sorted(axes):
                grad = np.expand_dims(grad, ax)

        # Now broadcast to original shape
        grad = np.broadcast_to(grad, self.shape)

        return (grad,)

class MeanBackward(GradFunction):
    def __init__(self, a: Tensor, axis=None, keepdims=False):
        self.parents = (a,)
        self.x_shape = a.shape
        self.axis = axis
        self.keepdims = keepdims
        if axis is None:
            self.count = np.prod(self.x_shape)
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            self.count = 1
            for ax in axes:
                self.count *= self.x_shape[ax]

    def grad(self, grad_output: np.ndarray):
        grad = grad_output / self.count

        # Same expansion logic as sum
        if not self.keepdims and self.axis is not None:
            axes = self.axis
            if not isinstance(axes, tuple):
                axes = (axes,)

            for ax in sorted(axes):
                grad = np.expand_dims(grad, ax)

        grad = np.broadcast_to(grad, self.x_shape)

        return (grad,)

class ReLUBackward(GradFunction):
    def __init__(self, a: Tensor):
        self.parents = (a,)
        self.positive_mask = (a._data > 0).astype(float)

    def grad(self, grad_output: np.ndarray):
        grad_a = grad_output * self.positive_mask
        return (grad_a,)

class SoftmaxBackward(GradFunction):
    def __init__(self, a: Tensor, softmax_output: Tensor):
        self.parents = (a,)
        self.softmax_output = softmax_output._data

    def grad(self, grad_output: np.ndarray):
        s = self.softmax_output
        grad_a = grad_output - np.sum(grad_output * s, axis=1, keepdims=True)
        grad_a *= s
        return (grad_a,)
    

class CrossEntropyBackward(GradFunction):
    def __init__(self, logits: Tensor, targets: Tensor, reduction: str):
        self.parents = (logits,)
        self.targets = targets._data
        self.logits = logits._data
        self.reduction = reduction
        self.N = logits.shape[0]

    def grad(self, grad_output):
        x = self.logits

        x_max = np.max(x, axis=1, keepdims=True)
        exp = np.exp(x - x_max)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        probs[np.arange(self.N), self.targets] -= 1

        if self.reduction == "mean":
            probs /= self.N

        if np.isscalar(grad_output):
            grad_logits = probs * grad_output
        else:
            grad_logits = probs * grad_output.reshape(-1, 1)

        return (grad_logits,)

class DropoutBackward(GradFunction):
    def __init__(self, a: Tensor, p: float, training: bool):
        self.parents = (a,)
        self.p = p
        self.training = training
        if training:
            self.mask = (np.random.rand(*a.shape) >= p).astype(float)
        else:
            self.mask = np.ones(a.shape)

    def grad(self, grad_output: np.ndarray):
        if self.training:
            grad_a = grad_output * self.mask
        else:
            grad_a = grad_output * (1 - self.p)
        return (grad_a,)