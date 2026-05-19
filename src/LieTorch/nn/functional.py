import numpy as np

from ..tensor.tensor import Tensor
from ..autograd.grad_function import CrossEntropyBackward, DropoutBackward, ExpBackward, MeanBackward, ReLUBackward, SoftmaxBackward, SumBackward
from ..autograd.grad_state import grad_enabled


def relu(a: Tensor) -> Tensor:
    requires_grad = grad_enabled() and a.requires_grad
    t = Tensor(np.maximum(0, a._data), requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = ReLUBackward(a)
    return t

def dropout(a: Tensor, p: float, training: bool) -> Tensor:
    if not training:
        return a
    mask = (np.random.rand(*a.shape) >= p).astype(float)
    requires_grad = grad_enabled() and a.requires_grad
    t = Tensor(a._data * mask, requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = DropoutBackward(a, p, training)
    return t

def softmax(a: Tensor) -> Tensor:
    x = a._data
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    probs = exp_x / sum_exp_x

    requires_grad = grad_enabled() and a.requires_grad
    t = Tensor(probs, requires_grad=requires_grad)
    if requires_grad:
        t.grad_fn = SoftmaxBackward(a)
    return t

def argmax(a: Tensor, axis=None, keepdims=False) -> Tensor:
    return Tensor(np.argmax(a._data, axis=axis, keepdims=keepdims), requires_grad=False)

def cross_entropy(logits: Tensor, target: Tensor, reduction='mean') -> Tensor:
    x = logits._data
    x_max = np.max(x, axis=1, keepdims=True)
    logsumexp = x_max + np.log(np.sum(np.exp(x - x_max), axis=1, keepdims=True))
    log_probs = x - logsumexp

    losses = -log_probs[np.arange(target._data.shape[0]), target._data]

    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    
    requires_grad = grad_enabled() and logits.requires_grad
    losses = Tensor(losses, requires_grad=requires_grad)
    if requires_grad:
        losses.grad_fn = CrossEntropyBackward(logits, target, reduction)
    return losses
