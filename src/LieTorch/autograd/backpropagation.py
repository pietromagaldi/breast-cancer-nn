def run_backward(tensor):
    import numpy as np
    stack = [(tensor, np.ones_like(tensor._data))]  # (node, gradient)

    while stack:
        t, grad = stack.pop()

        if t.retain_grad or t.grad_fn is None:
            if t.grad is None:
                t.grad = grad
            else:
                t.grad += grad

        if t.grad_fn is not None:
            grads = t.grad_fn(grad)

            for parent, g in zip(t.grad_fn.parents, grads):
                stack.append((parent, g))

            if not t.retain_grad:
                t.grad_fn.parents = None
                t.grad_fn = None