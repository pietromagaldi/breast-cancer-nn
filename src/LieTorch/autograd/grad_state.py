import threading

_grad_state = threading.local()
_grad_state.enabled = True

def grad_enabled():
    return _grad_state.enabled

def enable_grad():
    _grad_state.enabled = True

def set_grad_enabled(enabled: bool):
    _grad_state.enabled = enabled

class no_grad:
    def __enter__(self):
        self.prev_state = _grad_state.enabled
        _grad_state.enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        _grad_state.enabled = self.prev_state

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper