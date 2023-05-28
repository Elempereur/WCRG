import torch

def partial_reshape(x, new_shape, start=None, stop=None):
    """ Reshape a range of axes into a new shape: (prev, mid, end) to (prev, mid', end). """
    if start is None:
        start = 0
    if stop is None:
        stop = x.ndim
    return x.reshape(x.shape[:start] + new_shape + x.shape[stop:])
