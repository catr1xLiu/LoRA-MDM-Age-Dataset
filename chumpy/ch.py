"""
Minimal chumpy.ch module for SMPL compatibility.
"""
import numpy as np
import copyreg


def _ch_constructor(data, dtype=None):
    """Construct a Ch object from pickle data."""
    return Ch(data)


def _ch_reducer(obj):
    """Reduce a Ch object for pickling."""
    return (_ch_constructor, (np.asarray(obj), obj.dtype))


class Ch(np.ndarray):
    """Stub for chumpy's Ch class - wraps numpy arrays."""

    def __new__(cls, input_array=None, **kwargs):
        if input_array is None:
            input_array = np.array([])
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def r(self):
        """Return the underlying array value."""
        return np.array(self)


# Register pickle support for Ch class
copyreg.pickle(Ch, _ch_reducer, _ch_constructor)

# Compatibility aliases
array = Ch
