import propy
import numpy as np

__all__ = ['SurveyPoint']


class SurveyPoint(np.ndarray):
    """
    A class representing a survey point.

    Parameters
    ----------
    x, y, z : float
        The x, y, and z coordinates.
    """
    # Custom properties
    x = propy.index_property(0)
    y = propy.index_property(1)
    z = propy.index_property(2)

    def __new__(cls, x, y, z, **kwargs):
        obj = np.array([x, y, z], dtype='float').view(cls)
        obj.meta = dict(**kwargs)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.meta = getattr(obj, 'meta', {})

    __repr__ = propy.repr_method('x', 'y', 'z', 'meta')
