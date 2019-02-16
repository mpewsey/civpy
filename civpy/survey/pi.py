import propy
import numpy as np

__all__ = ['PI']


class PI(np.ndarray):
    """
    A class representing a point of intersection (PI) of an alignment.

    Parameters
    ----------
    x, y, z : float
        The x, y, and z coordinates.
    radius : float
        The radius of the horizontal curve. Use zero if a curve does not
        exist.
    """
    # Custom properties
    x = propy.index_property(0)
    y = propy.index_property(1)
    z = propy.index_property(2)

    def __new__(cls, x, y, z=0, radius=0):
        obj = np.array([x, y, z], dtype='float').view(cls)
        obj.radius = radius
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.radius = getattr(obj, 'radius', 0)

    __repr__ = propy.repr_method('x', 'y', 'z', 'radius')
