"""
Copyright (c) 2019, Matt Pewsey
"""

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
    def __new__(cls, x, y, z=0, radius=0):
        obj = np.array([x, y, z], dtype='float').view(cls)
        obj.radius = radius
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.radius = getattr(obj, 'radius', 0)

    def x():
        def fget(self):
            return self[0]
        def fset(self, value):
            self[0] = value
        return locals()
    x = property(**x())

    def y():
        def fget(self):
            return self[1]
        def fset(self, value):
            self[1] = value
        return locals()
    y = property(**y())

    def z():
        def fget(self):
            return self[2]
        def fset(self, value):
            self[2] = value
        return locals()
    z = property(**z())

    def __repr__(self):
        s = [
            ('x', self.x),
            ('y', self.y),
            ('z', self.z),
            ('radius', self.radius),
        ]

        s = ', '.join('{}={!r}'.format(x, y) for x, y in s)
        return '{}({})'.format(type(self).__name__, s)
