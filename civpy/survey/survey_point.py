"""
Copyright (c) 2019, Matt Pewsey
"""

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
    def __new__(cls, x, y, z, **kwargs):
        obj = np.array([x, y, z], dtype='float').view(cls)
        obj.meta = dict(**kwargs)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.meta = getattr(obj, 'meta', {})

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
            ('meta', self.meta),
        ]

        s = ', '.join('{}={!r}'.format(x, y) for x, y in s)
        return '{}({})'.format(type(self).__name__, s)
