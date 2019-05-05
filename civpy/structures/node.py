"""
Copyright (c) 2019, Matt Pewsey
"""

import copy
import numpy as np

__all__ = ['Node']


class Node(np.ndarray):
    """
    A class representing a structural node.

    Parameters
    ----------
    name : str
        A unique name for the node.
    x, y, z : float
        The x, y, and z coordinates of the node.
    symmetry : {None, 'x', 'y', 'xy'}
        The symmetry of the node.
    fx_free, fy_free, fz_free : bool
        The force fixities of the node in the x, y, and z directions.
    mx_free, my_free, mz_free : bool
        The moment fixities of the node about the x, y, and z axes.
    """
    X = 'x'
    Y = 'y'
    XY = 'xy'
    SYMMETRIES = (None, X, Y, XY)

    def __new__(cls, name, x=0, y=0, z=0, symmetry=None,
                fx_free=True, fy_free=True, fz_free=True,
                mx_free=True, my_free=True, mz_free=True):
        obj = np.array([x, y, z], dtype='float').view(cls)

        obj.name = name
        obj.symmetry = symmetry

        obj.fx_free = fx_free
        obj.fy_free = fy_free
        obj.fz_free = fz_free

        obj.mx_free = mx_free
        obj.my_free = my_free
        obj.mz_free = mz_free

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

        self.name = getattr(obj, 'name', '')
        self.symmetry = getattr(obj, 'symmetry', None)

        self.fx_free = getattr(obj, 'fx_free', True)
        self.fy_free = getattr(obj, 'fy_free', True)
        self.fz_free = getattr(obj, 'fz_free', True)

        self.mx_free = getattr(obj, 'mx_free', True)
        self.my_free = getattr(obj, 'my_free', True)
        self.mz_free = getattr(obj, 'mz_free', True)

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

    def symmetry():
        def fget(self):
            return self._symmetry
        def fset(self, value):
            if value not in self.SYMMETRIES:
                raise ValueError('Type {!r} must be in {!r}.'.format(value, self.SYMMETRIES))
            self._symmetry = value
        def fdel(self):
            del self._symmetry
        return locals()
    symmetry = property(**symmetry())

    def __repr__(self):
        s = [
            ('name', self.name),
            ('x', self.x),
            ('y', self.y),
            ('z', self.z),
            ('symmetry', self.symmetry),
            ('fx_free', self.fx_free),
            ('fy_free', self.fy_free),
            ('fz_free', self.fz_free),
            ('mx_free', self.mx_free),
            ('my_free', self.my_free),
            ('mz_free', self.mz_free),
        ]

        s = ', '.join('{}={!r}'.format(x, y) for x, y in s)
        return '{}({})'.format(type(self).__name__, s)

    def __str__(self):
        return self.name

    def copy(self):
        """Returns a copy of the node."""
        return copy.copy(self)

    def f_fixed(self):
        """Sets the node force reactions to fixed."""
        self.fx_free = self.fy_free = self.fz_free = False
        return self

    def m_fixed(self):
        """Sets the node moment reactions to fixed."""
        self.mx_free = self.my_free = self.mz_free = False
        return self

    def fixed(self):
        """Sets the node force and moment reactions to fixed."""
        return self.f_fixed().m_fixed()

    def fixities(self):
        """Returns the force and moment fixities for the node."""
        return [self.fx_free, self.fy_free, self.fz_free,
                self.mx_free, self.my_free, self.mz_free]

    def sym_nodes(self):
        """Returns the symmetric nodes for the node."""
        def primary():
            n = self.copy()
            n.name = '{}_p'.format(self.name)
            return n

        def x_sym():
            n = self.copy()
            n.name = '{}_x'.format(self.name)
            n[1] *= -1
            return n

        def y_sym():
            n = self.copy()
            n.name = '{}_y'.format(self.name)
            n[0] *= -1
            return n

        def xy_sym():
            n = self.copy()
            n.name = '{}_xy'.format(self.name)
            n[:2] *= -1
            return n

        if self.symmetry is None:
            return primary(),

        elif self.symmetry == 'x':
            return primary(), x_sym()

        elif self.symmetry == 'y':
            return primary(), y_sym()

        elif self.symmetry == 'xy':
            return primary(), x_sym(), y_sym(), xy_sym()
