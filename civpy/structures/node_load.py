"""
Copyright (c) 2019, Matt Pewsey
"""

import weakref
import numpy as np

__all__ = ['NodeLoad']


class NodeLoad(np.ndarray):
    """
    A class representing a load applied to a node.

    Parameters
    ----------
    node : str
        The name of the node to which the load will be applied.
    fx, fy, fz : float
        The applied global node forces.
    mx, my, mz : float
        The applied global moments.
    dx, dy, dz : float
        The applied node deflections.
    rx, ry, rz : float
        The applied node rotations.
    """
    def __new__(cls, node, fx=0, fy=0, fz=0, mx=0, my=0, mz=0,
                dx=0, dy=0, dz=0, rx=0, ry=0, rz=0):
        obj = np.array([fx, fy, fz, mx, my, mz,
                        dx, dy, dz, rx, ry, rz], dtype='float').view(cls)
        obj.node = node
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.node = getattr(obj, 'node', '')
        self.node_ref = None

    def node():
        def fget(self):
            return self._node
        def fset(self, value):
            if not isinstance(value, str):
                value = str(value)
            self._node = value
        def fdel(self):
            del self._node
        return locals()
    node = property(**node())

    def node_ref():
        def fget(self):
            value = self._node_ref
            if value is None:
                return value
            return value()
        def fset(self, value):
            if value is not None:
                value = weakref.ref(value)
            self._node_ref = value
        def fdel(self):
            del self._node_ref
        return locals()
    node_ref = property(**node_ref())

    def fx():
        def fget(self):
            return self[0]
        def fset(self, value):
            self[0] = value
        return locals()
    fx = property(**fx())

    def fy():
        def fget(self):
            return self[1]
        def fset(self, value):
            self[1] = value
        return locals()
    fy = property(**fy())

    def fz():
        def fget(self):
            return self[2]
        def fset(self, value):
            self[2] = value
        return locals()
    fz = property(**fz())

    def mx():
        def fget(self):
            return self[3]
        def fset(self, value):
            self[3] = value
        return locals()
    mx = property(**mx())

    def my():
        def fget(self):
            return self[4]
        def fset(self, value):
            self[4] = value
        return locals()
    my = property(**my())

    def mz():
        def fget(self):
            return self[5]
        def fset(self, value):
            self[5] = value
        return locals()
    mz = property(**mz())

    def dx():
        def fget(self):
            return self[6]
        def fset(self, value):
            self[6] = value
        def fdel(self):
            del self._dx
        return locals()
    dx = property(**dx())

    def dy():
        def fget(self):
            return self[7]
        def fset(self, value):
            self[7] = value
        def fdel(self):
            del self._dy
        return locals()
    dy = property(**dy())

    def dz():
        def fget(self):
            return self[8]
        def fset(self, value):
            self[8] = value
        def fdel(self):
            del self._dz
        return locals()
    dz = property(**dz())

    def rx():
        def fget(self):
            return self[9]
        def fset(self, value):
            self[9] = value
        def fdel(self):
            del self._rx
        return locals()
    rx = property(**rx())

    def ry():
        def fget(self):
            return self[10]
        def fset(self, value):
            self[10] = value
        def fdel(self):
            del self._ry
        return locals()
    ry = property(**ry())

    def rz():
        def fget(self):
            return self[11]
        def fset(self, value):
            self[11] = value
        def fdel(self):
            del self._rz
        return locals()
    rz = property(**rz())

    def __repr__(self):
        s = [
            'node={!r}'.format(self.node),
            'forces={!r}'.format((self.fx, self.fy, self.fz)),
            'moments={!r}'.format((self.mx, self.my, self.mz)),
            'defl={!r}'.format((self.dx, self.dy, self.dz)),
            'rot={!r}'.format((self.rx, self.ry, self.rz))
        ]

        return '{}({})'.format(type(self).__name__, ', '.join(s))

    def forces(self):
        """Returns the applied force and moment matrix."""
        return self[:6]

    def deflections(self):
        """Returns the applied deflection and rotation matrix."""
        return self[6:]

    def get_node(self):
        """Gets the referenced node."""
        if self.node_ref is None:
            raise ValueError('Node has not been set.')
        return self.node_ref

    def set_node(self, ndict):
        """
        Sets the node reference.

        Parameters
        ----------
        ndict : dict
            A dictionary mapping node names to node objects.
        """
        self.node_ref = ndict[self.node]
