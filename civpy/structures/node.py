import copy
import propy
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
    SYMMETRIES = (None, 'x', 'y', 'xy')

    # Custom properties
    name = propy.str_property('name')
    x = propy.index_property(0)
    y = propy.index_property(1)
    z = propy.index_property(2)
    symmetry = propy.enum_property('symmetry', set(SYMMETRIES))

    fx_free = propy.bool_property('fx_free')
    fy_free = propy.bool_property('fy_free')
    fz_free = propy.bool_property('fz_free')

    mx_free = propy.bool_property('mx_free')
    my_free = propy.bool_property('my_free')
    mz_free = propy.bool_property('mz_free')

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

    __repr__ = propy.repr_method('name', 'x', 'y', 'z', 'symmetry',
        'fx_free', 'fy_free', 'fz_free', 'mx_free', 'my_free', 'mz_free')

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
