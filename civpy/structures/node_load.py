import propy
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
    # Custom properties
    node = propy.str_property('node')
    _node_ref = propy.weakref_property('_node_ref')
    fx = propy.index_property(0)
    fy = propy.index_property(1)
    fz = propy.index_property(2)
    mx = propy.index_property(3)
    my = propy.index_property(4)
    mz = propy.index_property(5)
    dx = propy.index_property(6)
    dy = propy.index_property(7)
    dz = propy.index_property(8)
    rx = propy.index_property(9)
    ry = propy.index_property(10)
    rz = propy.index_property(11)

    def __new__(cls, node, fx=0, fy=0, fz=0, mx=0, my=0, mz=0,
                dx=0, dy=0, dz=0, rx=0, ry=0, rz=0):
        obj = np.array([fx, fy, fz, mx, my, mz,
                        dx, dy, dz, rx, ry, rz], dtype='float').view(cls)
        obj.node = node
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.node = getattr(obj, 'node', '')
        self._node_ref = None

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
        if self._node_ref is None:
            raise ValueError('Node has not been set.')
        return self._node_ref

    def set_node(self, ndict):
        """
        Sets the node reference.

        Parameters
        ----------
        ndict : dict
            A dictionary mapping node names to node objects.
        """
        self._node_ref = ndict[self.node]
