import copy
import propy
import numpy as np
from math import cos, sin
from functools import lru_cache

__all__ = [
    'rotation_matrix',
    'transformation_matrix',
    'local_stiffness',
    'clear_element_cache',
    'Element',
]


def rotation_matrix(dx, dy, dz, roll=0):
    """
    Returns the rotation matrix of shape (3, 3) for an element.

    Parameters
    ----------
    dx, dy, dz : float
        The distance changes from the i to j end of the element.
    roll : float
        The roll of the element counter clockwise about its axis.
    """
    l = (dx**2 + dy**2 + dz**2)**0.5
    rx, ry, rz = dx / l, dy / l, dz / l
    c, s = cos(-roll), sin(-roll)

    if rx == 0 and rz == 0:
        r = [[0, ry, 0], [-ry*c, 0, s], [ry*s, 0, c]]
    else:
        rxz = (rx**2 + rz**2)**0.5
        r = [[rx, ry, rz],
             [(-rx*ry*c - rz*s)/rxz, rxz*c, (-ry*rz*c + rx*s)/rxz],
             [(rx*ry*s - rz*c)/rxz, -rxz*s, (ry*rz*s + rx*c)/rxz]]

    return np.array(r, dtype='float')


@lru_cache(maxsize=1000)
def transformation_matrix(dx, dy, dz, roll=0):
    """
    Returns the transformation matrix of shape (12, 12) for an element.

    Parameters
    ----------
    dx, dy, dz : float
        The distance changes from the i to j end of the element.
    roll : float
        The roll of the element counter clockwise about its axis.
    """
    r = rotation_matrix(dx, dy, dz, roll)
    t = np.zeros((12, 12), dtype='float')
    t[:3,:3] = t[3:6,3:6] = t[6:9,6:9] = t[9:12,9:12] = r
    return t


@lru_cache(maxsize=1000)
def local_stiffness(l, lu, a, ix, iy, j, e, g,
                    imx_free=False, imy_free=False, imz_free=False,
                    jmx_free=False, jmy_free=False, jmz_free=False):
    """
    Returns the local stiffness matrix of shape (12, 12) of the element.

    Parameters
    ----------
    l : float
        The length of the element.
    lu : float
        The unstressed length of the element.
    a : float
        The cross sectional area of the element.
    ix, iy : float
        The moment of inertias about the local x and y axes of the element.
    j : float
        The polar moment of inertia of the element.
    e : float
        The modulus of elasticity of the element.
    g : float
        The modulus of rigidity of the element.
    imx_free, imy_free, imz_free : bool
        The i-end releases for the element.
    jmx_free, jmy_free, jmz_free : bool
        The j-end releases for the element.
    """
    k = np.zeros((12, 12), dtype='float')
    iy, iz = ix, iy
    f = e / l**3

    k[0, 0] = k[6, 6] = a*e/lu
    k[0, 6] = k[6, 0] = -k[0, 0]

    if not imz_free or not jmz_free:
        k[3, 3] = k[9, 9] = g*j/l
        k[9, 3] = k[3, 9] = -k[3, 3]

    if not imy_free:
        if not jmy_free:
            # Fixed-Fixed
            k[1, 1] = k[7, 7] = f*12*iz
            k[1, 7] = k[7, 1] = -k[1, 1]
            k[5, 5] = k[11, 11] = f*4*l**2*iz
            k[5, 11] = k[11, 5] = f*2*l**2*iz
            k[1, 5] = k[1, 11] = k[5, 1] = k[11 ,1] = f*6*l*iz
            k[5, 7] = k[7, 5] = k[7, 11] = k[11, 7] = -k[1, 5]
        else:
            # Fixed-Free
            k[1, 1] = k[7, 7] = f*3*iz
            k[1, 7] = k[7, 1] = -k[1, 1]
            k[1, 5] = k[5, 1] = f*3*l*iz
            k[5, 7] = k[7, 5] = -k[1, 5]
            k[5, 5] = f*3*l**2*iz
    elif not jmy_free:
        # Free-Fixed
        k[1, 1] = k[7, 7] = f*3*iz
        k[1, 7] = k[7, 1] = -k[1, 1]
        k[1, 11] = k[11 ,1] = f*3*l*iz
        k[7, 11] = k[11, 7] = -k[1, 11]
        k[11, 11] = f*3*l**2*iz

    if not imx_free:
        if not jmx_free:
            # Fixed-Fixed
            k[2, 2] = k[8, 8] = f*12*iy
            k[2, 8] = k[8, 2] = -k[2, 2]
            k[4, 8] = k[8, 4] = k[10, 8] = k[8, 10] = f*6*l*iy
            k[2, 4] = k[2, 10] = k[4, 2] = k[10, 2] = -k[4, 8]
            k[4, 4] = k[10, 10] = f*4*l**2*iy
            k[4, 10] = k[10, 4] = f*2*l**2*iy
        else:
            # Fixed-Free
            k[2, 2] = k[8, 8] = f*3*iy
            k[2, 8] = k[8, 2] = -k[2, 2]
            k[4, 8] = k[8, 4] = f*3*l*iy
            k[2, 4] = k[4, 2] = -k[4, 8]
            k[4, 4] = 3*l**2*iy
    elif not jmx_free:
        # Free-Fixed
        k[2, 2] = k[8, 8] = f*3*iy
        k[2, 8] = k[8, 2] = -k[2, 2]
        k[8, 10] = k[10, 8] = f*3*l*iy
        k[2, 10] = k[10, 2] = -k[8, 10]
        k[10, 10] = f*3*l**2*iy

    return k


def clear_element_cache():
    """Clears the element function cache."""
    local_stiffness.cache_clear()
    transformation_matrix.cache_clear()


class Element(object):
    """
    A class representing a structural element.

    Parameters
    ----------
    name : str
        A unique name for the element.
    inode, jnode : str
        The names of the nodes at the i and j ends of the element.
    group : :class:`.ElementGroup`
        The group assigned to the element.
    symmetry : {None, 'x', 'y', 'xy'}
        The symmetry of the element.
    roll : float
        The counter clockwise angle of roll about the length axis.
    imx_free, imy_free, imz_free : bool
        The rotational fixities at the i-node about the local x, y, and z axes.
    jmx_free, jmy_free, jmz_free : bool
        The rotational fixities at the j-node about the local x, y, and z axes.
    """
    SYMMETRIES = (None, 'x', 'y', 'xy')

    TRANSFORMS = {
        'x': {'p': 'x', 'x': 'p', 'y': 'xy', 'xy': 'y'},
        'y': {'p': 'y', 'x': 'xy', 'y': 'p', 'xy': 'x'},
    }

    # Custom properties
    name = propy.str_property('name')
    inode = propy.str_property('inode_name')
    jnode = propy.str_property('jnode_name')
    symmetry = propy.enum_property('symmetry', set(SYMMETRIES))

    imx_free = propy.bool_property('imx_free')
    imy_free = propy.bool_property('imy_free')
    imz_free = propy.bool_property('imz_free')

    jmx_free = propy.bool_property('jmx_free')
    jmy_free = propy.bool_property('jmy_free')
    jmz_free = propy.bool_property('jmz_free')

    inode_ref = propy.weakref_property('inode_ref')
    jnode_ref = propy.weakref_property('jnode_ref')

    def __init__(self, name, inode, jnode, group,
                 symmetry=None, roll=0, unstr_length=None,
                 imx_free=False, imy_free=False, imz_free=False,
                 jmx_free=False, jmy_free=False, jmz_free=False):
        self.name = name
        self.inode = inode
        self.jnode = jnode
        self.group = group
        self.symmetry = symmetry
        self.roll = roll
        self.unstr_length = unstr_length

        self.imx_free = imx_free
        self.imy_free = imy_free
        self.imz_free = imz_free

        self.jmx_free = jmx_free
        self.jmy_free = jmy_free
        self.jmz_free = jmz_free

        self.inode_ref = None
        self.jnode_ref = None

    __repr__ = propy.repr_method(
        'name', 'inode', 'jnode', 'group', 'symmetry', 'roll',
        'imx_free', 'imy_free', 'imz_free', 'jmx_free', 'jmy_free', 'jmz_free'
    )

    def __str__(self):
        return self.name

    def copy(self):
        """Returns a copy of the element."""
        return copy.copy(self)

    def i_free(self):
        """Sets the i end rotational fixities to free. Returns the element."""
        self.imx_free = self.imy_free = self.imz_free = True
        return self

    def j_free(self):
        """Sets the j end rotational fixities to free. Returns the element."""
        self.jmx_free = self.jmy_free = self.jmz_free = True
        return self

    def free(self):
        """Sets the end rotational fixities to free. Returns the element."""
        return self.i_free().j_free()

    def mx_free(self):
        """Sets the x rotational fixities to free. Returns the element."""
        self.imx_free = self.jmx_free = True
        return self

    def my_free(self):
        """Sets the y rotational fixities to free. Returns the element."""
        self.imy_free = self.jmy_free = True
        return self

    def mz_free(self):
        """Sets the z rotational fixities to free. Returns the element."""
        self.imz_free = self.jmz_free = True
        return self

    def set_nodes(self, ndict):
        """
        Sets the node references for the element.

        Parameters
        ----------
        ndict : dict
            A dictionary that maps node names to :class:`.Node` objects.
        """
        self.inode_ref = ndict[self.inode]
        self.jnode_ref = ndict[self.jnode]

    def get_nodes(self):
        """Returns the i and j node objects."""
        if self.inode_ref is None or self.jnode_ref is None:
            raise ValueError('Node references have not been set.')
        return self.inode_ref, self.jnode_ref

    def get_unstr_length(self):
        """
        If the unstressed length of the element is None, returns the initial
        distance between the nodes. If the unstressed length is a string,
        converts the string to a float and adds it to the initial distance
        between the nodes. Otherwise, returns the assigned unstressed length.
        """
        if self.unstr_length is None:
            return self.length()

        elif isinstance(self.unstr_length, str):
            return self.length() + float(self.unstr_length)

        return self.unstr_length

    def length(self, di=(0, 0, 0), dj=(0, 0, 0)):
        """
        Returns the length of the element between nodes.

        Parameters
        ----------
        di, dj : array
            The deflections at the i and j ends of the element.
        """
        xi, xj = self.get_nodes()
        di, dj = np.asarray(di), np.asarray(dj)
        delta = (xj - xi) + (dj - di)
        return np.linalg.norm(delta)

    def sym_elements(self):
        """Returns the symmetric elements for the element."""
        def trans(name, *sym):
            t = Element.TRANSFORMS
            n = name.split('_')

            for x in sym:
                n[-1] = t[x][n[-1]]

            return '_'.join(n)

        def primary():
            e = self.copy()
            e.name = '{}_p'.format(self.name)
            return e

        def x_sym():
            e = self.copy()
            e.name = '{}_x'.format(self.name)
            e.inode = trans(self.inode, 'x')
            e.jnode = trans(self.jnode, 'x')
            return e

        def y_sym():
            e = self.copy()
            e.name = '{}_y'.format(self.name)
            e.inode = trans(self.inode, 'y')
            e.jnode = trans(self.jnode, 'y')
            return e

        def xy_sym():
            e = self.copy()
            e.name = '{}_xy'.format(self.name)
            e.inode = trans(self.inode, 'x', 'y')
            e.jnode = trans(self.jnode, 'x', 'y')
            return e

        if self.symmetry is None:
            return primary(),

        elif self.symmetry == 'x':
            return primary(), x_sym()

        elif self.symmetry == 'y':
            return primary(), y_sym()

        elif self.symmetry == 'xy':
            return primary(), x_sym(), y_sym(), xy_sym()

    def rotation_matrix(self, di=(0, 0, 0), dj=(0, 0, 0)):
        """
        Returns the rotation matrix for the element.

        Parameters
        ----------
        di, dj : array
            The deflections at the i and j ends of the element.
        """
        xi, xj = self.get_nodes()
        di, dj = np.asarray(di), np.asarray(dj)
        dx, dy, dz = (xj - xi) + (dj - di)
        return rotation_matrix(dx, dy, dz, self.roll)

    def transformation_matrix(self, di=(0, 0, 0), dj=(0, 0, 0)):
        """
        Returns the transformation matrix for the element.

        Parameters
        ----------
        di, dj : array
            The deflections at the i and j ends of the element.
        """
        xi, xj = self.get_nodes()
        di, dj = np.asarray(di), np.asarray(dj)
        dx, dy, dz = (xj - xi) + (dj - di)
        return transformation_matrix(dx, dy, dz, self.roll)

    def local_stiffness(self, di=(0, 0, 0), dj=(0, 0, 0)):
        """
        Returns the local stiffness for the element.

        Parameters
        ----------
        di, dj : array
            The deflections at the i and j ends of the element.
        """
        group = self.group
        sect = group.section
        mat = group.material

        return local_stiffness(
            l=self.length(di, dj),
            lu=self.get_unstr_length(),
            a=sect.area,
            ix=sect.inertia_x,
            iy=sect.inertia_y,
            j=sect.inertia_j,
            e=mat.elasticity,
            g=mat.rigidity,
            imx_free=self.imx_free,
            imy_free=self.imy_free,
            imz_free=self.imz_free,
            jmx_free=self.jmx_free,
            jmy_free=self.jmy_free,
            jmz_free=self.jmz_free
        )

    def global_stiffness(self, di=(0, 0, 0), dj=(0, 0, 0)):
        """
        Returns the global stiffness matrix for the element.

        Parameters
        ----------
        di, dj : array
            The deflections at the i and j ends of the element.
        """
        di, dj = np.asarray(di), np.asarray(dj)
        t = self.transformation_matrix(di, dj)
        k = self.local_stiffness(di, dj)
        return t.T.dot(k).dot(t)
