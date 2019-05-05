"""
Copyright (c) 2019, Matt Pewsey
"""

import weakref
import numpy as np
from functools import lru_cache
from .element import transformation_matrix

__all__ = [
    'load_distances',
    'force_local_reactions',
    'moment_local_reactions',
    'local_reactions',
    'clear_element_load_cache',
    'ElementLoad',
]


def load_distances(dx, dy, dz, ix, delx):
    """
    Returns the load distances to where an element load is applied.

    Parameters
    ----------
    dx, dy, dz : float
        The element distance vector.
    ix, : float
        The distance from the i node of the element to where the beginning
        of the loads are applied.
    dx : float
        The distance from the ix position toward the j node over which
        the loads are applied.
    """
    l = (dx**2 + dy**2 + dz**2)**0.5

    l1 = l * abs(ix) if ix < 0 else ix
    l2 = l * abs(delx) if delx < 0 else delx
    l2 = l - l1 - l2

    if l1 > l or l1 < 0 or l2 > l or l2 < 0:
        raise ValueError('Load applied beyond element bounds.')

    return l, l1, l2


def force_local_reactions(fx, fy, fz, dx, dy, dz, roll, ix, delx):
    """
    Returns the local force reaction vector for an element.

    Parameters
    ----------
    fx, fy, fz : float
        The force vector.
    dx, dy, dz : float
        The element distance vector.
    roll : float
        The roll of the element.
    ix, : float
        The distance from the i node of the element to where the beginning
        of the loads are applied.
    dx : float
        The distance from the ix position toward the j node over which
        the loads are applied.
    """
    l, l1, l2 = load_distances(dx, dy, dz, ix, delx)

    t = transformation_matrix(dx, dy, dz, roll)

    if delx == 0:
        # Point load
        fsi = (l2**2 / l**3) * (3*l1 + l2)
        fmi = l1*l2**2 / l**2
        fsj = (l1**2 / l**3) * (l1 + 3*l2)
        fmj = -fmi
        fti = ftj = 0
        fai = l2 / l
        faj = l1 / l
    else:
        # Uniform load
        fsi = (l / 2) * (1 - (2*l**3 - 2*l1**2*l + l1**3)*l1/l**4 - (2*l - l2)*l2**3/l**4)
        fmi = (l**2 / 12) * (1 - (6*l**2 - 8*l1*l + 3*l1**2)*l1**2/l**4 - (4*l - 3*l2)*l2**3/l**4)
        fsj = (l / 2) * (1 - (2*l - l1)*l1**3/l**4 - (2*l**3 - 2*l2**2*l + l2**3)*l2/l**4)
        fmj = -(l**2 / 12) * (1 - (4*l - 3*l1)*l1**3/l**4 - (6*l**2 - 8*l2*l + 3*l2**2)*l2**2/l**4)
        fti = ftj = 0
        fai = (l / 2) * (l - l1 - l2) * (l - l1 + l2)
        faj = -fai

    fx, fy, fz = t[:3,:3].dot([fx, fy, fz])

    r = [-fx*fai, -fy*fsi, -fz*fsi, -fti, -fz*fmi, -fy*fmi,
         -fx*faj, -fy*fsj, -fz*fsj, -ftj, -fz*fmj, -fy*fmj]

    return np.array(r, dtype='float')


def moment_local_reactions(mx, my, mz, dx, dy, dz, roll, ix, delx):
    """
    Returns the local moment reaction vector for an element.

    Parameters
    ----------
    mx, my, mz : float
        The moment vector.
    dx, dy, dz : float
        The element distance vector.
    roll : float
        The roll of the element.
    ix, : float
        The distance from the i node of the element to where the beginning
        of the loads are applied.
    dx : float
        The distance from the ix position toward the j node over which
        the loads are applied.
    """
    l, l1, l2 = load_distances(dx, dy, dz, ix, delx)

    t = transformation_matrix(dx, dy, dz, roll)

    if delx == 0:
        # Point load
        fsi = -6*l1*l2 / l**3
        fmi = (l2 / l**2) * (l2 - 2*l1)
        fsj = -fsi
        fmj = (l1 / l**2) * (l1 - 2*l2)
        fti = l2 / l
        ftj = l1 / l
        fai = faj = 0
    else:
        # Uniform load
        fsi = 2*((l-l2)**3 - l1**3)/l**3 - 3*((l-l2)**2 - l1**2)/l**2
        fmi = ((l-l2) - l1) - 2*((l-l2)**2 - l1**2)/l + ((l-l2)**3 - l1**3)/l**2
        fsj = -fsi
        fmj = ((l-l2)**3 - l1**3)/l**2 - ((l-l2)**2 - l1**2)/l
        fti = ((l-l2) - l1) - ((l-l2)**2 - l1**2)/(2*l)
        ftj = ((l-l2)**2 - l1**2)/(2*l)
        fai = faj = 0

    mx, my, mz = t[:3,:3].dot([mx, my, mz])

    r = [-fai, -my*fsi, -mx*fsi, -mx*fti, -my*fmi, -mz*fmi,
         -faj, -my*fsj, -mx*fsj, -mx*ftj, -my*fmj, -mz*fmj]

    return np.array(r, dtype='float')


@lru_cache(maxsize=1000)
def local_reactions(fx, fy, fz, mx, my, mz, dx, dy, dz, roll, ix, delx,
                    imx_free, imy_free, imz_free, jmx_free, jmy_free, jmz_free):
    """
    Returns the local reaction vector for an element.

    Parameters
    ----------
    fx, fy, fz : float
        The force vector.
    mx, my, mz : float
        The moment vector.
    dx, dy, dz : float
        The element distance vector.
    roll : float
        The roll of the element.
    ix, : float
        The distance from the i node of the element to where the beginning
        of the loads are applied.
    dx : float
        The distance from the ix position toward the j node over which
        the loads are applied.
    imx_free, imy_free, imz_free : bool
        The fixities at the i end of the element.
    jmx_free, jmy_free, jmz_free : bool
        The fixities at the j end of the element.
    """
    r = force_local_reactions(fx, fy, fz, dx, dy, dz, roll, ix, delx)
    r += moment_local_reactions(mx, my, mz, dx, dy, dz, roll, ix, delx)

    # Adjust reactions for element end fixities
    if imz_free:
        if not jmz_free:
            # Free-Fixed
            r[9] += r[3]
            r[3] = 0
        else:
            # Free-Free
            r[3] = r[9] = 0
    elif jmz_free:
        # Fixed-Free
        r[3] += r[9]
        r[9] = 0

    if imx_free:
        if not jmx_free:
            # Free-Fixed
            r[1] -= 1.5 * r[5] / l
            r[7] += 1.5 * r[5] / l
            r[11] -= 0.5 * r[5]
            r[5] = 0
        else:
            # Free-Free
            r[1] -= (r[5] + r[11]) / l
            r[7] += (r[5] + r[11]) / l
            r[5] = r[11] = 0
    elif jmx_free:
        # Fixed-Free
        r[1] -= 1.5 * r[11] / l
        r[5] -= 0.5 * r[11]
        r[7] += 1.5 * r[11] / l
        r[11] = 0

    if imy_free:
        if not jmy_free:
            # Free-Fixed
            r[2] += 1.5 * r[4] / l
            r[8] -= 1.5 * r[4] / l
            r[10] -= 0.5 * r[4] / l
            r[4] = 0
        else:
            # Free-Free
            r[2] += (r[4] + r[10]) / l
            r[8] -= (r[4] + r[10]) / l
            r[4] = r[10] = 0
    elif jmy_free:
        # Fixed-Free
        r[2] += 1.5 * r[10] / l
        r[4] -= 0.5 * r[10]
        r[8] -= 1.5 * r[10] / l
        r[10] = 0

    return r


def clear_element_load_cache():
    """Clears the element load function cache."""
    local_reactions.cache_clear()


class ElementLoad(np.ndarray):
    """
    A class representing an element load.

    Parameters
    ----------
    element : str
        The name of the element to which the loads are applied.
    fx, fy, fz : float
        The global forces applied to the element.
    mx, my, mz : float
        The global moments applied to the element.
    ix, : float
        The distance from the i node at where the loads are applied.
    dx : float
        The distance from the ix position toward the j node over which
        the loads are applied.
    """
    def __new__(cls, element, fx=0, fy=0, fz=0, mx=0, my=0, mz=0, ix=0, dx=-1):
        obj = np.array([fx, fy, fz, mx, my, mz], dtype='float').view(cls)
        obj.element = element
        obj.ix = ix
        obj.dx = dx
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.element = getattr(obj, 'element', '')
        self.ix = getattr(obj, 'ix', 0)
        self.dx = getattr(obj, 'dx', 0)
        self.element_ref = None

    def element():
        def fget(self):
            return self._element
        def fset(self, value):
            if not isinstance(value, str):
                value = str(value)
            self._element = value
        def fdel(self):
            del self._element
        return locals()
    element = property(**element())

    def element_ref():
        def fget(self):
            value = self._element_ref
            if value is None:
                return value
            return value()
        def fset(self, value):
            if value is not None:
                value = weakref.ref(value)
            self._element_ref = value
        def fdel(self):
            del self._element_ref
        return locals()
    element_ref = property(**element_ref())

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

    def __repr__(self):
        s = [
            'element={!r}'.format(self.element),
            'forces={!r}'.format((self.fx, self.fy, self.fz)),
            'moments={!r}'.format((self.mx, self.my, self.mz)),
            'ix={!r}'.format(self.ix),
            'dx={!r}'.format(self.dx),
        ]

        return '{}({})'.format(type(self).__name__, ', '.join(s))

    def forces(self):
        """Returns the force vector."""
        return self[:3]

    def moments(self):
        """Returns the moment vector."""
        return self[3:6]

    def get_element(self):
        """Gets the referenced element."""
        if self._element_ref is None:
            raise ValueError('Element has not been set.')
        return self._element_ref

    def set_element(self, edict):
        """
        Sets the element reference.

        Parameters
        ----------
        edict : dict
            A dictionary mapping node names to node objects.
        """
        self._element_ref = edict[self.element]

    def local_reactions(self, di=(0, 0, 0), dj=(0, 0, 0)):
        """
        Returns the local end reactions for the element.

        Parameters
        ----------
        di, dj : array
            The deflections at the i and j ends of the element.
        """
        di, dj = np.asarray(di), np.asarray(dj)
        e = self.get_element()
        xi, xj = e.get_nodes()

        dx, dy, dz = (xj - xi) + (dj - di)
        fx, fy, fz = self.forces()
        mx, my, mz = self.moments()

        r = local_reactions(
            fx, fy, fz, mx, my, mz, dx, dy, dz,
            e.roll, self.ix, self.dx,
            e.imx_free, e.imy_free, e.imz_free,
            e.jmx_free, e.jmy_free, e.jmz_free
        )

        return r

    def global_reactions(self, di=(0, 0, 0), dj=(0, 0, 0)):
        """
        Returns the global end reactions for the element.

        Parameters
        ----------
        di, dj : array
            The deflections at the i and j ends of the element.
        """
        di, dj = np.asarray(di), np.asarray(dj)
        e = self.get_element()
        t = e.transformation_matrix(di, dj)
        q = self.local_reactions(di, dj)
        return t.T.dot(q)
