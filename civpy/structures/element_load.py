import propy
import numpy as np


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
    # Custom properties
    element = propy.str_property('element')
    fx = propy.index_property(0)
    fy = propy.index_property(1)
    fz = propy.index_property(2)
    mx = propy.index_property(3)
    my = propy.index_property(4)
    mz = propy.index_property(5)
    _element_ref = propy.weakref_property('_element_ref')

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
        self._element_ref = None

    def __repr__(self):
        s = [
            'element={!r}'.format(self.element),
            'forces={!r}'.format((self.fx, self.fy, self.fz)),
            'moments={!r}'.format((self.mx, self.my, self.mz)),
            'ix={!r}'.format(self.ix),
            'dx={!r}'.format(self.dx)
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
        t = e.transformation_matrix(di, dj)
        l = e.length(di, dj)

        l1 = l * abs(self.ix) if self.ix < 0 else self.ix
        l2 = l * abs(self.dx) if self.dx < 0 else self.dx
        l2 = l - l1 - l2

        if l1 > l or l1 < 0 or l2 > l or l2 < 0:
            raise ValueError('Load applied beyond element bounds.')

        # Forces
        if self.dx == 0:
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

        fx, fy, fz = t[:3,:3].dot(self.forces())
        r = np.array([-fx*fai, -fy*fsi, -fz*fsi, -fti, -fz*fmi, -fy*fmi,
                      -fx*faj, -fy*fsj, -fz*fsj, -ftj, -fz*fmj, -fy*fmj],
                      dtype='float')

        # Moments
        if self.dx == 0:
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

        mx, my, mz = t[:3,:3].dot(self.moments())
        r += np.array([-fai, -my*fsi, -mx*fsi, -mx*fti, -my*fmi, -mz*fmi,
                       -faj, -my*fsj, -mx*fsj, -mx*ftj, -my*fmj, -mz*fmj])

        # Adjust reactions for element end fixities
        if e.imz_free:
            if not e.jmz_free:
                # Free-Fixed
                r[9] += r[3]
                r[3] = 0
            else:
                # Free-Free
                r[3] = r[9] = 0
        elif e.jmz_free:
            # Fixed-Free
            r[3] += r[9]
            r[9] = 0

        if e.imx_free:
            if not e.jmx_free:
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
        elif e.jmx_free:
            # Fixed-Free
            r[1] -= 1.5 * r[11] / l
            r[5] -= 0.5 * r[11]
            r[7] += 1.5 * r[11] / l
            r[11] = 0

        if e.imy_free:
            if not e.jmy_free:
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
        elif e.jmy_free:
            # Fixed-Free
            r[2] += 1.5 * r[10] / l
            r[4] -= 0.5 * r[10]
            r[8] -= 1.5 * r[10] / l
            r[10] = 0

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
