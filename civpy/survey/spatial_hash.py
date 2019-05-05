from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle

__all__ = ['SpatialHash']


class SpatialHash(object):
    """
    A class representing a spatial hash structure for efficient distance
    queries.

    Parameters
    ----------
    points : list
        A list of points of shape (N, D).
    grid : float
        The width of each spatial hash grid element. For 2D spaces, this value
        represents the width and height of each square spatial hash partition.

    Examples
    --------
    The below example uses the :meth:`.query_point` and :meth:`.query_range`
    methods to search for points within the specified offset of a point and
    within the specified offset of a range, respectively. The found
    points are shown in green.

    .. plot:: ../examples/survey/spatial_hash_ex2.py
        :include-source:
    """
    def __init__(self, points, grid):
        self._grid = grid
        self._dict = {}
        self._add_points(points)

    def __repr__(self):
        return '{}({}, grid={!r})'.format(type(self).__name__, self.points.shape, self._grid)

    def _check_shape(self, point):
        """
        Checks that the input point conforms to the hash dimensionality.

        Parameters
        ----------
        point : array
            An array of shape (D,).
        """
        if point.shape[0] != self._dim:
            raise ValueError('Point is {}D but should be {}D.'
                .format(point.shape[0], self._dim))

    def _add_points(self, points):
        """
        Adds the input list of points to the spatial hash.

        Parameters
        ----------
        points : list
            A list of points of shape (N, D).
        """
        points = np.asarray(points)
        self._dim = points.shape[1]
        self.points = points

        hashes = self._multi_hash(points, norm=True)
        odict = self._dict

        for i, h in enumerate(hashes):
            if h not in odict:
                odict[h] = []
            odict[h].append(i)

    def _multi_hash(self, points, norm):
        """
        Returns a list of dictionary hash keys corresponding to the input
        points.

        Parameters
        ----------
        points : list
            A list of points of shape (N, D).
        norm : bool
            If True, normalizes the points to their grid index. Otherwise,
            assumes that the input points are grid indices.
        """
        if norm:
            points = np.asarray(points) // self._grid

        return (hash(tuple(x)) for x in points)

    def _hash(self, point, norm):
        """
        Returns the hash key corresponding to the input point.

        Parameters
        ----------
        point : list
            A list of shape (D,).
        norm : bool
            If True, normalizes the points to their grid index. Otherwise,
            assumes that the input points are grid indices.
        """
        if norm:
            point = np.asarray(point) // self._grid

        return hash(tuple(point))

    def multi_get(self, points, norm=True):
        """
        Parameters
        ----------
        points : list
            A list of points of shape (N, D).
        norm : bool
            If True, normalizes the points to their grid index. Otherwise,
            assumes that the input points are grid indices.
        """
        result = []
        odict = self._dict

        for x in self._multi_hash(points, norm):
            result.extend(odict.get(x, []))

        return np.asarray(np.unique(result), dtype='int')

    def get(self, point, norm=True):
        """
        Returns the point indices correesponding to the same hash as the input
        point.

        Parameters
        ----------
        point : list
            A list of shape (D,).
        norm : bool
            If True, normalizes the points to their grid index. Otherwise,
            assumes that the input points are grid indices.
        """
        point = self._hash(point, norm)
        return self._dict.get(point, [])

    def _query_point_hash(self, point, ro, ri):
        # Calculate worst case offsets
        diag = self._grid * self._dim**0.5
        ri = max(ri - diag, 0)
        ro = ro + diag

        # Create meshgrid of hash indices
        p = np.column_stack([point - ro, point + ro]) // self._grid
        p = [np.arange(a, b+1) for a, b in p]
        p = np.array(np.meshgrid(*p), dtype='int').T.reshape(-1, self._dim)

        # Filter hashes by distance
        dist = np.linalg.norm(point - self._grid * p, axis=1)

        if ri == 0:
            p = p[dist <= ro]
        else:
            p = p[(dist <= ro) & (dist >= ri)]

        return self.multi_get(p, norm=False)

    def query_point(self, point, ro, ri=0):
        """
        Returns an array of point indices for all points contained within
        the specified inner and outer radii from the input point.

        Parameters
        ----------
        point : list
            A list of shape (D,).
        ro : float
            The outer radius beyond which points will be excluded.
        ri : float
            The inner radius before which points will be excluded.
        """
        point = np.asarray(point)
        self._check_shape(point)

        # Get hash filtered points
        result = self._query_point_hash(point, ro, ri)
        p = self.points[result]

        # Filter points by distance
        dist = np.linalg.norm(p - point, axis=1)

        if ri == 0:
            f = (dist <= ro)
        else:
            f = (dist <= ro) & (dist >= ri)

        return result[f][dist[f].argsort()]

    def _query_range_hash(self, a, b, ro, ri, u, l):
        # Calculate worst case offsets
        diag = self._grid * self._dim**0.5
        ri = max(ri - diag, 0)
        ro = ro + diag

        # Create meshgrid of hash indices
        x = np.column_stack([a - ro, a - ro]).min(axis=1)
        y = np.column_stack([b + ro, b + ro]).max(axis=1)

        p = np.column_stack([x, y]) // self._grid
        p = [np.arange(x, y+1) for x, y in p]
        p = np.array(np.meshgrid(*p), dtype='int').T.reshape(-1, self._dim)

        # Filter hashes by projection and offset
        v = self._grid * p - b
        proj = np.dot(v, u)
        dist = np.linalg.norm(v - proj.reshape(-1, 1) * u, axis=1)
        del v

        if ri == 0:
            p = p[(proj >= -diag) & (proj <= l+diag) & (dist <= ro)]
        else:
            p = p[(proj >= -diag) & (proj <= l+diag) & (dist <= ro) & (dist >= ri)]

        return self.multi_get(p, norm=False)

    def query_range(self, a, b, ro, ri=0):
        """
        Returns an array of point indices for all points along the specified
        range within the inner and outer offsets.

        Parameters
        ----------
        a : list
            The starting point for the range. The point should be of shape (D,).
        b : list
            The ending point for the range. The point should be of shape (D,).
        ro : float
            The outer offset beyond which points will be excluded.
        ri : float
            The inner offset before which points will be excluded.
        """
        a = np.asarray(a)
        b = np.asarray(b)
        self._check_shape(a)
        self._check_shape(b)

        # Create unit vector for range
        u = a - b
        l = np.linalg.norm(u)

        if l == 0:
            return self.query_point(a, ro, ri)

        u = u / l

        # Get hash filtered points
        result = self._query_range_hash(a, b, ro, ri, u, l)
        p = self.points[result]

        # Filter points by projection and offset
        v = p - b
        proj = np.dot(v, u)
        dist = np.linalg.norm(v - proj.reshape(-1, 1) * u, axis=1)

        if ri == 0:
            f = (proj >= 0) & (proj <= l) & (dist <= ro)
        else:
            f = (proj >= 0) & (proj <= l) & (dist <= ro) & (dist >= ri)

        return result[f][dist[f].argsort()]

    def _plot_1d(self, ax, sym):
        """
        Creates a 1D plot.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes to which the plot will be added. If None, a new figure
            and axes will be created.
        sym : dict
            A dictionary of plot symbols with any of the following keys:

                * points: Point symbols, default is 'r.'
                * hash: Hash region color, default is 'b'
        """
        # Create plot
        if ax is None:
            lim = np.array([self.points.min(), self.points.max()])
            lim = self._grid * (lim // self._grid + [-1, 2])
            ticks = np.arange(lim[0], lim[1] + self._grid, self._grid)

            fig = plt.figure()
            ax = fig.add_subplot(111,
                xlim=lim,
                ylim=self._grid * np.array([-0.5, 0.5]),
                xticks=ticks,
                yticks=[0]
            )
            ax.grid('major', alpha=0.4)

        # Plot hash regions
        if sym['hash'] is not None:
            y = -0.5 * self._grid
            xs = self._grid * (self.points // self._grid)
            xs = set(map(tuple, xs))

            for x in xs:
                rect = Rectangle((x[0], y), self._grid, self._grid,
                    color=sym['hash'],
                    alpha=0.2,
                    zorder=1
                )
                ax.add_artist(rect)

        # Plot points
        if sym['points'] is not None:
            x = self.points
            ax.plot(x[:,0], np.zeros(x.shape[0]), sym['points'])

        return ax


    def _plot_2d(self, ax, sym):
        """
        Creates a 2D plot.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes to which the plot will be added. If None, a new figure
            and axes will be created.
        sym : dict
            A dictionary of plot symbols with any of the following keys:

                * points: Point symbols, default is 'r.'
                * hash: Hash region color, default is 'b'
        """
        # Create plot
        if ax is None:
            lim = np.array([self.points.min(), self.points.max()])
            lim = self._grid * (lim // self._grid + [-1, 2])
            ticks = np.arange(lim[0], lim[1] + self._grid, self._grid)

            fig = plt.figure()
            ax = fig.add_subplot(111,
                xlim=lim,
                ylim=lim,
                xticks=ticks,
                yticks=ticks,
                aspect='equal'
            )
            ax.grid('major', alpha=0.4)

        # Plot hash squares
        if sym['hash'] is not None:
            xs = self._grid * (self.points // self._grid)
            xs = set(map(tuple, xs))

            for x in xs:
                rect = Rectangle(x, self._grid, self._grid,
                    color=sym['hash'],
                    fill=False,
                    zorder=1
                )
                ax.add_artist(rect)

        # Plot points
        if sym['points'] is not None:
            x = self.points
            ax.plot(x[:,0], x[:,1], sym['points'])

        return ax

    def _plot_3d(self, ax, sym):
        """
        Creats a 3D plot.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes to which the plot will be added. If None, a new figure
            and axes will be created.
        sym : dict
            A dictionary of plot symbols with any of the following keys:

                * points: Point symbols, default is 'r.'
                * hash: Hash region color, default is 'b'
        """
        # Create plot
        if ax is None:
            lim = np.array([self.points.min(), self.points.max()])
            lim = self._grid * (lim // self._grid + [-1, 2])
            ticks = np.arange(lim[0], lim[1] + self._grid, self._grid)

            fig = plt.figure()
            ax = fig.add_subplot(111,
                projection='3d',
                xlim=lim,
                ylim=lim,
                zlim=lim,
                xticks=ticks,
                yticks=ticks,
                zticks=ticks,
                aspect='equal'
            )

        # Plot hash cubes
        if sym['hash'] is not None:
            xs = self._grid * (self.points // self._grid)
            xs = set(map(tuple, xs))

            cube = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                    [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]
            simplices = [[0, 1, 2, 3], [0, 1, 5, 4], [4, 5, 6, 7],
                         [5, 1, 2, 6], [6, 7, 3, 2], [4, 7, 3, 0]]
            cube = self._grid * np.array(cube)

            for x in xs:
                x = cube + x
                x = [[x[i] for i in s] for s in simplices]
                poly = Poly3DCollection(x, alpha=0.05)
                poly.set_facecolor(sym['hash'])
                poly.set_edgecolor(sym['hash'])
                ax.add_collection(poly)

        # Plot points
        if sym['points'] is not None:
            x = self.points
            ax.plot(x[:,0], x[:,1], x[:,2], sym['points'])

        return ax

    def plot(self, ax=None, symbols={}):
        """
        Creates a plot of the spatial hash. Cannot create plots for hashes
        greater than 3 dimensions.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes to which the plot will be added. If None, a new figure
            and axes will be created.
        symbols : dict
            A dictionary of plot symbols with any of the following keys:

                * points: Point symbols, default is 'r.'
                * hash: Hash region color, default is 'b'

        Examples
        --------
        .. plot:: ../examples/survey/spatial_hash_ex1.py
            :include-source:
        """
        # Plot symbols
        sym = dict(
            points='r.',
            hash='b'
        )
        sym.update(symbols)

        if self._dim == 1:
            return self._plot_1d(ax, sym)
        elif self._dim == 2:
            return self._plot_2d(ax, sym)
        elif self._dim == 3:
            return self._plot_3d(ax, sym)
        else:
            raise ValueError('Hash is {}D but plot only supports 1D, 2D, or 3D.'
                .format(self._dim))
