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

        hashes = self._multi_hash(points)
        hash_dict = self._dict

        for i, h in enumerate(hashes):
            if h not in hash_dict:
                hash_dict[h] = []
            hash_dict[h].append(i)

    def _multi_hash(self, points):
        """
        Returns a list of dictionary hash keys corresponding to the input
        points.

        Parameters
        ----------
        points : list
            A list of points of shape (N, D).
        """
        points = np.asarray(points)
        points = (points // self._grid).astype('int')
        return [hash(tuple(x)) for x in points]

    def _hash(self, point):
        """
        Returns the hash key corresponding to the input point.

        Parameters
        ----------
        point : list
            A list of shape (D,).
        """
        point = np.asarray(point)
        point = (point // self._grid).astype('int')
        return hash(tuple(point))

    def get(self, point):
        """
        Returns the point indices correesponding to the same hash as the input
        point.

        Parameters
        ----------
        point : list
            A list of shape (D,).
        """
        h = self._hash(point)
        return self._dict.get(h, [])

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
        diag = self._grid * self._dim**0.5
        hi = max(ri - diag, 0)
        ho = ro + diag
        result = [[]]

        x = np.column_stack([point - ro, point + ro])
        x = (x // self._grid).astype('int')
        x = [np.arange(a, b+1) for a, b in x]
        x = np.array(np.meshgrid(*x), dtype='float').T.reshape(-1, self._dim)
        x *= self._grid

        # Filter hashes
        dist = np.linalg.norm(point - x, axis=1)
        x = x[(dist <= ho) & (dist >= hi)]

        for p in x:
            p = self.get(p)
            result.append(p)

        # Evaluate points
        result = np.unique(np.concatenate(result)).astype('int')
        x = self.points[result]
        dist = np.linalg.norm(x - point, axis=1)
        f = (dist <= ro) & (dist >= ri)
        result = result[f][dist[f].argsort()]

        return result

    def query_range(self, start, stop, ro, ri=0):
        """
        Returns an array of point indices for all points along the specified
        range within the inner and outer offsets.

        Parameters
        ----------
        start : list
            The starting point for the range. The point should be of shape (D,).
        stop : list
            The ending point for the range. The point should be of shape (D,).
        ro : float
            The outer offset beyond which points will be excluded.
        ri : float
            The inner offset before which points will be excluded.
        """
        start = np.asarray(start)
        stop = np.asarray(stop)
        self._check_shape(start)
        self._check_shape(stop)
        unit = stop - start
        length = np.linalg.norm(unit)

        if length == 0:
            return self.query_point(start, ro, ri)

        unit = unit / length
        diag = self._grid * self._dim**0.5
        hi = max(ri - diag, 0)
        ho = ro + diag
        mi = -diag
        mo = length + diag
        result = [[]]

        a = np.column_stack([start - ro, stop - ro]).min(axis=1)
        b = np.column_stack([start + ro, stop + ro]).max(axis=1)

        x = np.column_stack([a, b])
        x = (x // self._grid).astype('int')
        x = [np.arange(a, b+1) for a, b in x]
        x = np.array(np.meshgrid(*x), dtype='float').T.reshape(-1, self._dim)
        x *= self._grid

        # Filter hashes
        s = x - start
        proj = np.dot(s, unit)
        off = np.linalg.norm(s - np.expand_dims(proj, 1)*unit, axis=1)
        x = x[(proj >= mi) & (proj <= mo) & (off <= ho) & (off >= hi)]

        for p in x:
            p = self.get(p)
            result.append(p)

        # Evaluate points
        result = np.unique(np.concatenate(result)).astype('int')
        x = self.points[result]
        s = x - start
        proj = np.dot(s, unit)
        off = np.linalg.norm(s - np.expand_dims(proj, 1)*unit, axis=1)
        f = (proj >= 0) & (proj <= length) & (off <= ro) & (off >= ri)
        result = result[f][off[f].argsort()]

        return result


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
