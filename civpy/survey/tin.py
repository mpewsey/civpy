from __future__ import division
import propy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from .spatial_hash import SpatialHash

__all__ = ['TIN']


class TIN(object):
    """
    A class for creating triangulated irregular networks (TIN) models for
    3D surfaces. Includes methods for performing elevation and distance queries.

    Parameters
    ----------
    name : str
        The name of the model.
    points : array
        An array of points of shape (N, 3).
    breaklines : list
        A list of arrays of points representing breakline polylines.
    max_edge : float
        The maximum edge length beyond which simplices will not be included
        in the triangulation. If None, no simplices will be removed.
    step : float
        The setup interval for generated points along breaklines.
    grid : float
        The grid spacing used for the created spatial hash.

    Examples
    --------
    The following example creates a TIN model in the shape of a pyramid
    then performs distances queries to its surface:

    .. plot:: ../examples/survey/tin_ex1.py
        :include-source:
    """
    def __init__(self, name, points=[], breaklines=[], max_edge=None,
                 step=0.1, grid=10):
        self.name = name
        self.breaklines = breaklines
        self.max_edge = max_edge
        self.step = step
        self._create_triangulation(points, grid)

    __repr__ = propy.repr_method('name')

    def _create_triangulation(self, points, grid):
        """
        Creates the Delaunay trianguation and spatial hash.

        Parameters
        ----------
        points : array
            An array of points of shape (N, 3).
        grid : float
            The spatial hash grid spacing.
        """
        points = np.asarray(points)
        b = self.breakpoints()

        if b.shape[0] > 0:
            if points.shape[0] > 0:
                points = points[points[:,2].argsort()[::-1]]
                points = np.concatenate([b, points])
            else:
                points = b

        self.points = points
        self.tri = Delaunay(points[:,:2])
        self.hash = SpatialHash(points[:,:2], grid)
        self._remove_simplices()

    def _remove_simplices(self):
        """
        Removes all simplices with any edge greater than the max edge.
        """
        if self.max_edge is not None:
            p = self.tri.points
            s = self.tri.simplices

            a, b, c = p[s[:,0]], p[s[:,1]], p[s[:,2]]

            f = ((np.linalg.norm(a - b) <= self.max_edge)
                 & (np.linalg.norm(b - c) <= self.max_edge)
                 & (np.linalg.norm(a - c) <= self.max_edge))

            self.tri.simplices = s[f]

    def breakpoints(self):
        """
        Returns an array of breakpoints for the assigned breaklines. The
        breakpoints are sorted by z coordinate from greatest to least.
        """
        points = [np.zeros((0, 3), dtype='float')]

        for line in self.breaklines:
            line = np.asarray(line)

            for i, (a, b) in enumerate(zip(line[:-1], line[1:])):
                m = a - b
                n = int(np.ceil(np.linalg.norm(m) / self.step))
                if n < 2: n = 2 if i == 0 else 1
                x = np.expand_dims(np.linspace(0, 1, n), 1)
                y = m * x + b
                points.append(y)

        points = np.concatenate(points)
        points = points[points[:,2].argsort()[::-1]]

        return points

    def find_simplices(self, points):
        """
        Finds the simplices which contain the (x, y) point. Returns the simplex
        index if a single point is input or an array of simplex indices if
        multiple points are input. If the returned simplex index is -1, then
        the (x, y) point is not contained within any simplices.

        Parameters
        ----------
        points : array
            An array of points of shape (2,), (3,), (N, 2) or (N, 3).
        """
        points = np.asarray(points)

        if len(points.shape) == 1:
            points = points[:2]
        else:
            points = points[:,:2]

        return self.tri.find_simplex(points)

    def _simplex_indices(self, indexes):
        """
        Returns the simplex indices that include the input point indices.

        Parameters
        ----------
        indexes : list
            A list of point indices for which connected simplices will be
            returned.
        """
        indexes = set(indexes)
        a = []

        for i, x in enumerate(self.tri.simplices):
            for j in x:
                if j in indexes:
                    a.append(i)

        a = np.unique(a).astype('int')
        return a

    def query_simplices(self, point, radius):
        """
        Returns the indices of all simplices that have a corner within the
        specified radius of the input point.

        Parameters
        ----------
        point : array
            A point of shape (2,) or (3,).
        radius : float
            The xy-plane radius used for the query.
        """
        point = np.asarray(point)
        i = self.hash.query_point(point[:2], radius)
        return self._simplex_indices(i)

    def normal(self, simplex):
        """
        Returns the normal vector for the specified simplex. The returned
        normal vector is also a unit vector.

        Parameters
        ----------
        simplex : int
            The index of the simplex.
        """
        p = self.points
        s = self.tri.simplices[simplex]
        a, b, c = p[s[0]], p[s[1]], p[s[2]]
        n = np.cross(b - a, c - a)
        return n / np.linalg.norm(n)

    def elevation(self, point):
        """
        Returns the elevation of the TIN surface at the input point. Returns
        NaN if the TIN surface does not exist at that point.

        Parameters
        ----------
        point : array
            A point for which the elevation will be calculated. The point
            must be of shape (2,) or (3,).
        """
        point = np.asarray(point)
        s = self.find_simplices(point)

        if s == -1:
            return np.nan

        p = self.points
        n = self.normal(s)
        s = self.tri.simplices[s]

        if n[2] != 0:
            # Plane elevation
            a = p[s[0]]
            a = np.dot(n, a)
            b = np.dot(n[:2], point[:2])
            return (a - b) / n[2]

        # Vertical plane. Use max edge elevation
        zs = []
        a, b, c = p[s[0]], p[s[1]], p[s[2]]

        for v, w in ((a, b), (a, c), (b, c)):
            dv = np.linalg.norm(point[:2] - v[:2])
            dvw = np.linalg.norm(w[:2] - v[:2])
            z = (w[2] - v[2]) * dv / dvw + v[2]
            zs.append(z)

        return max(zs)

    def barycentric_coords(self, point, simplex):
        """
        Returns the local barycentric coordinates for the input point.
        If any of the coordinates are less than 0, then the projection
        of the point onto the plane of the triangle is outside of the triangle.

        Parameters
        ----------
        point : array
            A point for which barycentric coordinates will be calculated.
            The point must be of shape (3,).
        simplex : int
            The index of the simplex.
        """
        point = np.asarray(point)

        p = self.points
        n = self.normal(simplex)
        s = self.tri.simplices[simplex]

        a, b, c = p[s[0]], p[s[1]], p[s[2]]

        u = b - a
        u = u / np.linalg.norm(u)
        v = np.cross(u, n)

        x1, y1 = np.dot(a, u), np.dot(a, v)
        x2, y2 = np.dot(b, u), np.dot(b, v)
        x3, y3 = np.dot(c, u), np.dot(c, v)

        det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        x1, y1, x2, y2 = y2 - y3, x3 - x2, y3 - y1, x1 - x3

        x, y = np.dot(point, u), np.dot(point, v)
        l1 = (x1*(x - x3) + y1*(y - y3)) / det
        l2 = (x2*(x - x3) + y2*(y - y3)) / det
        l3 = 1 - l1 - l2

        return np.array([l1, l2, l3])

    def query_distances(self, point, radius):
        """
        Finds the closest distances to all simplices within the specified
        xy-plane radius.

        Parameters
        ----------
        point : array
            An array of shape (3,).
        radius : float
            The radius within the xy-plane in which simplices will be queried.

        Returns
        -------
        distances : array
            An array of distances to simplices of shape (N,).
        tin_points : array
            An array of closest simplex points of shape (N, 3).
        """
        point = np.asarray(point)
        simplices = self.query_simplices(point, radius)

        p = self.points
        s = self.tri.simplices[simplices]
        bary = [self.barycentric_coords(point, x) for x in simplices]
        bary = np.min(bary, axis=1)

        tin = np.zeros((simplices.shape[0], 3), dtype='float')
        dist = np.zeros(simplices.shape[0], dtype='float')

        for i, x in enumerate(bary):
            if x >= 0:
                # Plane distance
                n = self.normal(simplices[i])
                d = np.dot(p[s[i, 0]] - point, n)
                tin[i] = d * n + point
                dist[i] = abs(d)
                continue

            # Edge distance
            a, b, c = p[s[i]]
            dist[i] = float('inf')

            for v, w in ((a, b), (b, c), (a, c)):
                u = w - v
                m = np.linalg.norm(u)
                u = u / m
                proj = np.dot(point - v, u)

                if proj <= 0:
                    r = v
                elif proj >= m:
                    r = w
                else:
                    r = proj * u + v

                d = np.linalg.norm(r - point)

                if d < dist[i]:
                    tin[i], dist[i] = r, d

        f = dist.argsort()
        return dist[f], tin[f]

    def plot_surface_3d(self, ax=None, cmap='terrain'):
        """
        Plots a the rendered TIN surface in 3D

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes to which the plot will be added. If None, a new figure
            and axes will be created.
        cmap : str
            The name of the color map to use.

        Examples
        --------
        .. plot:: ../examples/survey/tin_ex1.py
            :include-source:
        """
        if ax is None:
            mx = self.points.max(axis=0)
            c = 0.5 * (mx + self.points.min(axis=0))
            r = 1.1 * np.max(mx - c)
            xlim, ylim, zlim = np.column_stack([c - r, c + r])

            fig = plt.figure()
            ax = fig.add_subplot(111,
                title=self.name,
                projection='3d',
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                aspect='equal'
            )

        x = self.points
        ax.plot_trisurf(x[:,0], x[:,1], x[:,2],
            triangles=self.tri.simplices,
            cmap=cmap
        )

        return ax

    def plot_surface_2d(self, ax=None):
        """
        Plots a the triangulation in 2D.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes to which the plot will be added. If None, a new figure
            and axes will be created.

        Examples
        --------
        .. plot:: ../examples/survey/tin_ex3.py
            :include-source:
        """
        if ax is None:
            mx = self.points[:,:2].max(axis=0)
            c = 0.5 * (mx + self.points[:,:2].min(axis=0))
            r = 1.1 * np.max(mx - c)
            xlim, ylim = np.column_stack([c - r, c + r])

            fig = plt.figure()
            ax = fig.add_subplot(111,
                title=self.name,
                xlim=xlim,
                ylim=ylim,
                aspect='equal'
            )

        x = self.points
        ax.triplot(x[:,0], x[:,1], triangles=self.tri.simplices)

        return ax

    def plot_contour_2d(self, ax=None, cmap='terrain'):
        """
        Plots a the rendered TIN surface in 3D

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axes to which the plot will be added. If None, a new figure
            and axes will be created.
        cmap : str
            The name of the color map to use.

        Examples
        --------
        .. plot:: ../examples/survey/tin_ex2.py
            :include-source:
        """
        if ax is None:
            mx = self.points[:,:2].max(axis=0)
            c = 0.5 * (mx + self.points[:,:2].min(axis=0))
            r = 1.1 * np.max(mx - c)
            xlim, ylim = np.column_stack([c - r, c + r])

            fig = plt.figure()
            ax = fig.add_subplot(111,
                title=self.name,
                xlim=xlim,
                ylim=ylim,
                aspect='equal'
            )

        x = self.points
        contourf = ax.tricontourf(x[:,0], x[:,1], x[:,2], cmap=cmap)
        contour = ax.tricontour(x[:,0], x[:,1], x[:,2], colors='black')

        ax.clabel(contour, inline=True, fontsize=6)
        fig.colorbar(contourf)

        return ax
