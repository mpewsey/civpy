"""
Copyright (c) 2019, Matt Pewsey
"""

import attr
import numpy as np
import matplotlib.pyplot as plt
from .spatial_hash import SpatialHash

__all__ = ['Alignment']


@attr.s(hash=False)
class Alignment(object):
    """
    A class representing a survey alignment.

    Parameters
    ----------
    name : str
        Name of alignment.
    pis : list
        A list of :class:`.PI`.
    stakes : list
        A list of :class:`.SurveyStake`.
    grid : float
        The grid size used for spatial hash generation.
    view_offset : float
        The offset beyond which points will be ignored when generating station
        coordinates from global coordinates.
    view_margin : float
        The station margin at the beginning and end of the alignment. Beyond
        this threshold, generated station coordinates from global coordinates
        will be ignored.

    Examples
    --------
    .. plot:: ../examples/survey/alignment_ex1.py
        :include-source:
    """
    # Global class variables
    BISC_TOL = 1e-4 # Bisector station tolerance

    # Properties
    name = attr.ib()
    pis = attr.ib(default=[])
    stakes = attr.ib(default=[])
    grid = attr.ib(default=10)
    view_offset = attr.ib(default=15)
    view_margin = attr.ib(default=15)

    def set_stake_xy(self):
        """
        Sets the xy coordinates for all station stakes assigned to the
        alignment.
        """
        obj = []
        p = []

        for x in self.stakes:
            if x._type == 'station':
                obj.append(x)
                p.append((x.station, x.offset, x.rotation))

        p = np.array(p)
        c, s = np.cos(p[:,2]), np.sin(p[:,2])
        c, s = np.column_stack([c, -s]), np.column_stack([s, c])

        b = self.coordinates(p[:,0])
        p = self.coordinates(p[:,:2])
        p -= b

        c = np.einsum('ij,ij->i', p, c)
        s = np.einsum('ij,ij->i', p, s)
        p = np.column_stack([c, s])
        p += b

        for a, b in zip(obj, p):
            a[:2] = b

    def pi_coordinates(self):
        """
        Returns an array of PI coordinates of shape (N, 3).
        """
        if not self.pis:
            return np.zeros((0, 3), dtype='float')
        return np.array(self.pis, dtype='float')

    def pi_radii(self):
        """
        Returns an array of PI horizontal curve radii of shape (N,).
        """
        return np.array([x.radius for x in self.pis], dtype='float')

    def azimuths(self):
        """
        Returns an array of alignment azimuths in the shape (N,). Each element
        of the array corresponds to a PI index and represents the azimuth of
        the alignment ahead of that PI.
        """
        if not self.pis:
            return np.zeros(0, dtype='float')

        elif len(self.pis) == 1:
            return np.zeros(1, dtype='float')

        x = self.pi_coordinates()
        dx = x[1:,:2] - x[:-1,:2]
        az = np.arctan2(dx[:,0], dx[:,1])
        az = np.append(az, az[-1])

        return np.asarray(az, dtype='float')

    def deflection_angles(self):
        """
        Returns an array of PI deflection angles in the shape (N,). The angle
        is negative for turns to the left and positive for turns to the right.
        """
        if not self.pis:
            return np.zeros(0, dtype='float')

        elif len(self.pis) == 1:
            return np.zeros(1, dtype='float')

        az = self.azimuths()
        da = az[1:] - az[:-1]
        i = (np.abs(da) > np.pi)
        da[i] -= 2 * np.pi * np.sign(da[i])
        da = np.insert(da, 0, 0)

        return np.asarray(da, dtype='float')

    def tangent_ordinates(self):
        """
        Returns an array of tangent ordinates corresponding to each PI
        in the shape (N,). This value is the horizontal distance between
        the PI and PC and PI and PT.
        """
        r = self.pi_radii()
        da = self.deflection_angles()
        return r * np.abs(np.tan(da/2))

    def curve_lengths(self):
        """
        Returns an array of horizontal curve lengths corresponding to each PI
        in teh shape (N,). This value is the station distance between the
        PC and PT.
        """
        r = self.pi_radii()
        da = self.deflection_angles()
        return r * np.abs(da)

    def middle_ordinates(self):
        """
        Returns an array of middle ordinate distances corresponding to each PI
        in the shape (N,). This value is the horizontal distance between the
        MPC and midpoint of the chord line between the PC and PT.
        """
        r = self.pi_radii()
        da = np.abs(self.deflection_angles())
        return r * (1 - np.cos(da/2))

    def external_ordinates(self):
        """
        Returns an array of external ordinates corresponding to each PI
        in the shape (N,). This is the horizontal distance between the
        MPC and PI.
        """
        r = self.pi_radii()
        da = self.deflection_angles()
        return r * np.abs(np.tan(da/2) * np.tan(da/4))

    def chord_distances(self):
        """
        Returns an array of chord distances corresponding to each PI
        in teh shape (N,). This is the straight line horizontal distance
        between the PC and PT.
        """
        r = self.pi_radii()
        da = np.abs(self.deflection_angles())
        return 2 * r * np.sin(da/2)

    def pt_coordinates(self):
        """
        Returns an array of (x, y) coordinates for the Point of Tangents (PT)
        in the shape (N, 2).
        """
        if not self.pis:
            return np.zeros((0, 3), dtype='float')

        pi = self.pi_coordinates()
        az = self.azimuths()
        t = self.tangent_ordinates()
        t = np.expand_dims(t, 1)
        uv = np.column_stack([np.sin(az), np.cos(az)])
        pt = pi[:,:2] + t * uv

        return np.asarray(pt, dtype='float')

    def pc_coordinates(self):
        """
        Returns an array of (x, y) coordinates for the Point of Curves (PC)
        in the shape (N, 2).
        """
        if not self.pis:
            return np.zeros((0, 3), dtype='float')

        pi = self.pi_coordinates()
        az = self.azimuths()
        da = self.deflection_angles()
        t = self.tangent_ordinates()
        t = np.expand_dims(t, 1)
        az -= da
        uv = np.column_stack([np.sin(az), np.cos(az)])
        pc = pi[:,:2] - t * uv

        return np.asarray(pc, dtype='float')

    def mpc_coordinates(self):
        """
        Returns an array of (x, y) coordinates for the Midpoint of Curves (MPC)
        in the shape (N, 2).
        """
        if not self.pis:
            return np.zeros((0, 3), dtype='float')

        pi = self.pi_coordinates()
        az = self.azimuths()
        da = self.deflection_angles()
        e = self.external_ordinates()
        az += (np.pi - da) / 2
        da = np.expand_dims(da, 1)
        e = np.expand_dims(e, 1)
        uv = np.column_stack([np.sin(az), np.cos(az)])
        mpc = pi[:,:2] + np.sign(da) * e * uv

        return np.asarray(mpc, dtype='float')

    def rp_coordinates(self):
        """
        Returns an array of (x, y) coordinates for the Radius Points (RP)
        in the shape (N, 2).
        """
        if not self.pis:
            return np.zeros((0, 3), dtype='float')

        pi = self.pi_coordinates()
        az = self.azimuths()
        da = self.deflection_angles()
        e = self.external_ordinates()
        e = np.expand_dims(e, 1)
        r = self.pi_radii()
        r = np.expand_dims(r, 1)
        az += (np.pi - da) / 2
        uv = np.column_stack([np.sin(az), np.cos(az)])
        da = np.expand_dims(da, 1)
        rp = pi[:,:2] + np.sign(da) * (e + r) * uv

        return np.asarray(rp, dtype='float')

    def pt_stations(self):
        """
        Returns an array of (x, y) coordinates for the Point of Tangents (PT)
        in the shape (N, 2).
        """
        if not self.pis:
            return np.zeros(0, dtype='float')

        x = self.pi_coordinates()
        tan = self.tangent_ordinates()
        dist = np.linalg.norm(x[:-1,:2] - x[1:,:2], axis=1)
        dist = np.insert(dist, 0, 0)
        dist += self.curve_lengths() - tan
        sta = np.cumsum(dist)
        sta[1:] -= np.cumsum(tan[:-1])

        return np.asarray(sta, dtype='float')

    def pc_stations(self):
        """
        Returns an array of stations for the Point of Curves (PC) in the
        shape (N,).
        """
        if not self.pis:
            return np.zeros(0, dtype='float')

        sta = self.pt_stations() - self.curve_lengths()
        return np.asarray(sta, dtype='float')

    def mpc_stations(self):
        """
        Returns an array of stations for the Midpoint of Curves (MPC)
        in the shape (N,).
        """
        return 0.5 * (self.pt_stations() + self.pc_stations())

    def poc_transforms(self):
        """
        Returns the POC transforms in the shape (N, 2, 2). These transforms
        project (x, y) global coordinates to (offset, station) station
        coordinates relative to the PI angle bisector.
        """
        az = self.azimuths()
        da = self.deflection_angles()
        l = az - da / 2
        t = l + np.pi / 2
        t = np.column_stack([np.sin(t), np.cos(t), np.sin(l), np.cos(l)])

        return t.reshape(t.shape[0], 2, 2)

    def pot_transforms(self):
        """
        Returns the POT transforms in the shape (N, 2, 2). These transforms
        project (x, y) global coordinates to (offset, station) station
        coordinates relative to the tangent line between PI's.
        """
        l = self.azimuths()
        t = l + np.pi / 2
        t = np.column_stack([np.sin(t), np.cos(t), np.sin(l), np.cos(l)])
        return t.reshape(t.shape[0], 2, 2)

    def segment_indices(self, stations):
        """
        Determines the segment type and PI indices corresponding to the
        specified stations. Returns an array of shape (N, 2). The first column
        of the array contains 1 if the station is located along an alignment
        tangent or 2 if the station is located on a horizontal curve or
        alignment bisector. The second column contains the index corresponding
        to the PI where the point is located.

        Parameters
        ----------
        stations : array
            An array of stations of shape (N,).
        """
        sta = np.asarray(stations)
        pc_sta = self.pc_stations()
        pt_sta = self.pt_stations()
        s = SpatialHash(np.expand_dims(sta, 1), self.grid)

        # Set values beyond alignment limits
        r = np.zeros((sta.shape[0], 2), dtype='int')
        r[sta < 0] = 1, 0
        r[sta > pt_sta[-1]] = 1, pt_sta.shape[0]-1

        # POT segments
        ah = np.expand_dims(pc_sta[1:], 1)
        bk = np.expand_dims(pt_sta[:-1], 1)

        for i, (a, b) in enumerate(zip(ah, bk)):
            f = s.query_range(b, a, 0)
            r[f] = 1, i

        # POC segments
        f = (self.curve_lengths() == 0)
        pc_sta[f] -= Alignment.BISC_TOL
        pt_sta[f] += Alignment.BISC_TOL

        ah = np.expand_dims(pt_sta[1:-1], 1)
        bk = np.expand_dims(pc_sta[1:-1], 1)

        for i, (a, b) in enumerate(zip(ah, bk)):
            f = s.query_range(b, a, 0)
            r[f] = 2, i+1

        return r

    def _pot_coordinates(self, result, seg, sta_coords):
        """
        Assigns the POT coordinates for :meth:`.coordinates`.

        Parameters
        ----------
        result : array
            The array to which the results will be added.
        seg : array
            The segment indices array.
        sta_coords : array
            An array of station coordinates of shape (N, 2).
        """
        f = (seg[:,0] == 1)

        if not f.any():
            return

        sta = np.expand_dims(sta_coords[f,0], 1)
        off = np.expand_dims(sta_coords[f,1], 1)

        i = seg[f,1]
        t = self.pot_transforms()[i]
        tx, ty = t[:,0], t[:,1]
        pt_coord = self.pt_coordinates()[i]
        pt_sta = np.expand_dims(self.pt_stations()[i], 1)

        result[f] = tx * off + ty * (sta - pt_sta) + pt_coord

    def _poc_bisc_coordinates(self, result, seg, sta_coords):
        """
        Assigns the POC bisector coordinates for :meth:`.coordinates`.

        Parameters
        ----------
        result : array
            The array to which the results will be added.
        seg : array
            The segment indices array.
        sta_coords : array
            An array of station coordinates of shape (N, 2).
        """
        f = (seg[:,0] == 2) & (self.curve_lengths() == 0)[seg[:,1]]

        if not f.any():
            return

        off = np.expand_dims(sta_coords[f,1], 1)

        i = seg[f,1]
        tx = self.poc_transforms()[i,0]
        rp_coord = self.rp_coordinates()[i]

        result[f] = tx * off + rp_coord

    def _poc_curve_coordinates(self, result, seg, sta_coords):
        """
        Assigns the POC curve coordinates for :meth:`.coordinates`.

        Parameters
        ----------
        result : array
            The array to which the results will be added.
        seg : array
            The segment indices array.
        sta_coords : array
            An array of station coordinates of shape (N, 2).
        """
        l = self.curve_lengths()
        f = (seg[:,0] == 2) & (l != 0)[seg[:,1]]

        if not f.any():
            return

        sta = sta_coords[f,0]
        off = sta_coords[f,1]

        i = seg[f,1]
        tx = self.poc_transforms()[i,0]
        mpc_sta = self.mpc_stations()[i]
        rp_coord = self.rp_coordinates()[i]
        da = self.deflection_angles()[i]
        r = np.expand_dims(self.pi_radii()[i], 1)

        beta = da * (mpc_sta - sta) / l[i]
        c, s = np.cos(beta), np.sin(beta)
        c, s = np.column_stack([c, -s]), np.column_stack([s, c])

        c = np.einsum('ij,ij->i', tx, c)
        s = np.einsum('ij,ij->i', tx, s)

        tx = np.column_stack([c, s])
        da = np.sign(np.expand_dims(da, 1))
        off = np.expand_dims(off, 1)

        result[f] = tx * (off - da * r) + rp_coord

    def coordinates(self, sta_coords):
        """
        Returns the (x, y) or (x, y, z) global coordinates corresponding
        to the input station coordinates. Result is in the shape of (N, 2)
        of (N, 3).

        Parameters
        ----------
        sta_coords : array
            An array of (station), (station, offset), or (station, offset, z)
            coordinates of the shape (N,), (N, 2) or (N, 3).
        """
        sta_coords = np.asarray(sta_coords)

        # If shape is (N,), add zero offsets
        if len(sta_coords.shape) == 1:
            sta_coords = np.column_stack([sta_coords, np.zeros(sta_coords.shape[0])])

        result = np.zeros((sta_coords.shape[0], 2), dtype='float')
        seg = self.segment_indices(sta_coords[:,0])

        self._pot_coordinates(result, seg, sta_coords)
        self._poc_bisc_coordinates(result, seg, sta_coords)
        self._poc_curve_coordinates(result, seg, sta_coords)

        # Add z coordinate to result if available
        if sta_coords.shape[1] == 3:
            result = np.column_stack([result, sta_coords[:,2]])

        return np.asarray(result, dtype='float')

    def _pot_station_coordinates(self, result, spatial_hash, coords):
        """
        Adds the POT station coordinates within the view.

        Parameters
        ----------
        result : dict
            The dictionary to which the results will be added.
        spatial_hash : array
            The spatial hash.
        coords : array
            An array of coordinates of shape (N, 2) or (N, 3).
        """
        t = self.pot_transforms()
        pt_sta = self.pt_stations()
        pt_coord = self.pt_coordinates()

        bk = self.pt_coordinates()[:-1]
        ah = self.pc_coordinates()[1:]

        if t.shape[0] > 0:
            bk[0] -= self.view_margin * t[0, 1]
            ah[-1] += self.view_margin * t[-1, 1]

        for i, (a, b) in enumerate(zip(ah, bk)):
            f = spatial_hash.query_range(b, a, self.view_offset)

            if f.shape[0] == 0:
                continue

            delta = coords[f,:2] - pt_coord[i]
            sta = np.dot(delta, t[i,1]) + pt_sta[i]
            off = np.dot(delta, t[i,0])

            if coords.shape[1] == 3:
                p = np.column_stack([sta, off, coords[f,2]])
            else:
                p = np.column_stack([sta, off])

            for n, m in enumerate(f):
                if m not in result:
                    result[m] = []
                result[m].append(p[n])

    def _poc_station_coordinates(self, result, spatial_hash, coords):
        """
        Adds the POC station coordinates within the view.

        Parameters
        ----------
        result : dict
            The dictionary to which the results will be added.
        spatial_hash : array
            The spatial hash.
        coords : array
            An array of coordinates of shape (N, 2) or (N, 3).
        """
        l = self.curve_lengths()
        t = self.poc_transforms()
        da = self.deflection_angles()
        pc_sta = self.pc_stations()
        pt_sta = self.pt_stations()
        rp_coord = self.rp_coordinates()
        pt_coord = self.pt_coordinates()

        for i in range(1, len(self.pis)-1):
            r = self.pis[i].radius
            ro = r + self.view_offset
            ri = max(r - self.view_offset, 0)
            f = spatial_hash.query_point(rp_coord[i], ro, ri)

            if f.shape[0] == 0:
                continue

            if l[i] == 0:
                # Angle bisector
                delta = coords[f,:2] - pt_coord[i]
                sta = np.dot(delta, t[i,1]) + pt_sta[i]
                off = np.dot(delta, t[i,0])

                g = ((np.abs(off) <= self.view_offset)
                     & (sta >= pt_sta[i] - Alignment.BISC_TOL)
                     & (sta <= pt_sta[i] + Alignment.BISC_TOL))
            else:
                # Horizontal curve
                delta = pt_coord[i] - rp_coord[i]
                delta = np.arctan2(delta[0], delta[1])
                p = coords[f,:2] - rp_coord[i]
                delta -= np.arctan2(p[:,0], p[:,1])

                sta = pt_sta[i] - (l[i] / da[i]) * delta
                off = np.sign(da[i]) * (r - np.linalg.norm(p, axis=1))

                g = (sta >= pc_sta[i]) & (sta <= pt_sta[i])

            if coords.shape[1] == 3:
                p = np.column_stack([sta, off, coords[f,2]])[g]
            else:
                p = np.column_stack([sta, off])[g]

            for n, m in enumerate(f[g]):
                if m not in result:
                    result[m] = []
                result[m].append(p[n])

    def station_coordinates(self, coordinates):
        """
        Finds the (station, offset) or (station, offset, z) coordinates
        for the input global coordinates. Returns a dictionary of point
        indices with arrays of shape (N, 2) or (N, 3). If a point index
        is not in the dictionary, then no points are located along
        the alignment within the view threshold.

        Parameters
        ----------
        coordinates : array
            An array of (x, y) or (x, y, z) global coordinates in the shape
            (N, 2) or (N, 3).
        """
        coordinates = np.asarray(coordinates)
        s = SpatialHash(coordinates[:,:2], self.grid)
        result = {}

        self._pot_station_coordinates(result, s, coordinates)
        self._poc_station_coordinates(result, s, coordinates)

        for k, x in result.items():
            result[k] = np.array(x, dtype='float')

        return result

    def plot_plan(self, ax=None, step=1, symbols={}):
        """
        Plots a the plan view for the alignment.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            The axex to which to add the plot. If None, a new figure and axes
            will be created.
        step : float
            The step interval to use for plotting points along horizontal
            curves.
        symbols : dict
            A dictionary of symbols to use for the plot. The following keys
            are used:

                * `pi`: PI point symbol, default is 'r.'
                * `rp`: RP point symbol, default is 'c.'
                * `pc`: PC point symbol, default is 'b.'
                * `pt`: PT point symbol, default is 'b.'
                * `alignment`: Alignment lines, default is 'b-'
                * `stakes`: Stake symbols, default is 'rx'

        Examples
        --------
        .. plot:: ../examples/survey/alignment_ex1.py
            :include-source:
        """
        if ax is None:
            x = self.pi_coordinates()[:,:2]
            mx = x.max(axis=0)
            c = 0.5 * (mx + x.min(axis=0))
            r = 1.1 * (np.max(mx - c) + self.view_offset + self.view_margin)
            xlim, ylim = np.column_stack([c - r, c + r])

            fig = plt.figure()
            ax = fig.add_subplot(111,
                title=self.name,
                xlim=xlim,
                ylim=ylim,
                xlabel='X',
                ylabel='Y',
                aspect='equal'
            )
            ax.grid('major', alpha=0.2)

        sym = dict(
            pi='r.',
            rp='c.',
            pc='b.',
            pt='b.',
            alignment='b-',
            stakes='rx'
        )
        sym.update(symbols)

        pt = self.pt_coordinates()
        pc = self.pc_coordinates()

        if sym['alignment'] is not None:
            for a, b in zip(pt[:-1], pc[1:]):
                x = np.array([a, b])
                ax.plot(x[:,0], x[:,1], sym['alignment'])

            for a, b in zip(self.pt_stations(), self.pc_stations()):
                if a != b:
                    n = int(np.ceil((a - b) / step))
                    sta = np.linspace(b, a, n)
                    x = self.coordinates(sta)
                    ax.plot(x[:,0], x[:,1], sym['alignment'])

        if sym['pi'] is not None:
            x = self.pi_coordinates()
            ax.plot(x[:,0], x[:,1], sym['pi'])

        if sym['rp'] is not None:
            x = self.rp_coordinates()
            ax.plot(x[:,0], x[:,1], sym['rp'])

        if sym['pt'] is not None:
            ax.plot(pt[:,0], pt[:,1], sym['pt'])

        if sym['pc'] is not None:
            ax.plot(pc[:,0], pc[:,1], sym['pc'])

        if sym['stakes'] is not None and len(self.stakes) > 0:
            self.set_stake_xy()
            x = np.array(self.stakes)
            ax.plot(x[:,0], x[:,1], sym['stakes'])

        return ax
