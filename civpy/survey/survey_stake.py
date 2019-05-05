"""
Copyright (c) 2019, Matt Pewsey
"""

import numpy as np

__all__ = ['SurveyStake']


class SurveyStake(np.ndarray):
    """
    A class representing a survey stake. This method should be initialized
    using the :meth:`SurveyStake.init_xy` or :meth:`SurveyStake.init_station`
    class methods.
    """
    XY = 'xy'
    STATION = 'station'
    TYPES = (XY, STATION)

    def __new__(cls, x, y, z, station, offset, height, rotation, lock_z, _type,
                _init=False, **kwargs):
        if not _init:
            raise ValueError('SurveyStake should be initialized using the '
                'SurveyStake.init_xy or SurveyStake.init_station methods '
                'in lieu of the standard initializer.')

        obj = np.array([x, y, z], dtype='float').view(cls)
        obj.station = station
        obj.offset = offset
        obj.height = height
        obj.rotation = rotation
        obj.lock_z = lock_z
        obj._type = _type
        obj.meta = dict(**kwargs)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.station = getattr(obj, 'station', 0)
        self.offset = getattr(obj, 'offset', 0)
        self.height = getattr(obj, 'height', 0)
        self.rotation = getattr(obj, 'rotation', 0)
        self.lock_z = getattr(obj, 'lock_z', False)
        self._type = getattr(obj, '_type', 'xy')
        self.meta = getattr(obj, 'meta', {})

    def x():
        def fget(self):
            return self[0]
        def fset(self, value):
            self[0] = value
        return locals()
    x = property(**x())

    def y():
        def fget(self):
            return self[1]
        def fset(self, value):
            self[1] = value
        return locals()
    y = property(**y())

    def z():
        def fget(self):
            return self[2]
        def fset(self, value):
            self[2] = value
        return locals()
    z = property(**z())

    def _type():
        def fget(self):
            return self.__type
        def fset(self, value):
            if value not in self.TYPES:
                raise ValueError('Type {!r} must be in {!r}.'.format(value, self.TYPES))
            self.__type = value
        return locals()
    _type = property(**_type())

    def __repr__(self):
        s = [
            ('_type', self._type),
            ('x', self.x),
            ('y', self.y),
            ('z', self.z),
            ('station', self.station),
            ('offset', self.offset),
            ('lock_z', self.lock_z),
            ('meta', self.meta),
        ]

        s = ', '.join('{}={!r}'.format(x, y) for x, y in s)
        return '{}({})'.format(type(self).__name__, s)

    @classmethod
    def init_xy(cls, x, y, z=0, height=0, rotation=0, lock_z=False, **kwargs):
        """
        Initializes a survey stake based on an (x, y) global coordinate.

        Parameters
        ----------
        x, y, z : float
            The x, y, and z coordinates.
        height : float
            The height of the point above z.
        rotation : float
            The rotation of the point about its base point.
        lock_z : float
            If False, the alignment will be snapped to the TIN (if applicable)
            during certain updates. Otherwise, the z coordinate will remain
            fixed.
        """
        return cls(
            x=x, y=y, z=z,
            station=0,
            offset=0,
            height=height,
            rotation=rotation,
            lock_z=lock_z,
            _type='xy',
            _init=True,
            **kwargs
        )

    @classmethod
    def init_station(cls, station, offset=0, z=0, height=0, rotation=0,
                     lock_z=False, **kwargs):
        """
        Initializes a survey stake based on a survey station and offset.

        Parameters
        ----------
        station : float
            The alignment survey station.
        offset : float
            The offset from the alignment.
        z : float
            The z coordinate.
        height : float
            The height of the point above z.
        rotation : float
            The rotation of the point about its base point.
        lock_z : float
            If False, the alignment will be snapped to the TIN (if applicable)
            during certain updates. Otherwise, the z coordinate will remain
            fixed.
        """
        return cls(
            x=0, y=0, z=z,
            height=height,
            rotation=rotation,
            station=station,
            offset=offset,
            lock_z=lock_z,
            _type='station',
            _init=True,
            **kwargs
        )
