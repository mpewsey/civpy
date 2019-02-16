import propy
import numpy as np

__all__ = ['SurveyStake']


class SurveyStake(np.ndarray):
    """
    A class representing a survey stake. This method should be initialized
    using the :meth:`SurveyStake.init_xy` or :meth:`SurveyStake.init_station`
    class methods.
    """
    TYPES = ('xy', 'station')

    # Custom properties
    x = propy.index_property(0)
    y = propy.index_property(1)
    z = propy.index_property(2)
    lock_z = propy.bool_property('lock_z')
    _type = propy.enum_property('_type', TYPES)

    def __new__(cls, x, y, z, station, offset, height, rotation, lock_z, _type,
                _init=False, **kwargs):
        if not _init:
            raise ValueError('SurveyStake should be initialized using the '
                '`SurveyStake.init_xy` or `SurveyStake.init_station` methods '
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

    __repr__ = propy.repr_method('_type', 'x', 'y', 'z', 'station', 'offset',
                                 'lock_z', 'meta')

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
