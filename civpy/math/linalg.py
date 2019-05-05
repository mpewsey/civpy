"""
Copyright (c) 2019, Matt Pewsey
"""

import numpy as np
from math import cos, sin

__all__ = [
    'projection_angles',
    'rotation_matrix2',
    'rotation_matrix3',
    'rotate2',
    'rotate3',
]


def projection_angles(name):
    """
    Returns the rotation angles for the specified projection.

    Parameters
    ----------
    name : {'xy', 'xz', 'yz', 'yx', 'zx', 'zy'}
        The name of the projection.
    """
    if name == 'xy':
        return 0, 0, 0
    elif name == 'xz':
        return -np.pi/2, 0, 0
    elif name == 'yz':
        return -np.pi/2, 0, -np.pi/2
    elif name == 'yx':
        return 0, np.pi, np.pi/2
    elif name == 'zx':
        return np.pi/2, np.pi/2, 0
    elif name == 'zy':
        return np.pi, np.pi/2, np.pi
    else:
        raise ValueError('Invalid projection name: {!r}.'.format(name))


def rotation_matrix2(angle):
    """
    Returns the 2D rotation matrix.

    Parameters
    ----------
    angle : float
        The counter clockwise rotation angle in radians.
    """
    c, s = cos(angle), sin(angle)
    return np.array([[c, -s], [s, c]])


def rotation_matrix3(angle_x=0, angle_y=0, angle_z=0):
    """
    Returns the 3D rotation matrix.

    Parameters
    ----------
    angle : float
    """
    if angle_x != 0:
        c, s = cos(angle_x), sin(angle_x)
        r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        r = np.identity(3)

    if angle_y != 0:
        c, s = cos(angle_y), sin(angle_y)
        r = r.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))

    if angle_z != 0:
        c, s = cos(angle_z), sin(angle_z)
        r = r.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))

    return r


def rotate2(x, angle, origin=(0, 0)):
    """
    Rotates the input 2D vectors by the specified angle.

    Parameters
    ----------
    x : array
        One or multiple vectors to rotate.
    angle : float
        The counter clockwise rotation angle in radians.
    origin : array
        The point about which the rotation will be performed.
    """
    origin = np.asarray(origin)
    x = np.asarray(x) - origin
    r = rotation_matrix2(angle)
    return x.dot(r.T) + origin


def rotate3(x, angle_x=0, angle_y=0, angle_z=0, origin=(0, 0, 0)):
    """
    Rotates the input 3D vectors by the specified angles.

    Parameters
    ----------
    x : array
        One or multiple vectors to rotate.
    angle_x, angle_y, angle_z : float
        The counter clockwise rotation angles about the x, y, and z axes
        in radians.
    origin : array
        The point about which the rotation will be performed.
    """
    origin = np.asarray(origin)
    x = np.asarray(x) - origin
    r = rotation_matrix3(angle_x, angle_y, angle_z)
    return x.dot(r.T) + origin
