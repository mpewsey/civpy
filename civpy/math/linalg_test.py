import pytest
import numpy as np
from .linalg import *


def test_projection_angles():
    i = np.identity(3)

    a = np.abs(rotate3(i, *projection_angles('xy')).ravel())
    b = i.ravel()
    assert pytest.approx(a) == b

    a = np.abs(rotate3(i, *projection_angles('xz')).ravel())
    b = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).ravel()
    assert pytest.approx(a) == b

    a = np.abs(rotate3(i, *projection_angles('yz')).ravel())
    b = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).ravel()
    assert pytest.approx(a) == b

    a = np.abs(rotate3(i, *projection_angles('yx')).ravel())
    b = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]).ravel()
    assert pytest.approx(a) == b

    a = np.abs(rotate3(i, *projection_angles('zx')).ravel())
    b = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).ravel()
    assert pytest.approx(a) == b

    a = np.abs(rotate3(i, *projection_angles('zy')).ravel())
    b = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).ravel()
    assert pytest.approx(a) == b

    with pytest.raises(ValueError):
        projection_angles('xyz')


def test_rotate2():
    v = np.array([[1, 0], [0, 1]])
    a = rotate2(v, np.pi/2).ravel()
    b = np.array([[0, 1], [-1, 0]]).ravel()
    assert pytest.approx(a) == b

    v = np.array([1, 0])
    a = rotate2(v, np.pi/2)
    b = np.array([0, 1])
    assert pytest.approx(a) == b


def test_rotate3():
    v = np.array([1, 0, 0])

    a = rotate3(v, angle_z=np.pi/2)
    b = np.array([0, 1, 0])
    assert pytest.approx(a) == b

    a = rotate3(v, angle_x=np.pi/2)
    b = np.array([1, 0, 0])
    assert pytest.approx(a) == b

    a = rotate3(v, angle_y=-np.pi/2)
    b = np.array([0, 0, 1])
    assert pytest.approx(a) == b
