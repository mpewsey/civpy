import pytest
import numpy as np
from .spatial_hash import *


def SpatialHash1():
    np.random.seed(1283943)
    x = np.random.uniform(-60, 60, (2500, 2))
    return SpatialHash(x, 5)


def test_init():
    s = SpatialHash1()

    # Test that values were set properly
    assert s.points.shape == (2500, 2)
    assert s._dim == 2
    assert s._grid == 5


def test_repr():
    s = SpatialHash1()
    repr(s)


def test_bad_point():
    s = SpatialHash1()

    with pytest.raises(ValueError):
        s.query_point([0, 0, 0], 25)


def test_query_point():
    s = SpatialHash1()
    x = np.array([0, 0])

    i = s.query_point(x, 25, 10)
    p = s.points[i]

    dist = np.linalg.norm(x - s.points, axis=1)
    j = (dist <= 25) & (dist >= 10)
    q = s.points[j][dist[j].argsort()]

    assert (p == q).all()


def test_query_range_same_point():
    s = SpatialHash1()
    x = np.array([0, 0])

    i = s.query_range(x, x, 25, 10)
    p = s.points[i]

    dist = np.linalg.norm(x - s.points, axis=1)
    j = (dist <= 25) & (dist >= 10)
    q = s.points[j][dist[j].argsort()]

    assert (p == q).all()


def test_query_range():
    a = np.array([-30, -30])
    b = np.array([30, 30])
    s = SpatialHash1()

    i = s.query_range(a, b, 25, 10)
    q = s.points[i]

    unit = b - a
    length = np.linalg.norm(unit)
    unit = unit / length

    p = s.points - a
    proj = np.array([np.dot(x, unit) for x in p])
    proj = np.expand_dims(proj, 1)

    p = proj * unit + a
    off = np.linalg.norm(p - s.points, axis=1)
    proj = proj.ravel()

    j = (off <= 25) & (off >= 10) & (proj >= 0) & (proj <= length)
    p = s.points[j][off[j].argsort()]

    assert (p == q).all()


def test_plot_1d():
    np.random.seed(1283943)
    x = np.random.uniform(-60, 60, (2500, 1))
    s = SpatialHash(x, 5)
    s.plot()


def test_plot_2d():
    s = SpatialHash1()
    s.plot()


def test_plot_3d():
    np.random.seed(1283943)
    x = np.random.uniform(-60, 60, (2500, 3))
    s = SpatialHash(x, 5)
    s.plot()


def test_plot_invalid_dim():
    np.random.seed(1283943)
    x = np.random.uniform(-60, 60, (2500, 4))
    s = SpatialHash(x, 5)

    with pytest.raises(ValueError):
        s.plot()
