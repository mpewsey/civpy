import pytest
import numpy as np
from .tin import *


def TIN1():
    p = np.array([
        [-0.5, -0.5,   0],
        [ 0.5, -0.5,   0],
        [ 0.5,  0.5,   0],
        [-0.5,  0.5,   0],
        [   0,    0, 0.5]
    ])

    return TIN('TIN1', p)


def TIN2():
    p = np.array([
        [-0.5, -0.5,   0],
        [ 0.5, -0.5,   0],
        [ 0.5,  0.5,   0],
        [-0.5,  0.5,   0],
        [   0,    0, 0.5]
    ])

    return TIN('TIN2', p, max_edge=0.1)


def TIN3():
    p = np.array([
        [-0.5, -0.5,   0],
        [ 0.5, -0.5,   0],
        [ 0.5,  0.5,   0],
        [-0.5,  0.5,   0],
        [   0,    0, 0.5]
    ])

    # Breaklines
    b = np.linspace(0, 2*np.pi, 10)
    b = 0.5 * np.column_stack([np.cos(b), np.sin(b), np.zeros(len(b))])

    return TIN('TIN3', p, [b], step=100)


def test_repr():
    t = TIN1()
    repr(t)


def test_elevation():
    t = TIN1()
    p = [(0, 0, 1), (0, 1, 0), (0, 0.5, 0.5), (0, 0.25, 0.25)]
    zs = [t.elevation(x) for x in p]
    np.testing.assert_equal(zs, [0.5, np.nan, 0, 0.25])


def test_remove_simplices():
    t = TIN2()
    assert t.tri.simplices.shape[0] == 0


def test_breakpoints():
    t = TIN3()
    b = np.array(sorted(map(tuple, t.breaklines[0])))
    bp = np.array(sorted(map(tuple, t.breakpoints())))

    assert pytest.approx(bp) == b


def test_query_distances():
    t = TIN1()
    p = [(0, 0, 1), (0, 1, 0), (0, 0.5, 0.5), (0, 0.25, 0.25)]
    dist = []
    tin = []

    for x in p:
        a, b = t.query_distances(x, 5)
        dist.append(a[0])
        tin.append(b[0])

    dist = np.array(dist)
    tin = np.array(tin)

    a = np.array([0.5, 0.5, 0.353553391, 0])
    b = np.array([(0, 0, 0.5), (0, 0.5, 0), (0, 0.25, 0.25), (0, 0.25, 0.25)])

    assert pytest.approx(dist) == a
    assert pytest.approx(tin) == b


def test_plot_surface_3d():
    t = TIN1()
    t.plot_surface_3d()


def test_plot_surface_2d():
    t = TIN1()
    t.plot_surface_2d()


def test_plot_contour_2d():
    t = TIN1()
    t.plot_contour_2d()
