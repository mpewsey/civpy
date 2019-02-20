from .node import *


def test_repr():
    n = Node('1', 1, 2, 3)
    repr(Node)


def test_str():
    n = Node('1', 1, 2, 3)
    assert str(n) == '1'


def test_copy():
    n = Node('1', 1, 2, 3)
    m = n.copy()

    assert n is not m
    assert repr(n) == repr(m)


def test_fixities():
    n = Node('1', 1, 2, 3)
    a = tuple(n.fixities())
    b = tuple([True] * 6)

    assert a == b


def test_sym_nodes():
    # No symmetry
    n = Node('1', 1, 2, 3, symmetry=None)
    p, = n.sym_nodes()

    assert (p.name, p.x, p.y, p.z) == ('1_p', 1, 2, 3)

    # X symmetry
    n = Node('1', 1, 2, 3, symmetry='x')
    p, x = n.sym_nodes()

    assert (p.name, p.x, p.y, p.z) == ('1_p', 1, 2, 3)
    assert (x.name, x.x, x.y, x.z) == ('1_x', 1, -2, 3)

    # Y symmetry
    n = Node('1', 1, 2, 3, symmetry='y')
    p, y = n.sym_nodes()

    assert (p.name, p.x, p.y, p.z) == ('1_p', 1, 2, 3)
    assert (y.name, y.x, y.y, y.z) == ('1_y', -1, 2, 3)

    # XY symmetry
    n = Node('1', 1, 2, 3, symmetry='xy')
    p, x, y, xy = n.sym_nodes()

    assert (p.name, p.x, p.y, p.z) == ('1_p', 1, 2, 3)
    assert (x.name, x.x, x.y, x.z) == ('1_x', 1, -2, 3)
    assert (y.name, y.x, y.y, y.z) == ('1_y', -1, 2, 3)
    assert (xy.name, xy.x, xy.y, xy.z) == ('1_xy', -1, -2, 3)
