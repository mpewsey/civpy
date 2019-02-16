from .pi import *


def test_init():
    p = PI(1, 2, 3, 4)

    assert p.x == 1
    assert p.y == 2
    assert p.z == 3
    assert p.radius == 4


def test_repr():
    p = PI(1, 2, 3, 4)
    repr(p)
