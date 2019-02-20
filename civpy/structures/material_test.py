from .material import *


def test_repr():
    m = Material('dummy', elasticity=29_000, rigidity=11_500)
    repr(m)
