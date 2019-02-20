from .material import *


def test_repr():
    m = Material('dummy', elasticity=29000, rigidity=11500)
    repr(m)
