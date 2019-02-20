from .load_case import *


def test_repr():
    lc = LoadCase('dummy', None, None)
    repr(lc)
