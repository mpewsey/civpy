from .element_load import *


def test_repr():
    e = ElementLoad('1', fy=-0.25)
    repr(e)
