from .survey_point import *


def test_init():
    p = SurveyPoint(1, 2, 3)

    assert p.x == 1
    assert p.y == 2
    assert p.z == 3
    assert p.meta == {}

def test_repr():
    p = SurveyPoint(1, 2, 3)
    repr(p)
