import pytest
from .survey_stake import *


def test_init():
    args = [0] * 10

    with pytest.raises(ValueError):
        SurveyStake(*args)


def test_repr():
    p = SurveyStake.init_xy(1, 2, 3)
    repr(p)


def test_init_xy():
    p = SurveyStake.init_xy(
        x=1,
        y=2,
        z=3
    )

    assert p.x == 1
    assert p.y == 2
    assert p.z == 3
    assert p.lock_z == False
    assert p._type == 'xy'
    assert p.height == 0
    assert p.rotation == 0


def test_init_station():
    p = SurveyStake.init_station(
        station=1,
        offset=2,
        z=3
    )

    assert p.station == 1
    assert p.offset == 2
    assert p.z == 3
    assert p.lock_z == False
    assert p._type == 'station'
    assert p.height == 0
    assert p.rotation == 0
