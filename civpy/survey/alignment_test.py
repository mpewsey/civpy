import pytest
import numpy as np
from .pi import *
from .survey_stake import *
from .alignment import *


def Alignment1():
    p = [
        #  x,  y
        (  0,  0),
        ( 10, 10),
        ( 10, 20),
        (  0, 30),
        (-10, 20),
        (-10, 10),
        (  0,  0)
    ]

    p = [PI(*x) for x in p]

    return Alignment('Alignment1', pis=p)


def Alignment2():
    p = [
        #   x,    y  z,  r
        (-100, -200, 0,  0),
        (-200, -200, 0, 40),
        (-200,  200, 0, 40),
        ( 200,  200, 0, 40),
        ( 200, -200, 0, 40),
        ( 100, -200, 0, 40),
        ( 100,  100, 0, 40),
        (-100,  100, 0, 40),
        (-100, -100, 0, 40),
        (   0, -100, 0, 40),
        (   0,    0, 0,  0)
    ]

    q = [
        # sta, off, z, ht,      rot,
        (   0,  30, 0, 0, -0.78539),
        ( 100,  30, 0, 0,        0),
        ( 300, -30, 0, 0,        0),
        ( 475, -30, 0, 0,        0),
        (1000,   0, 0, 0,        0),
        (1975, -30, 0, 0,        0)
    ]

    p = [PI(*x) for x in p]
    q = [SurveyStake.init_station(*x) for x in q]

    return Alignment('Alignment2', pis=p, stakes=q)


def Alignment3():
    p = [
        #  x,   y, z,  r
        (  0,   0, 0,  0),
        (100,   0, 0, 10),
        (100, 100, 0, 10),
        (200, 100, 0,  0)
    ]

    p = [PI(*x) for x in p]

    return Alignment('Alignment3', pis=p)


def Alignment4():
    p = [
        # x,  y, z, r
        ( 0,  0, 0, 0),
        (10,  0, 0, 0),
        (10, 10, 0, 5),
        ( 0, 10, 0, 0)
    ]

    p = [PI(*x) for x in p]

    return Alignment('Alignment4', pis=p)


def Alignment5():
    p = [
        #  x,   y, z,  r
        (-10, -10, 0,  0),
        ( 10, -10, 0, 10),
        ( 10,  10, 0, 10),
        (-10,  10, 0, 10),
        (-10, -10, 0, 10),
        ( 10, -10, 0,  0),
        ( 10,  10, 0,  0),
        (-10,  10, 0,  0),
        (-10, -10, 0,  0)
    ]

    p = [PI(*x) for x in p]

    return Alignment('Alignment5', pis=p)


def test_pi_coordinates():
    p = np.array([
        #  x,  y, z
        (  0,  0, 0),
        ( 10, 10, 0),
        ( 10, 20, 0),
        (  0, 30, 0),
        (-10, 20, 0),
        (-10, 10, 0),
        (  0,  0, 0)
    ]).ravel()

    align = Alignment1()
    q = align.pi_coordinates().ravel()

    assert pytest.approx(p) == q


def test_pi_radii():
    align = Alignment2()
    a = align.pi_radii()
    b = np.array([0, 40, 40, 40, 40, 40, 40, 40, 40, 40, 0])

    assert pytest.approx(a) == b


def test_azimuths1():
    align = Alignment1()
    a = align.azimuths() * 180/np.pi
    b = np.array([45, 0, -45, -135, 180, 135, 135])

    assert pytest.approx(a) == b


def test_azimuths2():
    align = Alignment1()
    align.pis = list(reversed(align.pis))

    a = align.azimuths() * 180/np.pi
    b = np.array([-45, 0, 45, 135, 180, -135, -135])

    assert pytest.approx(a) == b


def test_deflection_angles1():
    align = Alignment1()
    a = align.deflection_angles() * 180/np.pi
    b = np.array([0, -45, -45, -90, -45, -45, 0])

    assert pytest.approx(a) == b


def test_deflection_angles2():
    align = Alignment1()
    align.pis = list(reversed(align.pis))

    a = align.deflection_angles() * 180/np.pi
    b = np.array([0, 45, 45, 90, 45, 45, 0])

    assert pytest.approx(a) == b


def test_curve_lengths():
    align = Alignment3()
    a = align.curve_lengths()
    b = np.array([0, 15.70796326794896, 15.70796326794896, 0])

    assert pytest.approx(a) == b


def test_tangent_ordinates():
    align = Alignment3()
    a = align.tangent_ordinates()
    b = np.array([0, 10, 10, 0])

    assert pytest.approx(a) == b


def test_chord_distances():
    align = Alignment3()
    a = align.chord_distances()
    b = np.array([0, 14.142135623730951, 14.142135623730951, 0])

    assert pytest.approx(a) == b


def test_middle_ordinates():
    align = Alignment3()
    a = align.middle_ordinates()
    b = np.array([0, 2.9289321881345245, 2.9289321881345245, 0])

    assert pytest.approx(a) == b


def test_external_ordinates():
    align = Alignment3()
    a = align.external_ordinates()
    b = np.array([0, 4.142135623730951, 4.142135623730951, 0])

    assert pytest.approx(a) == b


def test_pc_coordinates():
    align = Alignment3()
    a = align.pc_coordinates().ravel()
    b = np.array([[0, 0], [90, 0], [100, 90], [200, 100]]).ravel()

    assert pytest.approx(a) == b


def test_pt_coordinates():
    align = Alignment3()
    a = align.pt_coordinates().ravel()
    b = np.array([[0, 0], [100, 10], [110, 100], [200, 100]]).ravel()

    assert pytest.approx(a) == b


def test_rp_coordinates():
    align = Alignment3()
    a = align.rp_coordinates().ravel()
    b = np.array([[0, 0], [90, 10], [110, 90], [200, 100]]).ravel()

    assert pytest.approx(a) == b


def test_mpc_coordinates():
    align = Alignment3()
    a = align.mpc_coordinates().ravel()
    b = np.array([
        [           0,            0],
        [ 97.07106781,   2.92893219],
        [102.92893219,  97.07106781],
        [         200,          100]
    ]).ravel()

    assert pytest.approx(a) == b


def test_pc_stations():
    align = Alignment3()
    a = align.pc_stations()
    b = np.array([0, 90, 185.70796327, 291.41592654])

    assert pytest.approx(a) == b


def test_pt_stations():
    align = Alignment3()
    a = align.pt_stations()
    b = np.array([0, 105.70796327, 201.41592654, 291.41592654])

    assert pytest.approx(a) == b


def test_mpc_stations():
    align = Alignment3()
    a = align.mpc_stations()
    b = np.array([0, 97.85398163, 193.5619449, 291.41592654])

    assert pytest.approx(a) == b


def test_segment_indices():
    align = Alignment4()
    p = [0, 5, 10, 17, 25]
    a = align.segment_indices(p).ravel()
    b = np.array([[1, 0], [1, 0], [2, 1], [2, 2], [1, 2]]).ravel()

    assert pytest.approx(a) == b


def test_poc_transforms():
    align = Alignment4()
    a = align.poc_transforms().ravel()

    b = np.array([
        [[0, -1], [1, 0]],
        [[0.707106781, -0.707106781], [0.707106781, 0.707106781]],
        [[0.707106781, 0.707106781], [-0.707106781, 0.707106781]],
        [[0, 1], [-1, 0]]
    ]).ravel()

    assert pytest.approx(a) == b


def test_pot_transforms():
    align = Alignment4()
    a = align.pot_transforms().ravel()

    b = np.array([
        [[0, -1], [1, 0]],
        [[1, 0], [0, 1]],
        [[0, 1], [-1, 0]],
        [[0, 1], [-1, 0]]
    ]).ravel()

    assert pytest.approx(a) == b


def test_coordinates():
    align = Alignment5()

    p = np.array([
        (0, 0, 1),
        (10, 0, 2),
        (17.853981633974485, 0, 3),
        (17.853981633974485, -20, 4),
        (17.853981633974485, -10, 5),
        (17.853981633974485, 0, 6),
        (17.853981633974485, 10, 7),
        (25.707963267948966, 0, 8),
        (102.83185307, 0, 9),
        (122.83185307, 0, 10),
        (102.83185307, -14.142135623730951, 11),
        (122.83185307, -14.142135623730951, 12)
    ])

    a = align.coordinates(p)

    b = np.array([
        [-10, -10, 1],
        [0, -10, 2],
        [7.07106781, -7.07106781, 3],
        [-7.07106781, 7.07106781, 4],
        [0, 0, 5],
        [7.07106781, -7.07106781, 6],
        [14.1421356, -14.1421356, 7],
        [10, 0, 8],
        [10, 10, 9],
        [-10, 10, 10],
        [0, 0, 11],
        [0, 0, 12]
    ])

    assert pytest.approx(a) == b


def test_station_coordinates():
    np.random.seed(234098)
    align = Alignment5()
    p = np.random.uniform(-30, 30, (1000, 3))

    r = [
        [-10, -10, 1],
        [ 10,  10, 2],
        [ 10, -10, 3],
        [-10,  10, 4],
        [  0,   0, 5]
    ]

    p = np.concatenate([p, r])
    q = align.station_coordinates(p)
    assert len(q) > 0

    for i, x in q.items():
        b = align.coordinates(x)
        for a in b:
            assert pytest.approx(a) == p[i]


def test_set_stake_xy():
    align = Alignment5()

    p = np.array([
        (0, 0, 1),
        (10, 0, 2),
        (17.853981633974485, 0, 3),
        (17.853981633974485, -20, 4),
        (17.853981633974485, -10, 5),
        (17.853981633974485, 0, 6),
        (17.853981633974485, 10, 7),
        (25.707963267948966, 0, 8),
        (102.83185307, 0, 9),
        (122.83185307, 0, 10),
        (102.83185307, -14.142135623730951, 11),
        (122.83185307, -14.142135623730951, 12)
    ])

    align.stakes = [SurveyStake.init_station(*x) for x in p]

    align.set_stake_xy()
    a = np.array(align.stakes).ravel()

    b = np.array([
        [-10, -10, 1],
        [0, -10, 2],
        [7.07106781, -7.07106781, 3],
        [-7.07106781, 7.07106781, 4],
        [0, 0, 5],
        [7.07106781, -7.07106781, 6],
        [14.1421356, -14.1421356, 7],
        [10, 0, 8],
        [10, 10, 9],
        [-10, 10, 10],
        [0, 0, 11],
        [0, 0, 12]
    ]).ravel()

    assert pytest.approx(a) == b


def test_plot_plan():
    np.random.seed(238479)
    align = Alignment2()
    align.plot_plan()
