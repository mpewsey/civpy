import pytest
import numpy as np
from . import *


def Structure1():
    s = CrossSection('dummy',
        area=32.9,
        inertia_x=236,
        inertia_y=716,
        inertia_j=15.1
    )

    m = Material('dummy',
        elasticity=29000,
        rigidity=11500
    )

    g = ElementGroup('dummy', s, m)

    nodes = [
        Node('1', 0, 0, 0),
        Node('2', -240, 0, 0).fixed(),
        Node('3', 0, -240, 0).fixed(),
        Node('4', 0, 0, -240).fixed(),
    ]

    elements =[
        Element('1', '2', '1', g),
        Element('2', '3', '1', g, roll=np.deg2rad(-90)),
        Element('3', '4', '1', g, roll=np.deg2rad(-30)),
    ]

    return Structure('dummy', nodes, elements)


def test_rotation_matrix():
    struct = Structure1()
    struct._create_build()
    e1, e2, e3 = struct._build['elements']

    a = e1.rotation_matrix().ravel()
    b = np.identity(3).ravel()

    assert pytest.approx(a) == b

    a = e2.rotation_matrix().ravel()
    b = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).ravel()

    assert pytest.approx(a) == b

    a = e3.rotation_matrix().ravel()
    b = np.array([[0, 0, 1], [-0.5, 0.86603, 0], [-0.86603, -0.5, 0]]).ravel()

    assert pytest.approx(a, 0.01) == b


def test_transformation_matrix():
    struct = Structure1()
    struct._create_build()
    e1, e2, e3 = struct._build['elements']

    a = e1.transformation_matrix().ravel()
    b = np.identity(12).ravel()

    assert pytest.approx(a) == b

    a = e2.transformation_matrix().ravel()
    b = np.zeros((12, 12), dtype='float')
    r = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    b[:3,:3] = b[3:6,3:6] = b[6:9,6:9] = b[9:12,9:12] = r
    b = b.ravel()

    assert pytest.approx(a) == b

    a = e3.transformation_matrix().ravel()
    b = np.zeros((12, 12), dtype='float')
    r = np.array([[0, 0, 1], [-0.5, 0.86603, 0], [-0.86603, -0.5, 0]])
    b[:3,:3] = b[3:6,3:6] = b[6:9,6:9] = b[9:12,9:12] = r
    b = b.ravel()

    assert pytest.approx(a, 0.1) == b


def test_length():
    struct = Structure1()
    struct._create_build()
    e1, e2, e3 = struct._build['elements']

    a = e1.length()
    assert pytest.approx(a) == 240

    a = e2.length()
    assert pytest.approx(a) == 240

    a = e3.length()
    assert pytest.approx(a) == 240


def test_local_stiffness():
    struct = Structure1()
    struct._create_build()
    e1, _, _ = struct._build['elements']

    a = e1.local_stiffness().ravel()
    b = np.array([
        [3975.4, 0, 0, 0, 0, 0, -3975.4, 0, 0, 0, 0, 0],
        [0, 18.024, 0, 0, 0, 2162.9, 0, -18.024, 0, 0, 0, 2162.9],
        [0, 0, 5.941, 0, -712.92, 0, 0, 0, -5.941, 0, -712.92, 0],
        [0, 0, 0, 723.54, 0, 0, 0, 0, 0, -723.54, 0, 0],
        [0, 0, -712.92, 0, 114067, 0, 0, 0, 712.92, 0, 57033, 0],
        [0, 2162.9, 0, 0, 0, 346067, 0, -2162.9, 0, 0, 0, 173033],
        [-3975.4, 0, 0, 0, 0, 0, 3975.4, 0, 0, 0, 0, 0],
        [0, -18.024, 0, 0, 0, -2162.9, 0, 18.024, 0, 0, 0, -2162.9],
        [0, 0, -5.941, 0, 712.92, 0, 0, 0, 5.941, 0, 712.92, 0],
        [0, 0, 0, -723.54, 0, 0, 0, 0, 0, 723.54, 0, 0],
        [0, 0, -712.92, 0, 57033, 0, 0, 0, 712.92, 0, 114067, 0],
        [0, 2162.9, 0, 0, 0, 173033, 0, -2162.9, 0, 0, 0, 346067]
    ]).ravel()

    assert pytest.approx(a, 0.01) == b


def test_global_stiffness():
    struct = Structure1()
    struct._create_build()
    e1, e2, e3 = struct._build['elements']

    a = e1.global_stiffness().ravel()
    b = np.array([
        [3975.4, 0, 0, 0, 0, 0, -3975.4, 0, 0, 0, 0, 0],
        [0, 18.024, 0, 0, 0, 2162.9, 0, -18.024, 0, 0, 0, 2162.9],
        [0, 0, 5.941, 0, -712.92, 0, 0, 0, -5.941, 0, -712.92, 0],
        [0, 0, 0, 723.54, 0, 0, 0, 0, 0, -723.54, 0, 0],
        [0, 0, -712.92, 0, 114067, 0, 0, 0, 712.92, 0, 57033, 0],
        [0, 2162.9, 0, 0, 0, 346067, 0, -2162.9, 0, 0, 0, 173033],
        [-3975.4, 0, 0, 0, 0, 0, 3975.4, 0, 0, 0, 0, 0],
        [0, -18.024, 0, 0, 0, -2162.9, 0, 18.024, 0, 0, 0, -2162.9],
        [0, 0, -5.941, 0, 712.92, 0, 0, 0, 5.941, 0, 712.92, 0],
        [0, 0, 0, -723.54, 0, 0, 0, 0, 0, 723.54, 0, 0],
        [0, 0, -712.92, 0, 57033, 0, 0, 0, 712.92, 0, 114067, 0],
        [0, 2162.9, 0, 0, 0, 173033, 0, -2162.9, 0, 0, 0, 346067]
    ]).ravel()

    assert pytest.approx(a, 0.01) == b

    a = e2.global_stiffness().ravel()
    b = np.array([
        [5.941, 0, 0, 0, 0, -712.92, -5.941, 0, 0, 0, 0, -712.92],
        [0, 3975.4, 0, 0, 0, 0, 0, -3975.4, 0, 0, 0, 0],
        [0, 0, 18.024, 2162.9, 0, 0, 0, 0, -18.024, 2162.9, 0, 0],
        [0, 0, 2162.9, 346067, 0, 0, 0, 0, -2162.9, 173033, 0, 0],
        [0, 0, 0, 0, 723.54, 0, 0, 0, 0, 0, -723.54, 0],
        [-5.941, 0, 0, 0, 0, 712.92, 5.941, 0, 0, 0, 0, 712.92],
        [0, -3975.4, 0, 0, 0, 0, 0, 3975.4, 0, 0, 0, 0],
        [0, 0, -18.024, -2162.9, 0, 0, 0, 0, 18.024, -2162.9, 0, 0],
        [0, 0, 2162.9, 173033, 0, 0, 0, 0, -2162.9, 346067, 0, 0],
        [0, 0, 0, 0, -723.54, 0, 0, 0, 0, 0, 723.54, 0],
        [-712.92, 0, 0, 0, 0, 57033, 712.92, 0, 0, 0, 0, 114067]
    ]).ravel()

    a = e3.global_stiffness().ravel()
    b = np.array([
        [8.9618, -5.2322, 0, 627.87, 1075.4, 0, -8.9618, 5.2322, 0, 627.87, 1075.4, 0],
        [-5.2322, 15.003, 0, -1800.4, -627.87, 0, 5.2322, -15.003, 0, -1800.4, -627.87, 0],
        [0, 0, 3975.4, 0, 0, 0, 0, 0, -3975.4, 0, 0, 0],
        [627.87, -1800.4, 0, 288067, 100459, 0, -627.87, 1800.4, 0, 144033, 50229, 0],
        [1075.4, -627.87, 0, 100459, 172067, 0, -1075.4, 627.87, 0, 50229, 86033, 0],
        [0, 0, 0, 0, 0, 723.54, 0, 0, 0, 0, 0, -723.54],
        [-8.9618, 5.2322, 0, -627.87, -1075.4, 0, 8.9618, -5.2322, 0, -627.87, -1075.4, 0],
        [5.2322, -15.003, 0, 1800.4, 627.87, 0, -5.2322, 15.003, 0, 1800.4, 627.87, 0],
        [0, 0, -3975.4, 0, 0, 0, 0, 0, 3975.4, 0, 0, 0],
        [627.87, -1800.4, 0, 144033, 50229, 0, -627.87, 1800.4, 0, 288067, 100459, 0],
        [1075.4, -627.87, 0, 50229, 86033, 0, -1075.4, 627.87, 0, 100459, 172067, 0],
        [0, 0, 0, 0, 0, -723.54, 0, 0, 0, 0, 0, 723.54]
    ]).ravel()

    assert pytest.approx(a, 0.01) == b


def test_plot_2d():
    struct = Structure1()
    struct.plot_2d()


def test_plot_3d():
    struct = Structure1()
    struct.plot_3d()


def test_linear_analysis():
    struct = Structure1()

    nloads = [NodeLoad('1', mx=-1800, mz=1800)]
    eloads = [ElementLoad('1', fy=-0.25)]
    lc = LoadCase('1', nloads, eloads)

    r = struct.linear_analysis(lc)

    # Test global loads
    df = r['glob']

    # Loads
    s = ['force_x', 'force_y', 'force_z', 'moment_x', 'moment_y', 'moment_z']

    a = np.array(df[df['node'] == '1'][s].iloc[0])
    b = np.array([0, 0, 0, -1800, 0, 1800])
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['node'] == '2'][s].iloc[0])
    b = np.array([5.3757, 44.106, -0.74272, 2.1722, 58.987, 2330.5])
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['node'] == '3'][s].iloc[0])
    b = np.array([-4.6249, 11.117, -6.4607, -515.55, -0.76472, 369.67])
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['node'] == '4'][s].iloc[0])
    b = np.array([-0.75082, 4.7763, 7.2034, -383.5, -60.166, -4.702])
    assert pytest.approx(a, 0.01) == b

    # Deflections
    s = ['defl_x', 'defl_y', 'defl_z', 'rot_x', 'rot_y', 'rot_z']

    a = np.array(df[df['node'] == '1'][s].iloc[0])
    b = np.array([-1.3522, -2.7965, -1.812, -3.0021, 1.0569, 6.4986]) * 1e-3
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['node'] == '2'][s].iloc[0])
    b = np.zeros(6)
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['node'] == '3'][s].iloc[0])
    b = np.zeros(6)
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['node'] == '4'][s].iloc[0])
    b = np.zeros(6)
    assert pytest.approx(a, 0.01) == b


    # Test reactions
    df = r['react']
    s = ['force_x', 'force_y', 'force_z', 'moment_x', 'moment_y', 'moment_z']

    a = np.array(df[df['node'] == '2'][s].iloc[0])
    b = np.array([5.3757, 44.106, -0.74272, 2.1722, 58.987, 2330.5])
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['node'] == '3'][s].iloc[0])
    b = np.array([-4.6249, 11.117, -6.4607, -515.55, -0.76472, 369.67])
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['node'] == '4'][s].iloc[0])
    b = np.array([-0.75082, 4.7763, 7.2034, -383.5, -60.166, -4.702])
    assert pytest.approx(a, 0.01) == b


    # Test local loads
    df = r['loc']

    # Loads
    s = ['i_axial', 'i_shear_x', 'i_shear_y',
         'i_torsion', 'i_moment_x', 'i_moment_y',
         'j_axial', 'j_shear_x', 'j_shear_y',
         'j_torsion', 'j_moment_x', 'j_moment_y']

    a = np.array(df[df['element'] == '1'][s].iloc[0])
    b = np.array([5.3757, 44.106, -0.74272, 2.1722, 58.987, 2330.5,
                  -5.3757, 15.894, 0.74272, -2.1722, 119.27, 1055])
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['element'] == '2'][s].iloc[0])
    b = np.array([11.117, -6.4607, -4.6249, -0.76472, 369.67, -515.55,
                  -11.117, 6.4607, 4.6249, 0.76472, 740.31, -1035])
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['element'] == '3'][s].iloc[0])
    b = np.array([7.2034, 4.5118, -1.7379, -4.702, 139.65, 362.21,
                  -7.2034, -4.5118, 1.7379, 4.702, 277.46, 720.63])
    assert pytest.approx(a, 0.01) == b

    # Deflections
    s = ['i_defl_ax', 'i_defl_x', 'i_defl_y',
         'i_twist', 'i_rot_x', 'i_rot_y',
         'j_defl_ax', 'j_defl_x', 'j_defl_y',
         'j_twist', 'j_rot_x', 'j_rot_y']

    a = np.array(df[df['element'] == '1'][s].iloc[0])
    b = np.array([0, 0, 0, 0, 0, 0,
                  -1.3522, -2.7965, -1.812, -3.0021, 1.0569, 6.4986]) * 1e-3
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['element'] == '2'][s].iloc[0])
    b = np.array([0, 0, 0, 0, 0, 0,
                  -2.7965, -1.812, -1.3522, 1.0569, 6.4986, -3.0021]) * 1e-3
    assert pytest.approx(a, 0.01) == b

    a = np.array(df[df['element'] == '3'][s].iloc[0])
    b = np.array([0, 0, 0, 0, 0, 0,
                  -1.812, -1.7457, 2.5693, 6.4986, 2.4164, 2.0714]) * 1e-3
    assert pytest.approx(a, 0.01) == b
