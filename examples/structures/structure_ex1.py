# structure_ex1.py
import numpy as np
from civpy.structures import *

section = CrossSection(
    name='dummy',
    area=32.9,
    inertia_x=236,
    inertia_y=716,
    inertia_j=15.1
)

material = Material(
    name='dummy',
    elasticity=29000,
    rigidity=11500
)

group = ElementGroup(
    name='dummy',
    section=section,
    material=material
)

nodes = [
    Node('1', 0, 0, 0),
    Node('2', -240, 0, 0).fixed(),
    Node('3', 0, -240, 0).fixed(),
    Node('4', 0, 0, -240).fixed(),
]

elements =[
    Element('1', '2', '1', group),
    Element('2', '3', '1', group, roll=np.deg2rad(-90)),
    Element('3', '4', '1', group, roll=np.deg2rad(-30)),
]

nloads = [
    NodeLoad('1', mx=-1800, mz=1800)
]

eloads = [
    ElementLoad('1', fy=-0.25)
]

lc = LoadCase('1', nloads, eloads)

struct = Structure(
    name='dummy',
    nodes=nodes,
    elements=elements
)

struct.plot_3d()
result = struct.linear_analysis(lc)

result['glob']
#   load_case node   force_x    force_y   force_z     moment_x   moment_y  \
# 0         1    1  0.000000   0.000000  0.000000 -1800.000000   0.000000
# 1         1    2  5.375736  44.106293 -0.742724     2.172151  58.987351
# 2         1    3 -4.624913  11.117379 -6.460651  -515.545730  -0.764719
# 3         1    4 -0.750823   4.776328  7.203376  -383.501559 -60.166419

#       moment_z    defl_x    defl_y    defl_z     rot_x     rot_y     rot_z
# 0  1800.000000 -0.001352 -0.002797 -0.001812 -0.003002  0.001057  0.006499
# 1  2330.519663  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
# 2   369.671654  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
# 3    -4.701994  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
