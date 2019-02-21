# structure_ex2.py
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
    Node('1', 20, 20, 0, 'xy').fixed(),
    Node('2', 15, 15, 50, 'xy')
]

elements =[
    Element('1', '1_p', '1_x', group, 'y'),
    Element('2', '1_p', '1_y', group, 'x'),
    Element('3', '2_p', '2_x', group, 'y'),
    Element('4', '2_p', '2_y', group, 'x'),
    Element('5', '1_p', '2_p', group, 'xy'),
    Element('6', '1_p', '2_x', group, 'xy'),
    Element('7', '1_x', '2_xy', group, 'xy')
]

struct = Structure(
    name='dummy',
    nodes=nodes,
    elements=elements,
    symmetry=True
)

struct.plot_3d()
