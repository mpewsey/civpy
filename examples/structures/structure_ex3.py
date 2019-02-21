# structure_ex3.py
import numpy as np
from civpy.structures import *

section = CrossSection(
    name='dummy',
    area=1
)

material = Material(
    name='dummy',
    elasticity=29000
)

group = ElementGroup(
    name='dummy',
    section=section,
    material=material
)

nodes = [
    Node('1', 120, 120).fixed(),
    Node('2', 120, 0, fx_free=False),
    Node('3', 0, 0),
    Node('4', 0, 120)
]

elements =[
    Element('1', '1', '2', group),
    Element('2', '3', '2', group),
    Element('3', '3', '4', group),
    Element('4', '4', '1', group),
    Element('5', '1', '3', group),
    Element('6', '4', '2', group)
]

struct = Structure(
    name='dummy',
    nodes=nodes,
    elements=elements
)

struct.plot_2d()
