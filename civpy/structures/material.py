"""
Copyright (c) 2019, Matt Pewsey
"""

import attr

__all__ = ['Material']


@attr.s(hash=False)
class Material(object):
    """
    A class representing an engineered material.

    Parameters
    ----------
    name : str
        The name of the material.
    elasticity : float
        The modulus of elasticity.
    rigidity : float
        The modulus of rigidity.
    """
    name = attr.ib()
    elasticity = attr.ib()
    rigidity = attr.ib(default=0)
