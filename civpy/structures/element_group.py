"""
Copyright (c) 2019, Matt Pewsey
"""

import attr

__all__ = ['ElementGroup']


@attr.s(hash=False)
class ElementGroup(object):
    """
    A class representing a group of element properties.

    Parameters
    ----------
    name : str
        The name of the group.
    section : :class:`.CrossSection`
        The group cross section.
    material : :class:`.Material`
        The group material.
    """
    name = attr.ib()
    section = attr.ib()
    material = attr.ib()
