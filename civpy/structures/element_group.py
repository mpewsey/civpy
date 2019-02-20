import propy

__all__ = ['ElementGroup']


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
    # Custom properties
    name = propy.str_property('name')

    def __init__(self, name, section, material):
        self.name = name
        self.section = section
        self.material = material

    __repr__ = propy.repr_method('name', 'section', 'material')
