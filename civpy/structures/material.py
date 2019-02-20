import propy

__all__ = ['Material']


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
    # Custom properties
    name = propy.str_property('name')

    def __init__(self, name, elasticity, rigidity=0):
        self.name = name
        self.elasticity = elasticity
        self.rigidity = rigidity

    __repr__ = propy.repr_method('name', 'elasticity', 'rigidity')
