"""
Copyright (c) 2019, Matt Pewsey
"""

import attr

__all__ = ['LoadCase']


@attr.s(hash=False)
class LoadCase(object):
    """
    A class representing a structural load case.

    Parameters
    ----------
    name : str
        The name of the load case.
    node_loads : list
        A list of :class:`.NodeLoad` to apply with the load case.
    elem_loads : list
        A list of :class:`.ElementLoad` to apply with the load case.
    """
    # Custom properties
    name = attr.ib()
    node_loads = attr.ib(default=[])
    elem_loads = attr.ib(default=[])

    def set_nodes(self, ndict):
        """
        Sets the node references for all node loads assigned to the load case.

        Parameters
        ----------
        ndict : dict
            A dictionary mapping node names to node objects.
        """
        for n in self.node_loads:
            n.set_node(ndict)

    def set_elements(self, edict):
        """
        Sets the element references for all element loads assigned to the load
        case.

        Parameters
        ----------
        edict : dict
            A dictionary mapping element names to element objects.
        """
        for e in self.elem_loads:
            e.set_element(edict)
