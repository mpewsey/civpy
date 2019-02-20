import propy

__all__ = ['LoadCase']


class LoadCase(object):
    # Custom properties
    name = propy.str_property('name')

    def __init__(self, name, node_loads=[], elem_loads=[]):
        self.name = name
        self.node_loads = node_loads
        self.elem_loads = elem_loads

    __repr__ = propy.repr_method('name', 'node_loads', 'elem_loads')

    def set_nodes(self, ndict):
        for n in self.node_loads:
            n.set_node(ndict)

    def set_elements(self, edict):
        for e in self.elem_loads:
            e.set_element(edict)
