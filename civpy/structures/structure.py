import propy
import inspect
import numpy as np
import pandas as pd
from functools import wraps
import matplotlib.pyplot as plt
from .element import clear_element_cache
from .element_load import clear_element_load_cache
from ..math import rotation_matrix3

__all__ = ['Structure']


def build(func):
    """
    A decorator that ensures the :class:`.Structure` is built prior to
    executing the wrapped method.
    """
    def index(lst, s):
        for i, x in enumerate(lst):
            if x == s:
                return i
        return -1

    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        # If already built, perform the operation
        if obj.build:
            return func(obj, *args, **kwargs)

        # If not built, build it, perform the operation, and destroy the build
        if n == -1:
            lc = []
        elif n-1 < len(args):
            lc = args[n-1]
        else:
            lc = kwargs['lc']

        if not isinstance(lc, (list, tuple)):
            lc = [lc]

        obj._create_build(lc)
        result = func(obj, *args, **kwargs)
        obj.build.clear()
        clear_element_cache()
        clear_element_load_cache()

        return result

    # Determine if lc argument exists
    n = index(inspect.getfullargspec(func).args, 'lc')

    return wrapper


class Structure(object):
    """
    A class representing a structure.

    Parameters
    ----------
    name : str
        The name of the structure.
    nodes : list
        A list of :class:`.Node`.
    elements : list
        A list of :class:`.Element`.
    symmetry : bool
        If True, symmetry will be applied to the structure.

    Examples
    --------
    The following example creates an a structure and performs linear analysis
    for a load case.

    .. plot:: ../examples/structures/structure_ex1.py
        :include-source:
    """
    # Custom properties
    name = propy.str_property('name')
    symmetry = propy.bool_property('symmetry')

    def __init__(self, name, nodes, elements, symmetry=False):
        self.name = name
        self.nodes = nodes
        self.elements = elements
        self.symmetry = symmetry
        self.build = {}

    def _create_build(self, load_cases=[]):
        """
        Builds the structure and places all components in the object
        model dictionary.

        Parameters
        ----------
        load_cases : list
            A list of :class:`.LoadCase`.
        """
        if not self.symmetry:
            nodes = self.nodes
            elements = self.elements
        else:
            # Make symmetric components
            nodes = []
            for n in self.nodes:
                nodes += n.sym_nodes()

            elements = []
            for e in self.elements:
                elements += e.sym_elements()

        ndict = {n.name: n for n in nodes}
        edict = {e.name: e for e in elements}

        # Set nodes to elements
        for e in elements:
            e.set_nodes(ndict)

        # Set nodes and elements to loads
        for lc in load_cases:
            lc.set_nodes(ndict)
            lc.set_elements(edict)

        ndict = {n.name: 6*i for i, n in enumerate(nodes)}
        edict = {e.name: i for i, e in enumerate(elements)}

        self.build = {
            'nodes': nodes,
            'elements': elements,
            'ndict': ndict,
            'edict': edict,
            'load_cases': load_cases
        }

    @build
    def plot_3d(self, ax=None, symbols={}):
        """
        Plots the structure in 3D.

        Parameters
        ----------
        ax
            The axes to which the plot will be added. If None, a new figure
            and axes will be created.
        symbols : dict
            The plot symbols with any of the following keys:

                * 'nodes': The node point symbols, default is 'r.'
                * 'elements': The element lines, default is 'b--'.
        """
        # Build the structure
        x = np.array(self.build['nodes'])

        # Create figure is one not provided
        if ax is None:
            mx = x.max(axis=0)
            c = 0.5 * (mx + x.min(axis=0))
            rng = 1.1 * np.max(mx - c)
            xlim, ylim, zlim = np.column_stack([c - rng, c + rng])

            fig = plt.figure()
            ax = fig.add_subplot(111,
                projection='3d',
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                xlabel='X',
                ylabel='Y',
                zlabel='Z',
                aspect='equal'
            )

        # Symbols
        sym = dict(
            elements='b--',
            nodes='r.',
            ntext='k',
            etext='r'
        )
        sym.update(symbols)

        # Plot elements
        if sym['elements'] is not None:
            for e in self.build['elements']:
                e = np.array(e.get_nodes())
                ax.plot(e[:,0], e[:,1], e[:,2], sym['elements'])

        # Plot element text
        if sym['etext'] is not None:
            for e in self.build['elements']:
                p, q = e.get_nodes()
                p = (q - p) / 3 + p
                ax.text(p[0], p[1], p[2], e.name, ha='center',
                        va='center', color=sym['etext'])

        # Plot nodes
        if sym['nodes'] is not None:
            ax.plot(x[:,0], x[:,1], x[:,2], sym['nodes'])

        # Plot node text
        if sym['ntext'] is not None:
            for n in self.build['nodes']:
                ax.text(n[0], n[1], n[2], n.name, color=sym['ntext'])

        return ax

    @build
    def plot_2d(self, ax=None, angle_x=0, angle_y=0,
                angle_z=0, symbols={}):
        """
        Plots the 2D projection of the structure.

        Parameters
        ----------
        ax
            The axes to which the plot will be added. If None, a new figure
            and axes will be created.
        angle_x, angle_y, angle_z : float
            The rotation angles about the x, y, and z axes.
        symbols : dict
            The plot symbols with any of the following keys:

                * 'nodes': The node point symbols, default is 'r.'
                * 'elements': The element lines, default is 'b--'.
        """
        # Build the structure
        r = rotation_matrix3(angle_x, angle_y, angle_z).T
        x = np.array(self.build['nodes']).dot(r)

        # Create figure is one not provided
        if ax is None:
            mx = x.max(axis=0)
            c = 0.5 * (mx + x.min(axis=0))
            rng = 1.1 * np.max(mx - c)
            xlim, ylim, _ = np.column_stack([c - rng, c + rng])

            fig = plt.figure()
            ax = fig.add_subplot(111,
                xlim=xlim,
                ylim=ylim,
                xlabel="X'",
                ylabel="Y'",
                aspect='equal'
            )

        # Symbols
        sym = dict(
            elements='b--',
            nodes='r.',
            ntext='k',
            etext='r'
        )
        sym.update(symbols)

        # Plot elements
        if sym['elements'] is not None:
            for e in self.build['elements']:
                e = np.array(e.get_nodes()).dot(r)
                ax.plot(e[:,0], e[:,1], sym['elements'])

        # Plot element text
        if sym['etext'] is not None:
            for e in self.build['elements']:
                p = np.array(e.get_nodes()).dot(r)
                p = (p[1] - p[0]) / 3 + p[0]
                ax.text(p[0], p[1], e.name, ha='center',
                        va='center', color=sym['etext'])

        # Plot nodes
        if sym['nodes'] is not None:
            ax.plot(x[:,0], x[:,1], sym['nodes'])

        # Plot node text
        if sym['ntext'] is not None:
            for n in self.build['nodes']:
                p = n.dot(r)
                ax.text(p[0], p[1], n.name, color=sym['ntext'])

        return ax

    @build
    def global_stiffness(self, defl=None):
        """
        Returns the global stiffness matrix for the structure.

        Parameters
        ----------
        defl : array
            The deflection matrix. If None, all deflections will be
            assumed to be zero.
        """
        n = len(self.build['nodes'])
        k = np.zeros((6*n, 6*n), dtype='float')
        ndict = self.build['ndict']

        if defl is None:
            defl = np.zeros(6*n)

        for e in self.elements:
            i, j = ndict[e.inode], ndict[e.jnode]
            di, dj = defl[i:i+3], defl[j:j+3]

            ke = e.global_stiffness(di, dj)
            k[i:i+6,i:i+6] += ke[:6,:6]
            k[i:i+6,j:j+6] += ke[:6,6:12]
            k[j:j+6,i:i+6] += ke[6:12,:6]
            k[j:j+6,j:j+6] += ke[6:12,6:12]

        return k

    @build
    def global_node_loads(self, lc):
        """
        Returns the global node load matrix for the input load case.

        Parameters
        ----------
        lc : :class:`.LoadCase`
            The applied load case.
        """
        n = len(self.build['nodes'])
        q = np.zeros(6*n, dtype='float')
        ndict = self.build['ndict']

        for n in lc.node_loads:
            i = ndict[n.node]
            q[i:i+6] += n.forces()

        return q

    @build
    def local_elem_loads(self, lc, defl=None):
        """
        Returns the local element loads for the input load case.

        Parameters
        ----------
        lc : :class:`.LoadCase`
            The applied load case.
        defl : array
            The global node deflections.
        """
        n = len(self.build['nodes'])
        m = len(self.build['elements'])
        q = np.zeros((m, 12), dtype='float')
        ndict = self.build['ndict']
        edict = self.build['edict']

        if defl is None:
            defl = np.zeros(6*n)

        for e in lc.elem_loads:
            ref = e.get_element()
            i, j, k = ndict[ref.inode], ndict[ref.jnode], edict[ref.name]
            di, dj = defl[i:i+3], defl[j:j+3]
            q[k] += e.local_reactions(di, dj)

        return q

    @build
    def global_elem_loads(self, lc, defl=None):
        """
        Returns the global node load matrix for the input load case.

        Parameters
        ----------
        lc : :class:`.LoadCase`
            The applied load case.
        defl : array
            The global node deflections.
        """
        n = len(self.build['nodes'])
        q = np.zeros(6*n, dtype='float')
        ndict = self.build['ndict']

        if defl is None:
            defl = np.zeros(6*n)

        for e in lc.elem_loads:
            ref = e.get_element()
            i, j = ndict[ref.inode], ndict[ref.jnode]
            di, dj = defl[i:i+3], defl[j:j+3]

            f = e.global_reactions(di, dj)
            q[i:i+6] += f[:6]
            q[j:j+6] += f[6:12]

        return q

    @build
    def global_defl(self, lc):
        """
        Returns the global applied deflection matrix for the input load case.

        Parameters
        ----------
        lc : :class:`.LoadCase`
            The applied load case.
        """
        n = len(self.build['nodes'])
        d = np.zeros(6*n, dtype='float')
        ndict = self.build['ndict']

        for n in lc.node_loads:
            i = ndict[n.node]
            d[i:i+6] += n.deflections()

        return d

    def _create_summary(self, r):
        """
        Creates dataframe summaries for the structural analysis results.

        Parameters
        ----------
        r : dict
            A dictionary of result arrays.
        """
        n = len(self.build['nodes'])
        m = len(self.build['elements'])
        lc = self.build['load_cases']
        u = [x.fixities() for x in self.nodes] * len(lc)
        u = np.array(u, dtype='bool')

        # Global load data frame
        df1 = pd.DataFrame()
        df1['load_case'] = np.array([[l.name] * n for l in lc]).ravel()
        df1['node'] = [x.name for x in self.build['nodes']] * len(lc)

        # Process global forces
        x = np.array(r.pop('glob_force')).reshape(-1, 6)
        x[np.abs(x) < 1e-8] = 0
        df1['force_x'] = x[:,0]
        df1['force_y'] = x[:,1]
        df1['force_z'] = x[:,2]
        df1['moment_x'] = x[:,3]
        df1['moment_y'] = x[:,4]
        df1['moment_z'] = x[:,5]

        # Process global deflections
        x = np.array(r.pop('glob_defl')).reshape(-1, 6)
        x[np.abs(x) < 1e-8] = 0
        df1['defl_x'] = x[:,0]
        df1['defl_y'] = x[:,1]
        df1['defl_z'] = x[:,2]
        df1['rot_x'] = x[:,3]
        df1['rot_y'] = x[:,4]
        df1['rot_z'] = x[:,5]

        # Global reaction data frame
        df2 = df1.copy()
        del df2['defl_x'], df2['defl_y'], df2['defl_z']
        del df2['rot_x'], df2['rot_y'], df2['rot_z']

        df2.loc[u[:,0], 'force_x'] = np.nan
        df2.loc[u[:,1], 'force_y'] = np.nan
        df2.loc[u[:,2], 'force_z'] = np.nan
        df2.loc[u[:,3], 'moment_x'] = np.nan
        df2.loc[u[:,4], 'moment_y'] = np.nan
        df2.loc[u[:,5], 'moment_z'] = np.nan

        df2 = df2[~u.all(axis=1)].copy()
        df2 = df2.reset_index(drop=True)

        # Local reaction data frame
        df3 = pd.DataFrame()
        df3['load_case'] = np.array([[l.name] * m for l in lc]).ravel()
        df3['element'] = [x.name for x in self.build['elements']] * len(lc)

        # Process local forces
        x = np.array(r.pop('loc_force')).reshape(-1, 12)
        x[np.abs(x) < 1e-8] = 0

        df3['i_axial'] = x[:,0]
        df3['i_shear_x'] = x[:,1]
        df3['i_shear_y'] = x[:,2]
        df3['i_torsion'] = x[:,3]
        df3['i_moment_x'] = x[:,4]
        df3['i_moment_y'] = x[:,5]

        df3['j_axial'] = x[:,6]
        df3['j_shear_x'] = x[:,7]
        df3['j_shear_y'] = x[:,8]
        df3['j_torsion'] = x[:,9]
        df3['j_moment_x'] = x[:,10]
        df3['j_moment_y'] = x[:,11]

        # Process local deflections
        x = np.array(r.pop('loc_defl')).reshape(-1, 12)
        x[np.abs(x) < 1e-8] = 0

        df3['i_defl_ax'] = x[:,0]
        df3['i_defl_x'] = x[:,1]
        df3['i_defl_y'] = x[:,2]
        df3['i_twist'] = x[:,3]
        df3['i_rot_x'] = x[:,4]
        df3['i_rot_y'] = x[:,5]

        df3['j_defl_ax'] = x[:,6]
        df3['j_defl_x'] = x[:,7]
        df3['j_defl_y'] = x[:,8]
        df3['j_twist'] = x[:,9]
        df3['j_rot_x'] = x[:,10]
        df3['j_rot_y'] = x[:,11]

        return dict(glob=df1, react=df2, loc=df3)

    @build
    def linear_analysis(self, lc):
        """
        Performs linear analysis on the structure.

        Parameters
        ----------
        lc : :class:`.LoadCase` or list
            A load case or list of load cases to perform analysis for.
        """
        n = len(self.build['nodes'])
        k = self.global_stiffness()
        ndict = self.build['ndict']

        # Result dictionary
        r = dict(glob_force=[], glob_defl=[], loc_force=[], loc_defl=[])

        # Determine free and nonzero matrix rows and columns
        u = np.array([x.fixities() for x in self.nodes], dtype='bool').ravel()

        if not u.any():
            raise ValueError('No node fixities found.')

        u &= k.any(axis=1)
        v = ~u

        # Calculate inverse and create unknown-known stiffness partition
        ki = np.linalg.inv(k[u][:,u])
        kuv = k[u][:,v]

        for l in self.build['load_cases']:
            # Find unknown deflections and global forces
            d = self.global_defl(l)
            q = self.global_node_loads(l)
            qe = self.global_elem_loads(l)
            q -= qe
            d[u] = ki.dot(q[u] - kuv.dot(d[v]))
            q = k.dot(d) + qe

            # Add to results dictionary
            r['glob_force'].append(q)
            r['glob_defl'].append(d)

            # Find local forces
            q = self.local_elem_loads(l)

            for m, e in enumerate(self.build['elements']):
                i, j = ndict[e.inode], ndict[e.jnode]
                dl = np.array([d[i:i+6], d[j:j+6]]).ravel()
                dl = e.transformation_matrix().dot(dl)
                q[m] += e.local_stiffness().dot(dl)
                r['loc_defl'].append(dl)

            r['loc_force'].append(q)

        # Destory obsolete objects
        del k, ki, kuv, u, v, d, q

        return self._create_summary(r)
