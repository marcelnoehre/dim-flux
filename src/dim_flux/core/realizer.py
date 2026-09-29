import numpy as np

from odis import FormalContext

from dim_flux.utils.variables import Variables
from dim_flux.fca.lattice import compute_lectic_order
from dim_flux.core.projection import Projection
from dim_flux.core.lgs import LinearEquationSolver

class Realizer():
    '''
    Reference
    ---------
    @misc{dürrschnabel2019dimdrawnoveltool,
        title={DimDraw -- A novel tool for drawing concept lattices},
        author={Dominik Dürrschnabel and Tom Hanika and Gerd Stumme},
        year={2019},
        eprint={1903.00686},
        archivePrefix={arXiv},
        primaryClass={cs.CG},
        url={https://arxiv.org/abs/1903.00686}
    }
    '''
    def __init__(self,
            variables: Variables
        ):
        self.vars = variables
        self.context = variables.context
        
        self.coordinates = self.two_dimensional_extension()
        self.vars.coordinates = self.coordinates
        projection = Projection(self.vars)
        self.vars.coordinates = projection.coordinates
        self._derive_base_vectors()
        self.lectic_order = projection.lectic_order

    def two_dimensional_extension(self):
        '''
        Compute the two-dimensional extension of the lattice.

        Returns
        -------
        coordinates : Dict[int, List]
            Original DimDraw coordinates.
        '''
        # reduced context on the irreducible elements, indexed names keep odis independent of the labels
        ctx = FormalContext()
        for m in self.vars.attributes:
            ctx.add_attribute(f'm_{self.vars.attribute_map[m]}')
        for g in self.vars.objects:
            g_intent = self.vars.M & set(self.context.intention([g]))
            ctx.add_object(f'g_{self.vars.object_map[g]}', [f'm_{self.vars.attribute_map[m]}' for m in g_intent])
        drawing = ctx.draw("dimdraw", timeout_ms=self.vars.timeout_ms)
        self.lectic_order = compute_lectic_order(self.vars)
        self.coordinates = {
            c: (np.array([drawing.nodes[i].x, drawing.nodes[i].y]) * -1 * np.array([np.sqrt(2), 1/np.sqrt(2)])).tolist()
            for i, c in enumerate(self.lectic_order)
        }
        return self.coordinates

    def _derive_base_vectors(self):
        '''
        Derive base vectors by solving the system of linear equations
        '''
        lgs = LinearEquationSolver(self.vars, self.coordinates)
        success, vector_vars = lgs.solve_linear_equations()
        if success:
            self.base_vectors = dict({})
            for v in self.vars.elements:
                if v in self.vars.G:
                    self.base_vectors[v] = np.array([vector_vars[lgs.symbol('x', v)], vector_vars[lgs.symbol('y', v)]])
                else:
                    self.base_vectors[v] = np.array([-vector_vars[lgs.symbol('x', v)], -vector_vars[lgs.symbol('y', v)]])