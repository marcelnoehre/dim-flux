import numpy as np
from pathlib import Path

from odis import FormalContext

from src.fca.lattice import compute_lectic_order
from src.utils.variables import Variables

class DimDraw():
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
        '''
        Initialize DimDraw with a given 'realizer'.

        Parameters
        ----------
        variables: Variables
            The storage of variables
        '''
        self.vars = variables
        self.concepts = {
            c: self.vars.extents[c] | self.vars.intents[c]
            for c in self.vars.concepts
        }

    def two_dimensional_extension(self):
        '''
        Compute the two-dimensional extension of the lattice.

        Returns
        -------
        coordinates : Dict[int, List]
            Original DimDraw coordinates.
        '''
        if self.vars.cxt.endswith('.cxt'):
            cxt_path = Path(self.vars.cxt).resolve()
        else:
            cxt_path = Path(f'data/{self.vars.cxt}.cxt').resolve()

        ctx = FormalContext.from_file(str(cxt_path))
        drawing = ctx.draw("dimdraw")
        self.lectic_order = compute_lectic_order(self.vars)
        self.coordinates = {
            c: (np.array([drawing.nodes[i].x, drawing.nodes[i].y]) * -1 * np.array([np.sqrt(2), 1/np.sqrt(2)])).tolist()
            for i, c in enumerate(self.lectic_order)
        }
        return self.coordinates