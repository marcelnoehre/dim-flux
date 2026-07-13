import argparse
import numpy as np
import pandas as pd

from typing import Optional, Union
from dim_flux.utils.visualize import *
from dim_flux.utils.variables import Variables
from dim_flux.core.realizer import Realizer
from dim_flux.fdp.sup_inf import SupInfGraph
from dim_flux.fdp.init_layout import InitLayout
from dim_flux.fdp.forces import ForceDirectedPlacement

def run(cxt: Union[str, pd.DataFrame], export: bool = False) -> Optional[np.ndarray]:
    '''
    Run the DimFlux pipeline end to end.

    Parameters
    ----------
    cxt : str or pandas.DataFrame
        A path to a .cxt file, or a pandas DataFrame representing the incidence matrix.
    export : bool
        If True, write the PDF, GraphML and .pos exports to disk and return None.
        If False, skip all exports and return the computed positions instead.

    Returns
    -------
    positions : Optional[np.ndarray]
        The (x, y) coordinates of each concept in lectic order, or None if export is True.
    '''
    vars = Variables(cxt, {
        'plot_si_graph': False,
        'si_graph_annotations':  False,
        'plot_initial_layout':  False,
        'initial_layout_annotations':  False,
        'plot_optimized_layout':  False,
        'optimized_layout_annotations':  False,
        'plot_individual_forces':  False,
        'plot_combined_forces':  False,
        'plot_gradients':  False,
        'plot_origin':  False
    })

    mode = 'DimFlux' # 'PlanarityEnhancer'

    if mode == 'DimFlux':
        realizer = Realizer(vars)
        vars.base_vectors = realizer.base_vectors
        vars.coordinates = realizer.coordinates
        vars.lectic_order = realizer.lectic_order

        if vars.args.plot_initial_layout:
            plot_lattice(vars, 'Initial layout (Projected DimDraw)', vars.args.initial_layout_annotations, False)

    else:
        # Sup-Inf Graph
        sup_inf_graph = SupInfGraph(vars)
        vars.scalars = sup_inf_graph.scalars
        vars.order = sup_inf_graph.order
        vars.d_si_points = sup_inf_graph.d_si_points
        vars.n_1 = sup_inf_graph.n_1
        vars.n_2 = sup_inf_graph.n_2

        if vars.args.plot_si_graph:
            plot_si_graph(vars)

        # Initial Layout
        initial_layout = InitLayout(vars)
        vars.base_vectors = initial_layout.base_vectors
        vars.coordinates = initial_layout.coordinates

        if vars.args.plot_initial_layout:
            plot_lattice(vars, 'Initial layout', vars.args.initial_layout_annotations, False)

    # Optimize Layout
    forces = ForceDirectedPlacement(vars)
    vars.coordinates = forces.coordinates
    vars.final_forces = forces.final_forces

    if vars.args.plot_optimized_layout:
        plot_lattice(
            vars, 'Optimized layout', vars.args.optimized_layout_annotations,
            vars.args.plot_individual_forces or vars.args.plot_combined_forces or vars.args.plot_gradients
        )

    if export:
        pdf_export(vars, 'm4')
        graphml_export(vars, 'm4')
        pos_export(vars, vars.cxt)

    return np.array([vars.coordinates[c] for c in vars.lectic_order])


def main():
    parser = argparse.ArgumentParser(description='DimFlux: Force-Based Doubly-Additive Drawings')
    parser.add_argument(
        '--export', action='store_true',
        help='Write the PDF, GraphML and .pos exports to disk instead of printing the positions'
    )
    args = parser.parse_args()

    cxt = input('Path to .cxt file: ')
    positions = run(cxt, export=args.export)
    if positions is not None:
        print(positions)


if __name__ == "__main__":
    main()
