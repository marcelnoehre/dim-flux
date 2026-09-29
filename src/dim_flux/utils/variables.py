import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

from dim_flux.fca.context import *
from dim_flux.fca.lattice import *
from dim_flux.utils.parser import decode_cxt

@dataclass
class Args:
    plot_si_graph: bool = False
    si_graph_annotations: bool = False
    plot_initial_layout: bool = False
    initial_layout_annotations: bool = False
    plot_optimized_layout: bool = False
    optimized_layout_annotations: bool = False
    plot_individual_forces: bool = False
    plot_combined_forces: bool = False
    plot_gradients: bool = False
    plot_origin: bool = False

class Variables():
    '''
    A container class to store the formal context, concept lattice, 
    and parameters required for lattice drawing and force-directed layouts.

    Parameters
    ----------
    cxt : str or pandas.DataFrame
        A path to a .cxt file, or a pandas DataFrame representing the incidence matrix
        (objects as rows, attributes as columns). A string not ending in `.cxt` raises
        a ValueError.
    context : FormalContext
        The formal context decoded from the input file
    lattice : ConceptLattice
        The concept lattice derived from the context
    args : Args
        Configuration object for visualization settings
    join_irreducibles : Dict[int, str]
        Mapping of join-irreducible concept IDs to their representative object
    meet_irreducibles : Dict[int, str]
        Mapping of meet-irreducible concept IDs to their representative attribute
    objects : List[str]
        List of irreducible objects (one representative per join-irreducible concept)
    object_map : Dict[str, int]
        Mapping of irreducible object names to their integer indices
    object_closures : Dict[str, Set[str]]
        Mapping of irreducible objects to their closure sets restricted to G
    N_g : int
        The number of irreducible objects
    G : Set[str]
        The set of irreducible object names
    attributes : List[str]
        List of irreducible attributes (one representative per meet-irreducible concept)
    attribute_map : Dict[str, int]
        Mapping of irreducible attribute names to their integer indices
    attribute_closures : Dict[str, Set[str]]
        Mapping of irreducible attributes to their closure sets restricted to M
    N_m : int
        The number of irreducible attributes
    M : Set[str]
        The set of irreducible attribute names
    elements : List[str]
        Concatenated list of irreducible objects and attributes, the vectors the forces act on
    element_map : Dict[str, int]
        Mapping of element names to their integer indices
    N_e : int
        Total number of elements (irreducible objects + attributes)
    E : Set[str]
        The set of all elements
    concepts : List[int]
        List of concept IDs from the lattice
    N_c : int
        Total number of concepts in the lattice
    full_extents : Dict[int, Set[str]]
        Mapping of concept IDs to their extents in the original context
    full_intents : Dict[int, Set[str]]
        Mapping of concept IDs to their intents in the original context
    extents : Dict[int, Set[str]]
        Mapping of concept IDs to their extents restricted to G
    intents : Dict[int, Set[str]]
        Mapping of concept IDs to their intents restricted to M
    atoms : List[str]
        Objects belonging to concepts directly above the bottom concept
    coatoms : List[str]
        Attributes belonging to concepts directly below the top concept
    w_rep : float
        Repulsive force weight
    w_att : float
        Attractive force weight
    w_grav : float
        Gravitational force weight
    timeout_ms : Optional[int]
        Timeout in milliseconds for the DimDraw layout search and, as a
        fallback, for the force-directed optimization. If no proven/optimal
        result is found within this duration, the best result found so far
        is kept. If None, both searches run to a proven/converged optimum,
        which can take very long on a large lattice
    order : List
        The processing order for layout optimization
    scalars : np.ndarray
        Array of scalar values associated with each vector
    d_si_points : List
        Points used for sup inf graph calculations
    n_1 : int
        Left point of f_max
    n_2 : int
        Right point of f_max
    base_vectors : Dict
        Dictionary storing the base vectors for elements
    coordinates : Dict
        Dictionary mapping concepts to their computed coordinates
    final_forces : Dict
        Dictionary storing the resultant forces after optimization
    '''

    def __init__(
        self, cxt: Union[str, pd.DataFrame], args: Optional[Dict[str, bool]],
        w_rep: float = 100.0, w_att: float = 1.0, w_grav: float = 30.0,
        timeout_ms: Optional[int] = 1000
    ):

        if isinstance(cxt, pd.DataFrame):
            self.cxt = 'input'
            self.context = FormalContext.from_pandas(cxt)
        else:
            if not cxt.endswith('.cxt'):
                raise ValueError(f'Expected a path to a .cxt file, got: {cxt}')
            self.context = decode_cxt(cxt)
            self.cxt = Path(cxt).stem

        self.lattice = ConceptLattice.from_context(self.context)
        self.args: Args = Args(**(args or {}))

        # irreducible representatives, forces act on these instead of all objects and attributes
        self.join_irreducibles, self.meet_irreducibles = irreducible_representatives(
            self.lattice, self.context.object_names, self.context.attribute_names
        )
        irreducible_objects = set(self.join_irreducibles.values())
        irreducible_attributes = set(self.meet_irreducibles.values())

        # objects
        self.objects = [g for g in self.context.object_names if g in irreducible_objects]
        self.object_map = {g: i for i, g in enumerate(self.objects)}
        self.N_g = len(self.objects)
        self.G = set(self.objects)
        self.object_closures = {
            g: object_closure(self.context, {g}) & self.G
            for g in self.objects
        }

        # attributes
        self.attributes = [m for m in self.context.attribute_names if m in irreducible_attributes]
        self.attribute_map = {m: i for i, m in enumerate(self.attributes)}
        self.N_m = len(self.attributes)
        self.M = set(self.attributes)
        self.attribute_closures = {
            m: attribute_closure(self.context, {m}) & self.M
            for m in self.attributes
        }

        # elements
        self.elements = self.objects + self.attributes
        self.element_map = {
            v: i
            for i, v in enumerate(self.elements)
        }
        self.N_e = self.N_g + self.N_m
        self.E = set(self.elements)

        # concepts
        self.concepts = self.lattice.to_networkx().nodes
        self.N_c = len(self.concepts)
        self.full_extents = all_extents(self.lattice)
        self.full_intents = all_intents(self.lattice)
        self.extents = {c: e & self.G for c, e in self.full_extents.items()}
        self.intents = {c: i & self.M for c, i in self.full_intents.items()}
        self.atoms = [
            self.join_irreducibles[c]
            for c in self.lattice.parents(self.N_c - 1)
        ]
        self.coatoms = [
            self.meet_irreducibles[c]
            for c in self.lattice.children(0)
        ]

        # weights
        self.w_rep = w_rep
        self.w_att = w_att
        self.w_grav = w_grav
        self.timeout_ms = timeout_ms

        # global variables
        self.order = []
        self.scalars = np.zeros(self.N_e)        
        self.d_si_points = []
        self.n_1 = 0
        self.n_2 = 0
        self.base_vectors = dict({})
        self.coordinates = dict({})
        self.final_forces = dict({})
        self.lectic_order = []
