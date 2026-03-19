from z3 import *
from fcapy.lattice import ConceptLattice
from fcapy.context import FormalContext

from src.fca.lattice import *

class AdditiveRealizer:
    '''
    Compute an additive realizer for a given concept lattice.

    Parameters
    ----------
    context : FormalContext
        The formal context for which to compute the additive realizer.
    '''
    def __init__(self, context: FormalContext):
        self.context = context
        self.lattice = ConceptLattice.from_context(context)
        self.G = self.lattice.to_networkx()
        self.relations = self.G.edges
        self.concepts = self.G.nodes
        self.top = len(self.concepts)-1
        self.objects = set(context._object_names)
        self.features = set(context._attribute_names)
        self.incomparable_pairs = incomparability_graph(self.lattice).edges
        self.extents = all_extents(self.lattice)
        self.intents = all_intents(self.lattice)
        self.base_vectors = dict({})
        
        self.solver = Solver()
        self.dimension = 2
        self.dimensions = ['x', 'y']

        self._setup_smt_variables()
        self._setup_relations()

    def realizer(self):
        '''
        Compute an additive realizer using the z3 SMT solver.

        Returns
        -------
        realizer : List
            2D additive realizer or empty list if non exists

        '''
        if self.solver.check() == sat:
            # solved clauses
            self.model = self.solver.model()
            # prepare empty realizer
            realizer = {
                d: [None for _ in self.concepts]
                for d in self.dimensions
            }
            # derive base vectors
            for g in self.objects:
                self.base_vectors[g] = [float(self.model[Real(f'x_{g}')].as_fraction()), float(self.model[Real(f'y_{g}')].as_fraction())]
            for m in self.features:
                self.base_vectors[m] = [float(-self.model[Real(f'x_{m}')].as_fraction()), float(-self.model[Real(f'y_{m}')].as_fraction())]

            # insert concepts based on their vector sum
            for d in self.dimensions:
                for concept in self.concepts:
                    realizer[d][self.model[Int(f'{d}_{concept}')].as_long()] = concept

            return [le for le in realizer.values()]
        else:
            return []      
        
    def _setup_smt_variables(self):
        '''
        Define SMT variables for all concepts and base vectors.
        '''
        # base vectors
        self.smt_variables = {
            (d, v): Real(f'{d}_{v}')
            for d in self.dimensions
            for v in self.features.union(self.objects)
        }
        # concept = sum of base vectors
        # (A, B) -> sum(A) - sum(B)
        for d in self.dimensions:
            for concept in self.concepts:
                vec_G = (self.smt_variables[d, var] for var in self.extents[concept])
                vec_M = (self.smt_variables[d, var] for var in (self.features - self.intents[concept]))
                self.solver.add(Int(f'{d}_{concept}') == sum(vec_G) + sum(vec_M))

    def _setup_relations(self):
        '''
        Define SMT clauses for additivity.
        '''
        # related pairs: a > b
        for a, b in self.relations:
            for d in self.dimensions:
                # if > then the vector sum has to be >
                self.solver.add(Int(f'{d}_{a}') > Int(f'{d}_{b}'))

        # incomparable pairs
        for a, b in self.incomparable_pairs:
            a_vars = [Int(f'{d}_{a}') for d in self.dimensions]
            b_vars = [Int(f'{d}_{b}') for d in self.dimensions]
            # at least one extension has a < b
            a_lt_b = [a_vars[i] < b_vars[i] for i in range(self.dimension)]
            # at least one extension has a > b
            a_gt_b = [a_vars[i] > b_vars[i] for i in range(self.dimension)]
            self.solver.add(And(Or(*a_lt_b), Or(*a_gt_b)))
            # a != b in the same dimension
            for d in self.dimensions:
                self.solver.add(Int(f'{d}_{a}') != Int(f'{d}_{b}'))

        # Fix bottom and top to define range
        for d in self.dimensions:
            self.solver.add(Int(f'{d}_{len(self.concepts)-1}') == 0)
            self.solver.add(Int(f'{d}_0') == self.top)