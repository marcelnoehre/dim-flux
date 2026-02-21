import networkx as nx

from typing import Tuple, Set, Dict
from collections import deque 
from fcapy.lattice import ConceptLattice

def cover_relations(concept_lattice: ConceptLattice) -> Set[Tuple[int, int]]:
    '''
    Get the cover relations of a concept lattice.

    Parameters
    ----------
    concept_lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    cover_relations : Set[Tuple[int, int]]
        A set of tuples representing the cover relations of the lattice.
    '''
    return set(nx.transitive_reduction(concept_lattice.to_networkx()).edges)

def all_extents(
        lattice: ConceptLattice
    ) -> Dict[int, Set[str]]:
    '''
    Compute the extents for all concepts in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice

    Returns
    -------
    extents: Dict[int, Set[str]]
        A dictionary mapping concept IDs to their full extents
    '''
    seen = set({})
    queue = deque({len(lattice.to_networkx().nodes)-1})
    extents = dict({})

    while queue:
        concept = queue.popleft()
        
        # all childrens processed?
        if lattice.children(concept) <= extents.keys():
            # A_new \cup A_children
            extents[concept] = lattice.get_concept_new_extent(concept).union(
                *(extents[c] for c in lattice.children(concept))
            )

            # add parents to queue
            seen.update(lattice.parents(concept) - extents.keys())
            queue.extend(lattice.parents(concept) - extents.keys())

        # readd to queue
        else:
            queue.append(concept)

    return extents

def all_intents(
        lattice: ConceptLattice
    ) -> Dict[int, Set[str]]:
    '''
    Compute the intents for all concepts in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice

    Returns
    -------
    intents: Dict[int, Set[str]]
        A dictionary mapping concept IDs to their full intents
    '''
    seen = set({})
    queue = deque({0})
    intents = dict({})

    while queue:
        concept = queue.popleft()
        
        # all parents processed?
        if lattice.parents(concept) <= intents.keys():
            # B_new \cup B_parents
            intents[concept] = lattice.get_concept_new_intent(concept).union(
                *(intents[p] for p in lattice.parents(concept))
            )

            # add children to queue
            seen.update(lattice.children(concept) - intents.keys())
            queue.extend(lattice.children(concept) - intents.keys())

        # readd to queue
        else:
            queue.append(concept)

    return intents

def incomparability_graph(lattice: ConceptLattice) -> nx.Graph:
    '''
    Get the incomparability graph of a concept lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.
    
    Returns
    -------
    incomparability_graph : nx.Graph
        The incomparability graph of the lattice.
    '''
    return nx.complement(nx.transitive_closure(lattice.to_networkx()).to_undirected())
