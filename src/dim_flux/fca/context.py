from typing import Set, Tuple
from fcapy.context import FormalContext

def object_concept(
        context: FormalContext,
        g: str
    ) -> Tuple[Set[str], Set[str]]:
    '''
    Compute the object concept of g.

    Parameters
    ----------
    formal_context : FormalContext
        The formal context
    g : str
        The object

    Returns
    -------
    concept: Tuple[Set[str], Set[str]]
        The object concept
    '''
    return (set(context.extension(context.intention({g}))), set(context.intention({g})))

def attribute_concept(
        context: FormalContext,
        m: str
    ) -> Tuple[Set[str], Set[str]]:
    '''
    Compute the attribute concept of m.

    Parameters
    ----------
    context : FormalContext
        The formal context
    m : str
        The attribute

    Returns
    -------
    concept: Tuple[Set[str], Set[str]]
        The attribute concept
    '''
    return (set(context.extension({m})), set(context.intention(context.extension({m}))))

def object_closure(
        context: FormalContext,
        objects: Set[str]
    ) -> Set[str]:
    '''
    Compute the closure of a set of objects.

    Parameters
    ----------
    context : FormalContext
        The formal context
    objects : Set[str]
        The set of objects

    Returns
    -------
    closure: Set[str]
        The double-primed set of objects
    '''
    return set(context.extension(context.intention(objects)))

def attribute_closure(
        context: FormalContext,
        attributes: Set[str]
    ) -> Set[str]:
    '''
    Compute the closure of a set of attributes.

    Parameters
    ----------
    context : FormalContext
        The formal context
    attributes : Set[str]
        The set of attributes

    Returns
    -------
    closure: Set[str]
        The double-primed set of attributes
    '''
    return set(context.intention(context.extension(attributes)))
