import itertools
import logging
from typing import Iterable


def is_odd(x):
    "ha. ha."
    return x % 2 == 1


def odd_parity(tensor_orders: tuple[int, ...]) -> bool:
    """Check if sum of ranks is odd, which means that there is no way to connect them without dangling edge.

    Args:
        tensor_orders (tuple[int, ...]): Representation of the node ranks in the contraction.

    Returns:
        bool: True if sum of ranks is odd, False otherwise.
    """
    return is_odd(sum(tensor_orders))


def iter_tensor_terms(max_tensor_order, poly_order):
    # NOTE: This is not used anywhere, but it is a useful utility function for generating tensor terms.
    logging.warning("iter_tensor_terms is unmaintained and may be removed in future versions.")
    tensor_orders = list(range(1, max_tensor_order + 1))

    for porder in range(2, poly_order + 1):
        for tensor_order_set in itertools.combinations_with_replacement(tensor_orders, r=porder):
            if odd_parity(tensor_order_set):
                continue
            yield tensor_order_set
    return


def iter_homogenous_terms(order: int,
                          max_order: int=4) -> Iterable[tuple[int, ...]]:
    """ Generator for the homogenous terms, it returns list of nodes with same rank, sum of ranks has to even,
      because otherwise there is no way how to connect them without dangling edge.

    Args:
        order (int): Order of the rank in the contraction
        max_order (int, optional): Maximum number of elements in the contraction Defaults to 4.

    Yields:
        Iterable[tuple[int, ...]]: List of nodes with same rank, sum of ranks has to even.
    """
    # TODO: Change the names, order refers to l and max_order refers to the maximum number of tensors in the term.
    i = 2
    term = [order]
    while True:
        candidate = tuple(term * i)
        if not odd_parity(candidate):
            yield candidate
        if i == max_order:
            return
        i += 1


def iter_mixed_terms(o1: int, o2: int, max_order: int=4) -> Iterable[tuple[int, ...]]:
    """ Generator for the mixed terms, it returns list of nodes with their rank, sum of ranks has to even,
      because otherwise there is no way how to connect them without dangling edge.

    Args:
        o1 (int): First tensor order
        o2 (int): Second tensor order
        max_order (int, optional): Maximum number of elements in the contraction Defaults to 4.

    Yields:
        Iterable[tuple[int, ...]]: (o1, o1, o2, o2) for example, where o1 < o2 and the sum of the orders is even.
    """
    o1, o2 = min(o1, o2), max(o1, o2)

    total_order = 2
    while True:
        for j in range(total_order - 1, 0, -1):
            candidate = tuple([o1] * j + [o2] * (total_order - j))
            if not odd_parity(candidate):
                yield candidate
        if total_order == max_order:
            return
        total_order += 1

