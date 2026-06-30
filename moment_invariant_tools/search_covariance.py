

import torch
import logging
import numpy as np
from itertools import product
from collections.abc import Iterable


from moment_invariant_tools.search_invariants import build_grad_row, setup_search
from moment_invariant_tools.graph_check import GraphFilter
from moment_invariant_tools.construct_terms import odd_parity
from moment_invariant_tools.contraction_formats import formula_to_einsum
from moment_invariant_tools.construct_contractions import iter_indices


def perform_covariant_grad(ranks: tuple[int, ...],
                           contraction: tuple[tuple[int, ...], ...],
                           tensor_set: dict[int, torch.Tensor],
                           mapper_set: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """ Perform the contraction defined in `contraction` on the tensors defined in `tensor_set` (random values), `mapper_set` (index mappings),
        and compute the gradient of the output with respect to the input tensors.

    Args:
        ranks (tuple[int, ...]): Tuple of ranks corresponding to the input tensors.
        contraction (tuple[tuple[int, ...], ...]): Contraction defined as a tuple of tuples, where each inner tuple contains the indices for the corresponding tensor in `tensor_set`.
        tensor_set (dict[int, torch.Tensor]): Dictionary mapping tensor ranks to their corresponding tensors.
        mapper_set (dict[int, torch.Tensor]): Dictionary mapping tensor ranks to their corresponding index mappings.

    Returns:
        dict[int, torch.Tensor]: Dictionary mapping tensor ranks to their corresponding gradients.
    """
    input_tensors = [mapper_set[r] @ tensor_set[r] for r in ranks]

    # Enumerate the contraction to get the correct indices for the input tensors.
    einsum = formula_to_einsum(tuple((ranks, contraction)))
    outs = torch.einsum(einsum, *input_tensors)
    if outs.ndim == 0:
        outs = outs.unsqueeze(0)

    # Contraction grad dict 
    grads_list = []
    for out_idx in range(outs.shape[0]):
        logging.info(f"Computing gradient for output index {out_idx} of {outs.shape[0]}")
        grad_values = torch.autograd.grad(outs[out_idx], 
                                tensor_set.values(),
                                allow_unused=True,
                                retain_graph=True) 
        grad_dict = {k: v for k, v in zip(tensor_set.keys(), grad_values)}
        # Substitute none for zeros.
        for k, v in grad_dict.items():
            if v is None:
                grad_dict[k] = torch.zeros_like(tensor_set[k])
        grads_list.append(grad_dict)
    return grads_list


def test_covariants(grads_list: dict[int, torch.Tensor],
                    grad_matrix: torch.Tensor,
                    rank_set: list[int]) -> tuple[int, torch.Tensor]:
    """ Test the invariants

    Args:
        grads_list (dict[int, torch.Tensor]): List of gradients of the current contraction with respect to the input tensors, keyed by tensor rank.
        grad_matrix (torch.Tensor): The gradient matrix.
        rank_set (list[int]): A list of tensor ranks involved in the contraction.

    Returns:
        tuple[int, torch.Tensor]: The number of independent invariants found and the new gradient rows.
    """
    # Construct the gradient matrix from the grad map and try to add the new gradient row. If the rank increases, we found a new independent invariant!
    new_rows = torch.stack([build_grad_row(grads, rank_set) for grads in grads_list])
    grad_matrix =  new_rows if grad_matrix is None else torch.cat([grad_matrix, new_rows], dim=0)
    tol = None
    total_invariants = torch.linalg.matrix_rank(grad_matrix, atol=tol, rtol=tol).item()
    # Return the rank of the new gradient matrix, which is the total number of independent invariants found so far (including the new one).
    return total_invariants, new_rows


def remove_artificial_node(contraction):
    contraction = contraction[1:]
    for i in range(len(contraction)):
        if 0 in contraction[i]:
            return contraction[:i] + (tuple((j for j in contraction[i] if j != 0)),) + contraction[i+1:] 
    raise ValueError("Artificial node not found in contraction. This should not happen.")


def iter_homogenous_covariant_terms(l: int, type: int = 1, max_n: int = 4) -> Iterable[tuple[int, ...]]:

    """ Generator that returns homogeneuous with type dangling edges. 

    Args:
        o1 (int): First tensor order

    Yields: 
        Iterable[tuple[int, ...]]: (o1, o1, o1, type) for example,  sum of all including the type is even.
    """
    term = [l]
    assert (l > 1) and l % 2 == 1, "l has to be odd and greater than 1" 
    for _ in range(1, max_n):
        term.append(l)
        if type > l: 
            candidate = tuple(term + [type])
        else: 
            candidate = tuple([type] + term)
        if not odd_parity(candidate):
            yield candidate


def iter_heterogeneous_covariant_terms(o1: int, o2: int, n_max: int=4, type: int=1) -> Iterable[tuple[int, ...]]:
    """ Generator for the mixed terms, it returns list of nodes with their rank, sum of ranks has to be even,
      because otherwise there is no way how to connect them without dangling edge.

    Args:
        o1 (int): First tensor order
        o2 (int): Second tensor order
        n_max (int): Maximum number of elements in the contraction Defaults to 4.

    Yields:
        Iterable[tuple[int, ...]]: (o1, o1, o2, o2) for example, where o1 < o2 and the sum of the orders is even.
    """
    assert type == 1, "Only type 1 is supported for now."
    assert o1 != o2, "o1 and o2 must be different."
    o1, o2 = min(o1, o2), max(o1, o2)

    if (o1 + o2) % 2 != 1:
        logging.warning("Sum of tensor orders is odd, no valid contraction possible %s, %s.", o1, o2)

    total_order = 2
    while True:
        for j in range(total_order - 1, 0, -1):
            candidate = tuple([o1] * j + [o2] * (total_order - j))
            if odd_parity(candidate):
                yield candidate
        if total_order == n_max:
            return
        total_order += 1


def perform_search(l_max: int,
                   n_max: int,
                   spherical:bool = True):
    raise NotImplementedError("This function is not yet implemented. Please use the search_homogenous_covariants function instead.")
    rank_set = list(range(1, l_max + 1, 2))
    grad_map, mapper_set, tensor_set = setup_search(rank_set, spherical=spherical)
    search_homogenous_covariants(grad_map, rank_set, n_max, tensor_set, mapper_set)
    return grad_map


def search_homogenous_covariants(grad_map: dict[tuple[int, ...], torch.Tensor],
                                 rank_set: list[int],
                                 n_max: int,
                                 tensor_set: dict[int, torch.Tensor],
                                 mapper_set: dict[int, torch.Tensor],
                                 one_covariant_per_rank: bool = False):
    """ Search for homogenous covariants up to n_max and l_max.
    """
    logging.info(f"Searching for homogenous covariants with ranks {rank_set} and n_max={n_max}.")
    # Assert all ranks are odd
    assert all(r % 2 == 1 for r in rank_set), "All ranks must be odd"

    contractions = []
    # Create a separate grad matrix for each rank, and return it as a dictionary back. 
    assert len(grad_map) == 0, "grad_map should be empty at the start of the search"

    for l in rank_set:
        found_for_rank = False
        grad_matrix = None
        max_dof = 2 * l + 1
        current_invariants = 0
        # NOTE: In invariant testing this graph filter is missing. It depends if rank calc is faster or not.
        gfilter = GraphFilter()
        if l == 1: 
            contractions.append(((1,), ((1,),)))
            grad_map[((1,), ((1,),))] = perform_covariant_grad(ranks=(1,), 
                                                               contraction=((1,),),
                                                               tensor_set=tensor_set,
                                                               mapper_set=mapper_set)
            continue
        
        for term in iter_homogenous_covariant_terms(l, 1, n_max):
            if current_invariants == max_dof or found_for_rank:
                break
            for contraction in iter_indices(term):
                if gfilter(contraction):
                    _term, _contraction = term[1:], remove_artificial_node(contraction)
                    einsum = formula_to_einsum(tuple((_term, _contraction)))
                    logging.info(f"Processing term: {einsum} with ranks: {_term} and contraction: {_contraction}")
                    grads_list = perform_covariant_grad(ranks=_term, contraction=_contraction, tensor_set=tensor_set, mapper_set=mapper_set)
                    new_total, new_rows = test_covariants(grads_list, grad_matrix, rank_set)
                    logging.info(f"New total invariants: {new_total}, Current invariants: {current_invariants}, Expected max degrees of freedom: {max_dof}")
                    if new_total > current_invariants:
                        # If the rank increased, we found a new independent invariant! Update the current invariants count and add the new gradients to the grad matrix.
                        current_invariants = new_total
                        contractions.append((_term, _contraction))
                        grad_map[_term, _contraction] = grads_list
                        grad_matrix = torch.cat([grad_matrix, new_rows], dim=0) if grad_matrix is not None else new_rows
                        if one_covariant_per_rank:
                            found_for_rank = True
                            break
                    if current_invariants == max_dof:
                        break
                    elif current_invariants > max_dof:
                        raise ValueError(f"Current invariants {current_invariants} exceeded max degrees of freedom {max_dof} for l={l}.")


def _connect_artificial_node(contraction: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """
    Connects the artificial node (0) to the rest of the contraction.
    """
    edge_num, counts = np.unique([edge for node in contraction for edge in node], return_counts=True)
    assert np.sum(counts != 1) == len(edge_num) - 1, f"Contraction {contraction} does not have a unique dangling edge."
    dangling_edge = edge_num[counts == 1]
    assert len(dangling_edge) == 1, f"Contraction {contraction} does not have a unique dangling edge."
    dangling_edge = int(dangling_edge[0])
    return contraction + ((dangling_edge,),)


def search_heterogeneous_covariants(grad_map: dict[tuple[int, ...], torch.Tensor],
                                    rank_set: list[int],
                                    n_max: int,
                                    tensor_set: dict[int, torch.Tensor],
                                    mapper_set: dict[int, torch.Tensor],
                                    one_covariant_per_rank: bool = False):
    """ Search for heterogeneous covariants up to n_max and l_max.
    """
    logging.info(f"Searching for heterogeneous covariants with ranks {rank_set} and n_max={n_max}; only for pairs of odd ranks.")
    odd_ranks = [r for r in rank_set if r % 2 == 1]
    even_ranks = [r for r in rank_set if r % 2 == 0]
    grad_matrix = build_cov_grad_matrix(grad_map, rank_set) if len(grad_map) > 0 else None
    current_invariants = torch.linalg.matrix_rank(grad_matrix, atol=None, rtol=None).item() if grad_matrix is not None else 0

    for o1, o2 in product(odd_ranks, even_ranks):
        o1, o2 = min(o1, o2), max(o1, o2)
        for term in iter_heterogeneous_covariant_terms(o1, o2, n_max):
            gfilter = GraphFilter()
            logging.info(f"Start Term: {term}")
            for contraction in iter_indices(term):
                # Get the leave out edge.
                _, counts = np.unique(contraction[1], return_counts=True, sorted=True)
                assert counts[0] == 1, f"Contraction {contraction} does not have a unique edge to leave out."
                logging.info(f"Contraction: {contraction}")
                if gfilter(_connect_artificial_node(contraction)):
                    einsum = formula_to_einsum(tuple((term, contraction)))
                    logging.info(f"Processing term: {einsum} with ranks: {term} and contraction: {contraction}")
                    grads_list = perform_covariant_grad(ranks=term,
                                                        contraction=contraction,
                                                        tensor_set=tensor_set,
                                                        mapper_set=mapper_set)
                    new_total, new_rows = test_covariants(grads_list, grad_matrix, rank_set)
                    if new_total > current_invariants:
                        current_invariants = new_total
                        grad_map[term, contraction] = grads_list
                        grad_matrix = torch.cat([grad_matrix, new_rows], dim=0) if grad_matrix is not None else new_rows
                        logging.info(f"Found new independent invariant for term: {einsum} with ranks: {term} and contraction: {contraction}. Total invariants: {current_invariants}")
                        if one_covariant_per_rank:
                            break
                    logging.info(f"Found valid contraction: {contraction}")
                    # TODO: Test the grads and add them to the grad_map. 
                    #yield (term, contraction)

def _dof_for_rank(rank: int) -> int:
    """ Returns the degrees of freedom for a given rank. For odd ranks, this is 2*rank + 1. 

    Args:
        rank (int): The rank of the tensor.
    """
    assert rank >= 0, "Rank must be non-negative."
    return 2*rank + 1

def _build_grad_row(grads: dict[int, torch.Tensor], rank_set: list[int]) -> torch.Tensor:
    """ Build the gradient row from the grad_map.

    Args:
        grad_map (dict[tuple[int, ...], torch.Tensor]): Dictionary mapping (term, contraction) to their corresponding gradient rows.
        rank_set (list[int]): List of tensor ranks involved in the contractions.

    Returns:
        torch.Tensor: The complete gradient row.
    """
    # Put zeros in for missing ranks in the grad_map.
    row = torch.cat([grads[r] if r in grads else torch.zeros(_dof_for_rank(r)) for r in rank_set], dim=0)
    return row

def build_cov_grad_matrix(grad_map: dict[tuple[int, ...], torch.Tensor], rank_set: list[int]) -> torch.Tensor:
    """ Build the gradient matrix from the grad_map.

    Args:
        grad_map (dict[tuple[int, ...], torch.Tensor]): Dictionary mapping (term, contraction) to their corresponding gradient rows.
        rank_set (list[int]): List of tensor ranks involved in the contractions.

    Returns:
        torch.Tensor: The complete gradient matrix.
    """
    rows = []
    for grad_rows in grad_map.values():
        if isinstance(grad_rows, torch.Tensor):
            rows.append(_build_grad_row(grad_rows, rank_set))
        elif isinstance(grad_rows, list):
            for grads in grad_rows:
                rows.append(_build_grad_row(grads, rank_set))
        else: 
            raise ValueError(f"Unexpected type for rows: {type(grad_rows)}. Expected torch.Tensor or list of torch.Tensor.")
    grad_matrix = torch.stack(rows, dim=0)  # only works for this tensor order 2
    return grad_matrix

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    l_max = 3
    n_max = 4
    rank_set = list(range(1, l_max + 1, 1))
    odd_ranks = [r for r in rank_set if r % 2 == 1]
    
    grad_map, mapper_set, tensor_set = setup_search(rank_set, spherical=True)
    search_homogenous_covariants(grad_map, odd_ranks, n_max, tensor_set, mapper_set)
    grad_matrix = build_cov_grad_matrix(grad_map, rank_set)
    total_invariants = torch.linalg.matrix_rank(grad_matrix, atol=None, rtol=None).item()
    print(f"Total independent coordinates found: {total_invariants} for ranks {rank_set} and n_max={n_max}.")
    search_heterogeneous_covariants(grad_map, rank_set, n_max, tensor_set, mapper_set)
    grad_matrix = build_cov_grad_matrix(grad_map, rank_set)
    total_invariants = torch.linalg.matrix_rank(grad_matrix, atol=None, rtol=None).item()
    print(f"Total independent coordinates found after heterogeneous search: {total_invariants} for ranks {rank_set} and n_max={n_max}.")

