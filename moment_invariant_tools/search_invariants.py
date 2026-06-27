import copy
import itertools
import torch
import logging

from .perform_contractions import perform_contraction_grad
from .construct_tensors import sample_reduced_tensor, sample_solid_tensor
from .construct_tensors import cartesian_irreducible_mapping, cartesian_solid_mapping
from .construct_contractions import iter_indices
from .construct_terms import iter_homogenous_terms, iter_mixed_terms


def expected_hom_invariants(rank: int) -> int:
    """Expected number of homogenous invariants for a given rank. Homogenous invariants are those that can be constructed from a single rank tensor."""
    assert rank >= 0, "Rank must be non-negative"
    if rank in [0, 1]:
        return 1
    else:
        return 2 * rank - 2


def expected_mixed_invariants(rank1: int, rank2: int) -> int:
    """Expected number of mixed invariants for a given pair of ranks. Mixed invariants are those that can be constructed from two different rank tensors."""
    assert rank1 >= 0 and rank2 >= 0, "Ranks must be non-negative"
    assert rank1 != rank2, "Ranks must be different for mixed invariants"

    rank1, rank2 = map(lambda x: x(rank1, rank2), [min, max])  # ha. ha. ha.
    if rank1 == 1:
        return 2
    if rank1 >= 1:
        return 3
    raise ValueError(f"Something went wrong with ranks: {rank1},{rank2}")


def test_invariants(grads: dict[int, torch.Tensor],
                    grad_map: dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]],
                    rank_set: list[int]) -> int:
    """ Test the invariants

    Args:
        grads (dict[int, torch.Tensor]): Gradients of the current contraction with respect to the input tensors, keyed by tensor rank.
        grad_map (dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]]): A mapping of previously found invariants, where the key is a tuple of the ranks of the tensors involved in the contraction and the value is a dictionary of gradients for those invariants.
        rank_set (list[int]): A list of tensor ranks involved in the contraction.

    Returns:
        int: The number of independent invariants found.
    """
    if len(grad_map) == 0:  # No invariants in the grad_map yet!
        return 1
    # Construct the gradient matrix from the grad map and try to add the new gradient row. If the rank increases, we found a new independent invariant!
    grad_matrix = build_grad_matrix(grad_map, rank_set)
    new_row = build_grad_row(grads, rank_set)
    grad_matrix = torch.cat([grad_matrix, new_row.unsqueeze(0)], dim=0)
    tol = None
    total_invariants = torch.linalg.matrix_rank(grad_matrix, atol=tol, rtol=tol).item()
    # Return the rank of the new gradient matrix, which is the total number of independent invariants found so far (including the new one).
    return total_invariants


def get_current_invariants(grad_map: dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]], rank_set: list[int])-> int:
    """Get the current number of independent invariants found so far, based on rank of the gradient matrix constructed from the grad_map.
    grad_map is a dict of (term, contraction) -> dict of rank -> gradient tensor. We construct the gradient matrix from this and return its rank.
    
    Args: 
        grad_map (dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]]): A mapping of previously found invariants, where the key is a tuple of the ranks of the tensors involved in the contraction and the value is a dictionary of gradients for those invariants.
        rank_set (list[int]): A list of tensor ranks involved in the contraction.
    """
    if not grad_map:
        # No invariants found yet, so the rank is 0.
        return 0
    grad_matrix = build_grad_matrix(grad_map, rank_set)
    total_invariants = torch.linalg.matrix_rank(grad_matrix).item()
    return total_invariants


def build_grad_row(grads: dict[int, torch.Tensor], rank_set: list[int]) -> torch.Tensor:
    """Build a row for the gradient matrix from the gradients of different ranks.

    Args:
        grads (dict[int, torch.Tensor]): Grads w.r.t. tensor of different ranks 
        rank_set (list[int]): A list of tensor ranks involved in the contraction.

    Returns:
        torch.Tensor: A row for the gradient matrix.
    """
    row = torch.cat([grads[r] for r in rank_set])
    return row


def build_grad_matrix(grad_map: dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]], rank_set: list[int]) -> torch.Tensor:
    """Build the gradient matrix from the grad_map, which is a dict of (term, contraction) -> dict of rank -> gradient tensor.
    
    Args:
        grad_map (dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]]): A mapping of previously found invariants, where the key is a tuple of the ranks of the tensors involved in the contraction and the value is a dictionary of gradients for those invariants.
        rank_set (list[int]): A list of tensor ranks involved in the contraction.

    Returns:
        torch.Tensor: The gradient matrix.
    """
    rows = [build_grad_row(g, rank_set) for g in grad_map.values()]
    grad_matrix = torch.stack(rows, dim=0)  # only works for this tensor order 2
    return grad_matrix


def setup_search(rank_set: list[int], spherical=True) -> tuple[dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Set up the search for invariants by creating the grad_map, mapper_set, and tensor_set.

    Args:
        rank_set (list[int]): A list of tensor ranks involved in the contraction.
        spherical (bool, optional): For spherical functions only. Defaults to True. Traces are zero. 

    Returns:
        tuple[dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]], dict[int, torch.Tensor], dict[int, torch.Tensor]]: 
        A tuple containing the grad_map (grad w.r.t. tensors), mapper_set (from irreducible representations to STF tensors), and tensor_set (randomly sampled (irreducible part of) tensors).  
    """
    if spherical:
        mapping = cartesian_irreducible_mapping
        sampler = sample_reduced_tensor
    else:
        mapping = cartesian_solid_mapping
        sampler = sample_solid_tensor

    grad_map = {}

    mapper_set = {r: mapping(r).to(torch.float64) for r in rank_set}

    tensor_set = {r: sampler(r).requires_grad_(True) for r in rank_set}
    return grad_map, mapper_set, tensor_set


def search_homogenous_invariants(
    grad_map: dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]],
    rank_set: list[int],
    max_order: int,
    # TODO: Different ordering then the inhomogenous search, we should be consistent.
    tensor_set: dict[int, torch.Tensor],
    mapper_set: dict[int, torch.Tensor],
    one_invariant_per_term: bool=True,
    skip_if_hom_found: bool=True
):
    """Search for invariants created by a contraction of a single rank tensors.

    Args:
        grad_map (dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]]): Gradients w.r.t. found contractions (invariants) so far, keyed by the contraction (tuple of ranks, tuple of indices).
        rank_set (list[int]): A list of tensor ranks allowed in the contraction.
        max_order (int): The maximum order of the invariants to search for. 
        # TODO: Is max_order n or l, we should be consistent with the rest of the codebase. I think n, but we should check.
        tensor_set (dict[int, torch.Tensor]): A dictionary of input tensors, keyed by their rank.
        mapper_set (dict[int, torch.Tensor]): A dictionary of mapping tensors, keyed by their rank.
        one_invariant_per_term (bool, optional): Whether to stop searching for invariants once one is found for each term. Defaults to True.
        skip_if_hom_found (bool, optional): _description_. Defaults to True.
    """

    current_invariants = 0
    for r in rank_set:
        expected = expected_hom_invariants(r)
        start_invariants = get_current_invariants(grad_map, rank_set)

        logging.info("Finding homogenous for %s Expect %s", r, expected)
        for term in iter_homogenous_terms(r, max_order):
            logging.info("Start Term %s", term)

            for contraction in iter_indices(term):
                # Calculate the gradient of the invariants w.r.t. random input tensors.
                # Returns a dict of rank to gradient tensor.
                # TODO: Grads can be calculated only for the ranks in the term, so that the jacobian is smaller. This is a small optimization, but it can be done.
                grads = perform_contraction_grad(term, contraction, tensor_set, mapper_set)

                # Calculate rank of the new gradient matrix with the new gradients added as a row. 
                new_total = test_invariants(grads, grad_map, rank_set)

                if new_total > current_invariants:
                    # If the rank increased, we found a new independent invariant! Update the current invariants count and add the new gradients to the grad map.
                    current_invariants = new_total
                    grad_map[term, contraction] = grads
                    logging.info("Found one! %s %s", term, contraction)
                    if one_invariant_per_term:
                        break
                    if skip_if_hom_found and current_invariants - start_invariants == expected:
                        break

            if skip_if_hom_found and current_invariants - start_invariants == expected:
                break


def search_inhomogenous_invariants(
    grad_map: dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]],
    rank_set: list[int],
    tensor_set: dict[int, torch.Tensor],
    mapper_set: dict[int, torch.Tensor],
    max_order: int,
    one_invariant_per_term: bool=True,
    skip_if_mixed_found: bool=True
):
    """Search for invariants created by a contraction of a two distinct rank tensors

    Args:
        grad_map (dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]]): Gradients w.r.t. found contractions (invariants) so far, keyed by the contraction (tuple of ranks, tuple of indices).
        rank_set (list[int]): A list of tensor ranks allowed in the contraction.
        tensor_set (dict[int, torch.Tensor]): A dictionary of input tensors, keyed by their rank.
        max_order (int): The maximum order of the invariants to search for. 
        # TODO: Is max_order n or l, we should be consistent with the rest of the codebase. I think n, but we should check.
        mapper_set (dict[int, torch.Tensor]): A dictionary of mapping tensors, keyed by their rank.
        one_invariant_per_term (bool, optional): _description_. Defaults to True.
        skip_if_mixed_found (bool, optional): _description_. Defaults to True.
    """
    current_invariants = get_current_invariants(grad_map, rank_set)
    for rank_pair in itertools.combinations(rank_set, 2):
        print("Searching mixed:", rank_pair)
        cur_mixed_invariants = 0
        num_expected = expected_mixed_invariants(*rank_pair)

        for term in iter_mixed_terms(*rank_pair, max_order):
            print("Start Term", term)

            for contraction in iter_indices(term):
                grads = perform_contraction_grad(term, contraction, tensor_set, mapper_set)

                new_total = test_invariants(grads, grad_map, rank_set)

                if new_total > current_invariants:
                    current_invariants = new_total
                    grad_map[term, contraction] = grads
                    logging.info("Found one! %s %s", term, contraction)
                    cur_mixed_invariants += 1
                    if one_invariant_per_term:
                        break
            if skip_if_mixed_found and cur_mixed_invariants == num_expected:
                break


def report_results(grad_map: dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]],
                   rank_set: list[int],
                   max_order: int):
    """Read all the found invariants from the grad_map and report the number of independent invariants found for each term, as well as the total number of independent invariants found.

    Args:
        grad_map (dict[tuple[tuple[int, ...]], dict[int, torch.Tensor]]): Gradients w.r.t. found contractions (invariants) so far, keyed by the contraction (tuple of ranks, tuple of indices).
        rank_set (list[int]): A list of tensor ranks allowed in the contraction.
        max_order (int): The maximum order of the invariants to search for.
    """
    row_indices = {k: i for i, k in enumerate(grad_map.keys())}
    contract_from_row = {v: k for k, v in row_indices.items()}

    grad_matrix = torch.stack(
        [torch.cat([v[r] for r in rank_set]) for v in grad_map.values()], axis=0
    )  # only works for this tensor order 2

    print("input orders:", rank_set)
    print("maximum factors:", max_order)

    prev_rank = 0
    this_rank = 0
    for i in range(grad_matrix.shape[0]):
        this_rank = torch.linalg.matrix_rank(grad_matrix[: i + 1]).item()
        ranks, indices = contract_from_row[i]
        independent = this_rank > prev_rank

        if len(set(ranks)) == 1:
            type_ = f"hom. {ranks[0]}"
        else:
            type_ = f"mixed: {tuple(set(ranks))}"

        contract_sig = " ".join(["".join([str(y) for y in x]) for x in indices])

        print(i, independent, type_, ranks, contract_sig)  # because [:1] means using 0 only
        prev_rank = this_rank
    print("Found total independent invariants:", this_rank)


def perform_search(rank_set, 
                   max_order,
                   one_invariant_per_term=True,
                   skip_if_mixed_found=True, 
                   skip_if_hom_found=True):
    """_summary_

    Args:
        rank_set (_type_): _description_
        max_order (_type_): _description_
        one_invariant_per_term (bool, optional): _description_. Defaults to True.
        skip_if_mixed_found (bool, optional): _description_. Defaults to True.
        skip_if_hom_found (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    grad_map, mapper_set, tensor_set = setup_search(rank_set)

    search_homogenous_invariants(
        grad_map,
        rank_set,
        max_order,
        tensor_set,
        mapper_set,
        one_invariant_per_term=one_invariant_per_term,
        skip_if_hom_found=skip_if_hom_found,
    )
    search_inhomogenous_invariants(
        grad_map,
        rank_set,
        tensor_set,
        mapper_set,
        max_order,
        one_invariant_per_term=one_invariant_per_term,
        skip_if_mixed_found=skip_if_mixed_found,
    )

    report_results(grad_map, rank_set, max_order)
    return grad_map


def perform_independent_searches(rank_set, max_order, one_invariant_per_term=True, skip_if_mixed_found=True, skip_if_hom_found=True):

    full_grad_map, mapper_set, tensor_set = setup_search(rank_set)

    for r in rank_set:
        grad_map = {}
        search_homogenous_invariants(
            grad_map,
            {r},
            max_order,
            tensor_set,
            mapper_set,
            one_invariant_per_term=one_invariant_per_term,
            skip_if_hom_found=skip_if_hom_found,
        )
        full_grad_map.update(grad_map)

    for rank_pair in itertools.combinations(rank_set, 2):
        rank_pair = set(rank_pair)
        grad_map = {}
        search_inhomogenous_invariants(
            grad_map,
            rank_pair,
            tensor_set,
            mapper_set,
            max_order,
            one_invariant_per_term=one_invariant_per_term,
            skip_if_mixed_found=skip_if_mixed_found,
        )
        full_grad_map.update(grad_map)

    report_results(full_grad_map, rank_set, max_order)
    return full_grad_map


def perform_overcomplete_search(rank_set, max_order):
    """_summary_

    Args:
        rank_set (_type_): _description_
        max_order (_type_): _description_

    Returns:
        _type_: _description_
    """

    full_grad_map, mapper_set, tensor_set = setup_search(rank_set)

    rank_grad_maps = {}
    for r in rank_set:
        grad_map = {}
        search_homogenous_invariants(grad_map, {r}, max_order, tensor_set, mapper_set, one_invariant_per_term=False, skip_if_hom_found=True)
        full_grad_map.update(grad_map)
        rank_grad_maps[r] = grad_map

    for rank_pair in itertools.combinations(rank_set, 2):
        rank_pair = set(rank_pair)
        r0, r1 = rank_pair
        # This kinda represents the vanishing, you take just the ranks!!
        grad_map = {**rank_grad_maps[r0], **rank_grad_maps[r1]}
        search_inhomogenous_invariants(
            grad_map, rank_pair, tensor_set, mapper_set, max_order, one_invariant_per_term=False, skip_if_mixed_found=True
        )
        full_grad_map.update(grad_map)

    report_results(full_grad_map, rank_set, max_order)
    return full_grad_map


def search_term_iterable(terms):

    terms = list(terms)
    rank_set = set()
    max_order = 0
    for t in terms:
        rank_set.update(t)
        max_order = max(max_order, len(t))

    current_invariants = 0
    grad_map, mapper_set, tensor_set = setup_search(rank_set)
    for term in terms:
        print("Searching", term)

        for contraction in iter_indices(term):
            grads = perform_contraction_grad(term, contraction, tensor_set, mapper_set)

            new_total = test_invariants(grads, grad_map, rank_set)

            if new_total > current_invariants:
                current_invariants = new_total
                grad_map[term, contraction] = grads
                print("Found one!", term, contraction)

    report_results(grad_map, rank_set, max_order)
    return grad_map


def test_independence(contractions, spherical=True):
    all_ranks = set(len(x) for c in contractions for x in c)

    grad_map, mapper_set, tensor_set = setup_search(all_ranks, spherical=spherical)

    for c in contractions:
        this_ranks = tuple(len(t) for t in c)
        grads = perform_contraction_grad(this_ranks, c, tensor_set, mapper_set)
        grad_map[this_ranks, c] = grads

    report_results(grad_map, all_ranks, max_order="Not given")
    return grad_map
