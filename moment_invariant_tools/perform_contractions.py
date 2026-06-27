import torch
import logging

from .construct_tensors import cartesian_irreducible_mapping


def contract(tensors: list[torch.Tensor], indices: list[tuple[int, ...]]) -> torch.Tensor:
    """Contract a list of tensors along specified indices.

    Args:
        tensors (list[torch.Tensor]): List of tensors to contract. It matches the order of the indices list.
        indices (list[tuple[int, ...]]): List of indices for each tensor. ((0, 1), (1, 2)) means contract the second index of the first tensor with the first index of the second tensor.

    Returns:
        torch.Tensor: The contracted tensor.
    """
    args = list(zip(tensors, indices))
    args = [x for y in args for x in y]

    return torch.einsum(*args)


def contraction_mapping(mappers, indices):
    logging.warning("This not-used and might be removed in future versions.")
    ranks = [len(m.shape) - 1 for m in mappers]
    start_rank_indexes = sum(ranks)  # conservatively, should start 1/2 lower and be fine.
    new_indices = [(*ind, r) for r, ind in enumerate(indices, start=start_rank_indexes)]
    out_indices = [ind[-1] for ind in new_indices]

    args = [(m, ind) for m, ind in zip(mappers, new_indices)]
    args = [i for t in args for i in t]

    return torch.einsum(*args, out_indices)


def make_contraction_map(indices):
    logging.warning("This not-used and might be removed in future versions.")
    ranks = [len(i) for i in indices]
    rank_set = set(ranks)
    mappers = {r: cartesian_irreducible_mapping(r) for r in rank_set}

    return contraction_mapping([mappers[r] for r in ranks], indices)


def perform_contraction_grad(ranks: tuple[int, ...],
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
    output = contract(input_tensors, contraction)

    # Calculate the gradient of the output with respect to the input tensors.
    # TODO: Change this to the https://docs.pytorch.org/docs/2.12/generated/torch.func.jacrev.html
    contraction_grad_values = torch.autograd.grad(output, tensor_set.values(), allow_unused=True)

    contraction_grad_dict = {k: v for k, v in zip(tensor_set.keys(), contraction_grad_values)}
    # Substitute none for zeros.
    for k, v in contraction_grad_dict.items():
        if v is None:
            contraction_grad_dict[k] = torch.zeros_like(tensor_set[k])
    return contraction_grad_dict
