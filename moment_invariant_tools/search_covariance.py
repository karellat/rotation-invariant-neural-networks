
import torch
from moment_invariant_tools.search_invariants import build_grad_row
from moment_invariant_tools.contraction_formats import formula_to_einsum

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
        print(f"Computing gradient for output index {out_idx} of {outs.shape[0]}")
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