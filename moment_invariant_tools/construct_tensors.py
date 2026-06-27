import torch
import itertools
import collections
import logging

NDIM = 3  # number of spatial dimensions


def trace(tensor: torch.Tensor, axis_1: int, axis_2: int) -> torch.Tensor:
    """ Calculate tensor trace over axis_1 and axis_2

    Args:
        tensor (torch.Tensor): The input tensor.
        axis_1 (int): The first axis along which to calculate the trace.
        axis_2 (int): The second axis along which to calculate the trace.

    Returns:
        torch.Tensor: The trace of the tensor along the specified axes.
    """
    return torch.diagonal(tensor, 0, axis_1, axis_2).sum(dim=-1)


def tpl(int_tensor: torch.Tensor) -> tuple[int, ...]:
    """ Convert a tensor of integers to a tuple of integers.
    Args:
        int_tensor (torch.Tensor): The input tensor of integers.

    Returns:
        tuple[int, ...]: The tuple of integers.
    """
    return tuple(x.item() for x in int_tensor.unbind())


# TODO: This would be nice to pack into a class.
def build_full_mapping(ind_order: int, ndim: int=NDIM) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a mapping from Cartesian tensors to symmetric bases.

    For a given rank, produces a mapping with shape [ndim,..., ndim, independent_values],
    where ndim repeats according to the tensor rank. Independent values are represented
    using one-hot vectors like [1, 0, 0, 0, 0] or signed vectors like [-1, 0, -1, 0, 0]
    for trace-constrained elements, encoding which indices must be summed to maintain
    the trace-free symmetric basis. The first output is typically used; the other two
    are provided for debugging and understanding the mapping structure.

    Args:
        ind_order (int): The rank of the tensor to be mapped.
        ndim (int, optional): Number of spatial dimensions. Defaults to NDIM.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Three tensors representing:
            - Mapping from Cartesian to STF (symmetric trace-free) basis
            - Mapping from Cartesian to symmetric basis
            - Mapping from symmetric to trace-free symmetric basis
    """
    assert ind_order >= 1, "Tensor order must be at least 1."
    assert ndim >= 1, "Number of dimensions must be at least 1."

    # First, construct the map from the full cartesian space to a symmetric basis,
    # mapping each element from the cartesian space to the sorted version of its indices.
    input_ind = []
    output_ind = []
    for ind in itertools.product(range(ndim), repeat=ind_order):
        input_ind.append(ind)
        output_ind.append(tuple(sorted(ind)))
    out_ind = torch.as_tensor(output_ind)
    in_ind = torch.as_tensor(input_ind)

    # List of symmetric indices only.
    needed_ind = torch.unique(out_ind, dim=0)

    # Figure out numbering for nonsymmetric indices within the symmetric ones.
    # # This step is of somewhat large size, could maybe be reduced with better algorithm.
    equal = (needed_ind == out_ind.unsqueeze(1)).all(dim=2)
    out_order = torch.where(equal)[1]

    # build sparse matrix -> all values are 1
    ind_comb = torch.stack(
        (torch.arange(len(out_order)), out_order),
    )
    xform = torch.sparse_coo_tensor(ind_comb, torch.ones(len(out_order)), dtype=torch.int64).to_dense()
    # This maps symmetric space to d^order space
    # shape (d^o, t(o)) where t(o) is the triangular number.

    # Now we need to build the map from traceless symmetric tensors to symmetric ones.

    # Which indices are constrainted by tracelessness
    # Constrained elements are ones that end with 2,2 in the full cartesian space.
    # # Again, may use more memory than needed here.
    constrained_elements = (needed_ind[:, -2:] == torch.as_tensor([2, 2])).all(dim=-1)
    constrained_pos = torch.where(constrained_elements)[0]
    # Set of indices which are constrained by trace.
    constrained_ind = needed_ind[constrained_elements]
    # print("constrained_ind",constrained_ind)

    # Which indices are not constrained
    free_elements = ~constrained_elements
    free_pos = torch.where(free_elements)[0]

    if ind_order == 1:
        # Exception to above which is wrong when only one dimension
        constrained_ind = []
        free_elements = [True, True, True]

    # Now we use regular python b/c it is easier to construct these loops.

    # Map from the index space to the symmetric basis number.
    bare_sym_map = {tpl(x): i for i, x in enumerate(needed_ind.unbind(0))}
    # print("bare sym map",bare_sym_map)
    # Map from the index space to traceless symmetric baseless number, but only for values
    # that are in both (i.e. not affected by tracelessness)
    bare_traceless_map = {tpl(x): i for i, x in enumerate(needed_ind[free_elements].unbind(0))}

    # Map from symetric indices to traceless ones, but onyl for values that are in both.
    sym_traceless_map = {bare_sym_map[x]: j for x, j in bare_traceless_map.items()}

    # Initialize full map in "csr" form based on unconstrained elements.
    sym_traceless_csr = collections.defaultdict(list)
    for k, v in sym_traceless_map.items():
        sym_traceless_csr[k].append((v, 1))

    # Extend map for trace-constrained elements
    for x in map(tpl, constrained_ind):
        x_sym = bare_sym_map[x]
        # print(x,x_sym)
        new_sym_vals = []
        # Get the two other components x...00 and x_...11 (constraint is on x...22)
        for k in (0, 1):
            a = tuple(sorted(x[:-2] + (k, k)))
            aa = bare_sym_map[a]
            new_sym_vals.append(aa)

        # Write this is in terms of
        new_traceless_vals = collections.Counter()
        for aa in new_sym_vals:
            for aaa, v in sym_traceless_csr[aa]:
                new_traceless_vals[aaa] += -v
        new_traceless_vals = [(k, v) for k, v in new_traceless_vals.items()]
        sym_traceless_csr[x_sym].extend(new_traceless_vals)

    # Rewrite in coo form
    sym_traceless_coo = collections.Counter()
    for i, row in sym_traceless_csr.items():
        for j, val in row:
            sym_traceless_coo[i, j] += val

    # Convert into pytorch tensor
    ind, vals = zip(*list(sym_traceless_coo.items()))
    ind = torch.as_tensor(ind).T
    vals = torch.as_tensor(vals)
    xform2 = torch.sparse_coo_tensor(ind, vals, size=(xform.shape[1], 2 * ind_order + 1))

    # The full map is simply the map (cartesian index, symmetric) @ (symmetric, sym. traceless)
    full_xform = xform @ xform2

    # unflatten cartesian set.
    full_xform = full_xform.reshape(*((ndim,) * ind_order), -1)
    xform = xform.reshape(*((ndim,) * ind_order), -1)
    return full_xform, xform, xform2


def cartesian_irreducible_mapping(ind_order: int) -> torch.Tensor:
    """Generate a mapping from Cartesian tensors to irreducible representations (symmetric trace-free tensors).

    Args:
        ind_order (int): The rank of the tensor to be mapped.

    Returns:
        torch.Tensor: A tensor mapping from Cartesian to irreducible representations, with shape [ndim,..., ndim, (2 * ind_order + 1)], where ndim repeats according to the tensor rank.
          Independent values are represented using one-hot vectors like [1, 0, 0, 0, 0] or signed vectors like [-1, 0, -1, 0, 0] for trace-constrained elements, encoding which indices must be summed to maintain the trace-free symmetric basis.
    """
    full_xform, _, _ = build_full_mapping(ind_order)

    return full_xform


def cartesian_solid_mapping(ind_order: int) -> torch.Tensor:
    """Generate a mapping from Cartesian tensors to solid harmonics (symmetric tensors).

    Args:
        ind_order (int): The rank of the tensor to be mapped.

    Returns:
        torch.Tensor: A tensor mapping from Cartesian to solid harmonics, with shape [ndim,..., ndim, independent_values], where ndim repeats according to the tensor rank.
          Independent values are represented using one-hot vectors like [1, 0, 0, 0, 0].
    """
    _, cartesian_symmetric, _ = build_full_mapping(ind_order)

    return cartesian_symmetric


def check_traceless(tensor: torch.Tensor) -> bool:
    """Check if Cartesian tensor [ndim, ...., ndim], len([ndim, ..., ndim]) = tensor order is traceless, meaning that all elements of the form tensor[..., i, i] sum to zero for all i.

    Args:
        tensor (torch.Tensor): The Cartesian tensor to check.

    Returns:
        bool: True if the tensor is traceless, False otherwise.
    """
    tensor_order = tensor.dim()

    for i, j in itertools.combinations(range(tensor_order), r=2):
        close = torch.allclose(trace(tensor, 0, 1), torch.as_tensor(0, dtype=tensor.dtype))
        if not close:
            return False
    # This else statement is a joke, it is not needed and confuses the way this function works.
    else:
        return True


def check_symmetric(tensor: torch.Tensor) -> bool:
    """Check if Cartesian tensor [ndim, ...., ndim], len([ndim, ..., ndim]) = tensor order is symmetric, meaning that all elements of the form tensor[..., i, j, ...] are equal to tensor[..., j, i, ...] for all i, j.

    Args:
        tensor (torch.Tensor): The Cartesian tensor to check.

    Returns:
        bool: True if the tensor is symmetric, False otherwise.
    """
    tensor_order = tensor.dim()

    for p in itertools.permutations(range(tensor_order), r=tensor_order):
        close = torch.allclose(tensor.permute(p), tensor)
        if not close:
            return False
    # This else statement is a joke, it is not needed and confuses the way this function works.
    else:
        return True


def sample_reduced_tensor(rank: int, dtype=torch.float64) -> torch.Tensor:
    """Sample the independent values of a symmetric tracefree tensor.

    Returns random values for the (2*rank + 1) independent components of a symmetric
    trace-free tensor of the given rank. Note that this returns only the independent
    values, not the full tensor.

    Args:
        rank (int): The rank of the tensor.
        dtype (_type_, optional): The data type of the sampled values. Defaults to torch.float64.

    Returns:
        torch.Tensor: Random independent values for the symmetric trace-free tensor.
    """
    assert rank >= 0, "Rank must be at least 0."
    rand_vals = torch.rand(2 * rank + 1, dtype=dtype)
    return rand_vals


def sample_solid_tensor(rank: int, dtype=torch.float64) -> torch.Tensor:
    """
    Sample the independent values of a symmertic tensor solid harmonic tensor.
    Args:
        rank (int): The rank of the tensor.
        dtype (_type_, optional): The data type of the sampled values. Defaults to torch.float64.

    Returns:
        torch.Tensor: Random independent values for the symmetric solid harmonic tensor.
    """
    assert rank >= 0, "Rank must be at least 0."
    size = (rank + 1) * (rank + 2) / 2
    size = int(size)
    rand_vals = torch.rand(size, dtype=dtype)
    return rand_vals


def sample_tensor(map_: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """Given the mapping produced by build_full_mapping, sample a random tensor in the Cartesian space by sampling random independent values for the symmetric trace-free basis
    and applying the mapping to produce the full Cartesian tensor. This is useful for testing and understanding the structure of the mapping.

    Args:
        map_ (torch.Tensor): The mapping tensor.
        dtype (_type_, optional): The data type of the sampled values. Defaults to torch.float64.

    Returns:
        torch.Tensor: The sampled Cartesian tensor.
    """
    n_components = map_.shape[-1]
    rand_vals = torch.rand(n_components, dtype=dtype)
    return map_.to(dtype) @ rand_vals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test SFT Tensors
    for rank in range(1, 8):
        logging.info(f"Testing rank {rank}...")
        mapping = cartesian_irreducible_mapping(rank)
        tensor = sample_tensor(mapping, dtype=torch.float64)
        assert check_traceless(tensor), f"Tensor of rank {rank} is not traceless."
        assert check_symmetric(tensor), f"Tensor of rank {rank} is not symmetric."
    # Test Solid Harmonic Tensors
    for rank in range(1, 8):
        logging.info(f"Testing rank {rank}...")
        mapping = cartesian_solid_mapping(rank)
        tensor = sample_tensor(mapping, dtype=torch.float64)
        assert check_symmetric(tensor), f"Tensor of rank {rank} is not symmetric."
    logging.info("All tests passed!")
