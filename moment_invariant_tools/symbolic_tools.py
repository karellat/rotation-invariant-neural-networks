import numpy as np
import sympy as sp

from IPython.display import display, Math
from .construct_tensors import cartesian_irreducible_mapping

def symbolic_tensor_from_mapping(name, mapping):
    """
    Build a symbolic Cartesian tensor from a Cartesian-to-irreducible mapping.

    mapping shape:
        [3, ..., 3, 2*l + 1]

    The final axis indexes the independent STF symbols.
    """
    if hasattr(mapping, "detach"):
        coeffs = mapping.detach().cpu().numpy()
    else:
        coeffs = np.asarray(mapping)

    n_components = coeffs.shape[-1]
    variables = np.array(
        sp.symbols(f"{name}_0:{n_components}"),
        dtype=object,
    )

    tensor = np.tensordot(
        coeffs.astype(int),
        variables,
        axes=([-1], [0]),
    )

    return tensor, list(variables)

def symbolic_SFT_tensors(names, rank_set):
    tensors = []
    variables = []
    assert len(names) == len(rank_set), "names and rank_set must have the same length"
    
    for i, rank in enumerate(rank_set):
        mapping = cartesian_irreducible_mapping(rank)
        tensor, var_list = symbolic_tensor_from_mapping(names[i], mapping)
        tensors.append(tensor)
        variables.extend(var_list)
    return tensors, variables

def einsum_expr(subscripts, *arrays, expand=False):
    """
    Convert an einsum contraction into a SymPy expression.

    For scalar contractions, returns a SymPy expression.
    For non-scalar outputs, returns a NumPy array of SymPy expressions.
    """
    out = np.einsum(subscripts, *arrays, optimize=False)

    if isinstance(out, np.ndarray):
        return out

    out = sp.sympify(out)
    return sp.expand(out) if expand else out

def display_tensor_latex(T, name="T"):
    sp.init_printing(use_latex=True)
    rank = T.ndim

    if rank == 1:
        expr = sp.Matrix(T)
        display(Math(rf"{name} = {sp.latex(expr)}"))

    elif rank == 2:
        expr = sp.Matrix(T)
        display(Math(rf"{name}_{{ij}} = {sp.latex(expr)}"))

    elif rank == 3:
        for k in range(T.shape[2]):
            expr = sp.Matrix(T[:, :, k])
            display(Math(rf"{name}_{{ij{k}}} = {sp.latex(expr)}"))

    elif rank == 4:
        for k in range(T.shape[2]):
            for l in range(T.shape[3]):
                expr = sp.Matrix(T[:, :, k, l])
                display(Math(rf"{name}_{{ij{k}{l}}} = {sp.latex(expr)}"))

    else:
        raise ValueError("Only rank 1-4 tensors are supported")