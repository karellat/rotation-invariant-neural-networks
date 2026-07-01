import collections
import re
from collections.abc import Iterable


_EINSUM_LABEL_RE = re.compile(r"^[A-Za-z]*$")

def formula_to_einsum(formula: tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]) -> str:
    """Convert a contraction formula to an einsum string.

    Args:
        formula (tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]): A contraction formula in the format (ranks, edges).

    Returns:
        str: An einsum string representing the contraction.
    """
    ranks, edges = formula
    if not is_valid_formula(formula):
        raise ValueError("Invalid contraction formula.")

    # Create a mapping from edge labels to einsum labels (a-z)
    label_mapping = {}
    current_label = ord('a')
    for edge_tuple in edges:
        for label in edge_tuple:
            if label not in label_mapping:
                label_mapping[label] = chr(current_label)
                current_label += 1

    # Build the einsum input strings
    edge_counter = collections.Counter()
    input_strings = []
    for edge_tuple in edges:
        input_strings.append(''.join(label_mapping[label] for label in edge_tuple))
        edge_counter.update(label_mapping[label] for label in edge_tuple)

    # Check ranks and if they don't match add dangling edges to input and output string
    for idx, (rank, edge_tuple) in enumerate(zip(ranks, edges)):
        if len(edge_tuple) < rank:
            # Add dangling edges to input and output string
            dangling_edges = rank - len(edge_tuple)
            for _ in range(dangling_edges):
                new_label = chr(current_label)
                edge_counter[new_label] += 1
                current_label += 1
                input_strings[idx] += new_label

    # Determine the output string (free edges)
    output_string = ""
    for label, count in edge_counter.items():
        if count == 1:
            if output_string == "":
                output_string += label
            else: 
                output_string += ", " + label
        elif count == 2 or count == 0:
            # Every ok
            continue
        else:
            raise ValueError(f"Invalid edge count for label '{label}': {count}")

    return ','.join(input_strings) + '->' + output_string

def einsum_to_formula(einsum: str) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Convert an einsum string to a contraction formula.

    Args:
        einsum (str): An einsum string representing the contraction.

    Returns:
        tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]: A contraction formula in the format (ranks, edges).
    """
    if not is_valid_einsum(einsum):
        raise ValueError("Invalid einsum string.")

    einsum = einsum.replace(" ", "")
    if "->" in einsum:
        input_part, output_part = einsum.split("->")
    else:
        input_part, output_part = einsum, None

    operands = input_part.split(",")
    ranks = tuple(len(operand) for operand in operands)
    edges = tuple(tuple(ord(label) - ord('a') for label in operand) for operand in operands)

    return ranks, edges

def is_valid_formula(formula: tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]) -> bool:
    """Return True if a contraction formula has a valid graph-like structure.

    Formula format is

        (ranks), ((edges for node 0), ..., (edges for node n))

    A valid formula has one edge tuple per rank, each edge tuple length matches
    its rank, and every edge label appears either once (dangling/free edge) or
    twice (contracted edge). Labels appearing three or more times would be a
    hyperedge, not an ordinary graph contraction.
    """
    if not isinstance(formula, tuple) or len(formula) != 2:
        return False

    ranks, edges = formula
    if not _is_tuple_of_nonnegative_ints(ranks):
        return False
    if not isinstance(edges, tuple) or len(edges) != len(ranks):
        return False

    edge_labels = []
    for rank, node_edges in zip(ranks, edges):
        if not _is_tuple_of_nonnegative_ints(node_edges):
            return False
        if len(node_edges) > rank:
            return False
        if len(set(node_edges)) != len(node_edges):
            # A repeated label on one tensor is a self-trace/loop. The current
            # graph/formula code assumes ordinary edges between tensor nodes.
            return False
        edge_labels.extend(node_edges)

    return _has_valid_edge_counts(edge_labels)


def is_valid_einsum(einsum: str) -> bool:
    """Return True if an einsum string describes a valid graph-like contraction.

    This accepts strings such as ``"ij,jk,ki->"`` and ``"ij,jk,k->i"``.
    Ellipses and repeated labels within a single operand are intentionally not
    supported here because they do not map cleanly to the formula/multigraph
    format used in this package.
    """
    if not isinstance(einsum, str):
        return False

    einsum = einsum.replace(" ", "")
    if not einsum:
        return False
    if einsum.count("->") > 1:
        return False

    if "->" in einsum:
        input_part, output_part = einsum.split("->")
    else:
        input_part, output_part = einsum, None

    operands = input_part.split(",")
    if not operands or any(operand == "" for operand in operands):
        return False
    if any(not _EINSUM_LABEL_RE.fullmatch(operand) for operand in operands):
        return False
    if any(len(set(operand)) != len(operand) for operand in operands):
        return False

    input_labels = "".join(operands)
    counts = collections.Counter(input_labels)
    if not _has_valid_edge_counts(input_labels):
        return False

    free_labels = {label for label, count in counts.items() if count == 1}
    if output_part is None:
        return True

    if not _EINSUM_LABEL_RE.fullmatch(output_part):
        return False
    if len(set(output_part)) != len(output_part):
        return False

    output_labels = set(output_part)
    if not output_labels <= set(counts):
        return False

    return output_labels == free_labels


def _is_tuple_of_nonnegative_ints(values: object) -> bool:
    return isinstance(values, tuple) and all(isinstance(value, int) and value >= 0 for value in values)


def _has_valid_edge_counts(labels: Iterable[object]) -> bool:
    return all(count <= 2 for count in collections.Counter(labels).values())
