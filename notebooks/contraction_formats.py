import collections
import string

import networkx as nx

# Clean code

FORMULA_EXAMPLES = [
    (
        "E1, i->i, DOF=3",
        ((1,), ((),)),
    ),
    (
        "E2, i, ij -> j, DOF=3",
        ((1, 2), ((0,), (0,))),
    ),
    (
        "I1, i,i -> , DOF=1",
        ((1, 1), ((0,), (0,))),
    ),
    (
        "I2, ij, ij -> , DOF=1",
        ((2, 2), ((0, 1), (0, 1))),
    ),
    (
        "I3, ij, jk, ik -> , DOF=1",
        ((2, 2, 2), ((0, 1), (0, 2), (1, 2))),
    ),
]

# Generated code 
def index_symbols():
    lower_symbols = "ijklmnopqrstuvwxyzabcdefgh"

    yield from lower_symbols
    yield from string.ascii_uppercase

    index = 0
    while True:
        yield f"_{index}"
        index += 1


def formula_to_einsum(formula):
    """
    Convert ``(ranks, edge_specs)`` into an einsum-style signature.

    Repeated edge labels are contracted. Edge labels that appear once, plus
    unlabeled rank-padding slots, become output indices.
    """
    ranks, edge_specs = formula

    if len(ranks) != len(edge_specs):
        raise ValueError(
            "formula must include one edge specification for each rank"
        )

    label_counts = collections.Counter(
        edge_label
        for edge_labels in edge_specs
        for edge_label in edge_labels
    )
    symbol_stream = index_symbols()
    label_symbols = {}
    operands = []
    output_symbols = []

    for node, (rank, edge_labels) in enumerate(zip(ranks, edge_specs)):
        if len(edge_labels) > rank:
            raise ValueError(
                f"node {node} has rank {rank}, but {len(edge_labels)} "
                "edge labels were provided"
            )

        operand_symbols = []

        for edge_label in edge_labels:
            if label_counts[edge_label] > 2:
                raise ValueError(
                    f"edge label {edge_label!r} appears "
                    f"{label_counts[edge_label]} times; labels must appear "
                    "at most twice"
                )

            if edge_label not in label_symbols:
                label_symbols[edge_label] = next(symbol_stream)

            symbol = label_symbols[edge_label]
            operand_symbols.append(symbol)

            if label_counts[edge_label] == 1:
                output_symbols.append(symbol)

        for _ in range(len(edge_labels), rank):
            symbol = next(symbol_stream)
            operand_symbols.append(symbol)
            output_symbols.append(symbol)

        operands.append("".join(operand_symbols))

    return f"{', '.join(operands)} -> {''.join(output_symbols)}"


def einsum_to_formula(signature):
    inputs, _output = signature.replace(" ", "").split("->", 1)
    operands = inputs.split(",") if inputs else []
    parsed_operands = []
    symbol_counts = collections.Counter()

    for operand in operands:
        symbols = []
        i = 0
        while i < len(operand):
            if operand[i] == "_":
                j = i + 1
                while j < len(operand) and operand[j].isdigit():
                    j += 1
                symbol = operand[i:j]
                i = j
            else:
                symbol = operand[i]
                i += 1
            symbols.append(symbol)
            symbol_counts[symbol] += 1
        parsed_operands.append(symbols)

    label_ids = {}
    next_label = 0
    edge_specs = []
    for symbols in parsed_operands:
        edge_labels = []
        for symbol in symbols:
            if symbol_counts[symbol] > 1:
                label = label_ids.get(symbol)
                if label is None:
                    label = next_label
                    label_ids[symbol] = label
                    next_label += 1
                edge_labels.append(label)
        edge_specs.append(tuple(edge_labels))

    return tuple(map(len, parsed_operands)), tuple(edge_specs)


def add_dangling_edge(G, node, edge_label=None, name=None, **attrs):
    """Attach one dangling edge to a real graph node."""
    if name is None:
        stub_count = sum(
            1 for _, data in G.nodes(data=True)
            if data.get("kind") == "stub"
        )
        name = f"dangling_{stub_count}"

    stub = ("stub", name)
    G.add_node(stub, kind="stub", edge_label=edge_label)
    G.add_edge(
        node,
        stub,
        kind="dangling",
        edge_label=edge_label,
        **attrs,
    )
    return stub


def graph_from_formula(formula):
    """
    Construct a tensor-contraction graph from ``(ranks, edge_specs)``.

    Example
    -------
    ``((2, 2, 2, 4), ((0, 1), (1, 2), (3, 4), (0, 2, 3, 4)))``
    creates four real nodes. Matching edge labels connect nodes; edge labels
    that occur only once become dangling edges. If a node rank is larger than
    its listed edge labels, the missing edges are also added as dangling edges.
    """
    ranks, edge_specs = formula

    if len(ranks) != len(edge_specs):
        raise ValueError(
            "formula must include one edge specification for each rank"
        )

    G = nx.MultiGraph()
    label_occurrences = collections.defaultdict(list)

    for node, (rank, edge_labels) in enumerate(zip(ranks, edge_specs)):
        if len(edge_labels) > rank:
            raise ValueError(
                f"node {node} has rank {rank}, but {len(edge_labels)} "
                "edge labels were provided"
            )

        G.add_node(node, kind="tensor", rank=rank)

        for slot, edge_label in enumerate(edge_labels):
            label_occurrences[edge_label].append((node, slot))

        for missing_slot in range(len(edge_labels), rank):
            add_dangling_edge(
                G,
                node,
                name=f"node_{node}_slot_{missing_slot}",
                slot=missing_slot,
                reason="rank_padding",
            )

    for edge_label, occurrences in label_occurrences.items():
        if len(occurrences) == 1:
            node, slot = occurrences[0]
            add_dangling_edge(
                G,
                node,
                edge_label=edge_label,
                name=f"edge_{edge_label}",
                slot=slot,
                reason="unmatched_label",
            )
        elif len(occurrences) == 2:
            (u, u_slot), (v, v_slot) = occurrences
            G.add_edge(
                u,
                v,
                kind="contraction",
                edge_label=edge_label,
                slots=(u_slot, v_slot),
            )
        else:
            raise ValueError(
                f"edge label {edge_label!r} appears {len(occurrences)} "
                "times; labels must appear at most twice"
            )

    return G


def formula_example_graphs(examples=FORMULA_EXAMPLES):
    """Build the example graphs as ``(title, graph)`` pairs."""
    return [
        (formula_to_einsum(formula), graph_from_formula(formula))
        for title, formula in examples
    ]


def plot_formula_examples(examples=FORMULA_EXAMPLES):
    """Plot formula examples with the contraction visualization helpers."""
    import matplotlib.pyplot as plt

    try:
        from contraction_visualization import draw_graph_example
    except ImportError:
        from notebooks.contraction_visualization import draw_graph_example

    graph_examples = formula_example_graphs(examples)
    fig, axes = plt.subplots(1, len(graph_examples), figsize=(4 * len(graph_examples), 4))

    if len(graph_examples) == 1:
        axes = [axes]

    for ax, (title, graph) in zip(axes, graph_examples):
        draw_graph_example(ax, graph, title)

    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plot_formula_examples()
    plt.show()
