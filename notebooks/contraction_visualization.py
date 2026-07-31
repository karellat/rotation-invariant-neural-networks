import itertools

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PARALLEL_EDGE_SPACING = 0.08
MAX_PARALLEL_EDGE_SPAN = 0.14


def add_dangling_edge(G, u, name=None, **attrs):
    """
    Add a dangling edge attached to real node u.
    The other endpoint is a special stub node.
    """
    if name is None:
        name = f"dangling_{sum(1 for _, d in G.nodes(data=True) if d.get('kind') == 'stub')}"

    stub = ("stub", name)
    G.add_node(stub, kind="stub")
    G.add_edge(u, stub, kind="dangling", **attrs)
    return stub


def set_rank_from_degree(G):
    for node, data in G.nodes(data=True):
        if data.get("kind") != "stub":
            G.nodes[node]["rank"] = G.degree(node)


def real_and_stub_nodes(G):
    real_nodes = [
        node for node, data in G.nodes(data=True)
        if data.get("kind") != "stub"
    ]
    stub_nodes = [
        node for node, data in G.nodes(data=True)
        if data.get("kind") == "stub"
    ]
    return real_nodes, stub_nodes


def polygon_layout(nodes, radius=0.62, start_angle=90):
    if len(nodes) == 1:
        return {nodes[0]: np.array([0.0, 0.0])}

    angles = np.deg2rad(
        start_angle + np.linspace(0, 360, len(nodes), endpoint=False)
    )
    return {
        node: radius * np.array([np.cos(angle), np.sin(angle)])
        for node, angle in zip(nodes, angles)
    }


def square_layout(nodes, radius=0.58):
    corners = np.array([
        [-radius, radius],
        [radius, radius],
        [radius, -radius],
        [-radius, -radius],
    ])
    return {node: corner for node, corner in zip(nodes, corners)}


def two_node_layout(nodes, radius=0.58):
    return {
        nodes[0]: np.array([-radius, 0.0]),
        nodes[1]: np.array([radius, 0.0]),
    }


def shared_triangle_layout(G, nodes):
    if len(nodes) != 5:
        return None

    simple_neighbors = {
        node: set(G.neighbors(node)) & set(nodes)
        for node in nodes
    }
    shared_nodes = [
        node for node, neighbors in simple_neighbors.items()
        if len(neighbors) == 4
    ]

    if len(shared_nodes) != 1:
        return None

    shared = shared_nodes[0]
    neighbors = list(simple_neighbors[shared])
    neighbor_pairs = [
        (u, v) for u, v in itertools.combinations(neighbors, 2)
        if G.number_of_edges(u, v) > 0
    ]

    if len(neighbor_pairs) != 2:
        return None

    paired_nodes = {node for pair in neighbor_pairs for node in pair}
    if paired_nodes != set(neighbors):
        return None

    left_pair, right_pair = sorted(
        [tuple(sorted(pair, key=str)) for pair in neighbor_pairs],
        key=lambda pair: str(pair),
    )

    return {
        shared: np.array([0.0, 0.0]),
        left_pair[0]: np.array([-0.82, 0.48]),
        left_pair[1]: np.array([-0.82, -0.48]),
        right_pair[0]: np.array([0.82, 0.48]),
        right_pair[1]: np.array([0.82, -0.48]),
    }


def simple_contraction_graph(G, nodes):
    simple_graph = nx.Graph()
    simple_graph.add_nodes_from(nodes)
    simple_graph.add_edges_from(contraction_edges(G))
    return simple_graph


def graph_bridges(simple_graph):
    bridges = []

    for u, v in simple_graph.edges():
        test_graph = simple_graph.copy()
        test_graph.remove_edge(u, v)

        if not nx.has_path(test_graph, u, v):
            bridges.append((u, v))

    return bridges


def cycle_components_after_removing_bridges(simple_graph, bridges):
    non_bridge_graph = simple_graph.copy()
    non_bridge_graph.remove_edges_from(bridges)

    cycle_components = []
    for component in nx.connected_components(non_bridge_graph):
        component = set(component)
        subgraph = non_bridge_graph.subgraph(component)

        if len(component) >= 3 and subgraph.number_of_edges() >= len(component):
            cycle_components.append(component)

    return cycle_components


def component_attachment_nodes(simple_graph, component):
    return [
        node for node in component
        if any(neighbor not in component for neighbor in simple_graph.neighbors(node))
    ]


def bridge_path_layout(G, nodes):
    simple_graph = simple_contraction_graph(G, nodes)

    if not nx.is_connected(simple_graph):
        return None

    bridges = graph_bridges(simple_graph)
    if not bridges:
        return None

    cycle_components = cycle_components_after_removing_bridges(simple_graph, bridges)
    if len(cycle_components) != 2:
        return None

    left_component, right_component = sorted(
        cycle_components,
        key=lambda component: min(str(node) for node in component),
    )
    left_attachments = component_attachment_nodes(simple_graph, left_component)
    right_attachments = component_attachment_nodes(simple_graph, right_component)

    if len(left_attachments) != 1 or len(right_attachments) != 1:
        return None

    left_anchor = left_attachments[0]
    right_anchor = right_attachments[0]

    bridge_graph = nx.Graph()
    bridge_graph.add_edges_from(bridges)

    if not nx.has_path(bridge_graph, left_anchor, right_anchor):
        return None

    path = nx.shortest_path(bridge_graph, left_anchor, right_anchor)
    path_edges = {tuple(sorted(edge, key=str)) for edge in zip(path, path[1:])}
    bridge_edges = {tuple(sorted(edge, key=str)) for edge in bridge_graph.edges()}

    if path_edges != bridge_edges:
        return None

    cycle_nodes = left_component | right_component
    path_middle = set(path[1:-1])
    if set(nodes) != cycle_nodes | path_middle:
        return None

    return bow_tie_path_positions(path, left_component, right_component)


def add_cycle_lobe_positions(pos, component, anchor, side, lobe_width=0.78, lobe_height=0.58):
    outer_nodes = sorted(component - {anchor}, key=str)
    count = len(outer_nodes)

    if count == 0:
        return

    if count == 1:
        pos[outer_nodes[0]] = pos[anchor] + np.array([side * lobe_width, 0.0])
        return

    if count == 2:
        pos[outer_nodes[0]] = pos[anchor] + np.array([side * lobe_width, lobe_height])
        pos[outer_nodes[1]] = pos[anchor] + np.array([side * lobe_width, -lobe_height])
        return

    angles = np.linspace(70, -70, count)
    for node, angle in zip(outer_nodes, angles):
        radians = np.deg2rad(angle)
        pos[node] = pos[anchor] + np.array([
            side * lobe_width * np.cos(radians),
            lobe_height * np.sin(radians),
        ])


def bow_tie_path_positions(path, left_component, right_component, spacing=0.42):
    center_offset = (len(path) - 1) / 2
    pos = {
        node: np.array([(index - center_offset) * spacing, 0.0])
        for index, node in enumerate(path)
    }

    add_cycle_lobe_positions(pos, left_component, path[0], side=-1)
    add_cycle_lobe_positions(pos, right_component, path[-1], side=1)
    return pos


def contraction_edges(G):
    return [
        (u, v) for u, v, data in G.edges(data=True)
        if data.get("kind") != "dangling" and u != v
    ]


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def count_edge_crossings(G, pos):
    crossings = 0
    edges = contraction_edges(G)

    for (u1, v1), (u2, v2) in itertools.combinations(edges, 2):
        if len({u1, v1, u2, v2}) < 4:
            continue
        if segments_cross(pos[u1], pos[v1], pos[u2], pos[v2]):
            crossings += 1

    return crossings


def ordered_square_layout(G, nodes):
    if len(nodes) != 4:
        return square_layout(nodes)

    first, *rest = nodes
    best_layout = None
    best_crossings = None

    for ordered_nodes in ((first, *permutation) for permutation in itertools.permutations(rest)):
        pos = square_layout(ordered_nodes)
        crossings = count_edge_crossings(G, pos)

        if best_crossings is None or crossings < best_crossings:
            best_layout = pos
            best_crossings = crossings

    return best_layout


def graph_layout(G, real_nodes):
    if len(real_nodes) == 2:
        return two_node_layout(real_nodes)
    if len(real_nodes) == 4:
        return ordered_square_layout(G, real_nodes)
    bridge_layout = bridge_path_layout(G, real_nodes)
    if bridge_layout is not None:
        return bridge_layout
    shared_layout = shared_triangle_layout(G, real_nodes)
    if shared_layout is not None:
        return shared_layout
    return polygon_layout(real_nodes)


def parallel_offsets(count, spacing=PARALLEL_EDGE_SPACING, max_span=MAX_PARALLEL_EDGE_SPAN):
    if count <= 1:
        return np.array([0.0])

    span = min(spacing * (count - 1), max_span)
    return np.linspace(-span / 2, span / 2, count)


def add_parallel_dangling_positions(
    G,
    pos,
    stub_nodes,
    length=0.42,
    spacing=PARALLEL_EDGE_SPACING,
    node_gap=0.0,
):
    """
    Put dangling endpoints outside the graph and side-by-side per anchor.
    Returns short edge segments that are parallel when several stubs share an anchor.
    """
    if len(pos) == 1:
        center = np.array([0.0, 0.0])
    else:
        center = np.mean(np.array(list(pos.values())), axis=0)

    grouped_stubs = {}

    for stub in stub_nodes:
        anchor = next(G.neighbors(stub))
        grouped_stubs.setdefault(anchor, []).append(stub)

    segments = []

    for anchor, stubs in grouped_stubs.items():
        outward = pos[anchor] - center
        outward_norm = np.linalg.norm(outward)

        if outward_norm == 0:
            outward = np.array([0.0, 1.0])
        else:
            outward = outward / outward_norm

        tangent = np.array([-outward[1], outward[0]])
        offsets = parallel_offsets(len(stubs), spacing=spacing)

        for stub, offset in zip(stubs, offsets):
            start = pos[anchor] + node_gap * outward + offset * tangent
            end = start + length * outward

            pos[stub] = end
            segments.append((start, end))

    return segments


def normal_edge_segments(G, pos, spacing=PARALLEL_EDGE_SPACING):
    grouped_edges = {}

    for u, v, data in G.edges(data=True):
        if data.get("kind") == "dangling":
            continue

        edge_key = tuple(sorted((u, v), key=str))
        grouped_edges.setdefault(edge_key, []).append((u, v))

    segments = []

    for edges in grouped_edges.values():
        offsets = parallel_offsets(len(edges), spacing=spacing)

        for (u, v), offset in zip(edges, offsets):
            start = pos[u]
            end = pos[v]
            direction = end - start
            direction_norm = np.linalg.norm(direction)

            if direction_norm == 0:
                tangent = np.array([1.0, 0.0])
            else:
                direction = direction / direction_norm
                tangent = np.array([-direction[1], direction[0]])

            shift = offset * tangent
            segments.append((start + shift, end + shift))

    return segments


def colors_by_rank(G, nodes):
    rank_palette = {
        0: "#e5e7eb",
        1: "#93c5fd",
        2: "#5eead4",
        3: "#fbbf24",
        4: "#f87171",
        5: "#c084fc",
    }
    fallback_color = "#d1d5db"

    return [
        rank_palette.get(G.nodes[node].get("rank"), fallback_color)
        for node in nodes
    ]


def draw_segments(ax, segments, color, linewidth, linestyle="-"):
    for start, end in segments:
        ax.add_line(
            Line2D(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                solid_capstyle="round",
            )
        )


def draw_graph_example(ax, G, title):
    real_nodes, stub_nodes = real_and_stub_nodes(G)
    for node in real_nodes:
        G.nodes[node].setdefault("rank", G.degree(node))

    pos = graph_layout(G, real_nodes)
    dangling_segments = add_parallel_dangling_positions(G, pos, stub_nodes)
    coordinates = np.array(list(pos.values()))

    draw_segments(
        ax,
        normal_edge_segments(G, pos),
        color="#374151",
        linewidth=2.6,
    )
    draw_segments(
        ax,
        dangling_segments,
        color="#64748b",
        linewidth=2.2,
        linestyle=(0, (4, 4)),
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=real_nodes,
        node_size=1350,
        node_color=colors_by_rank(G, real_nodes),
        edgecolors="#1f2937",
        linewidths=2.3,
        ax=ax,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={node: G.nodes[node].get("rank", G.degree(node)) for node in real_nodes},
        font_size=15,
        font_weight="bold",
        font_color="#111827",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=stub_nodes,
        node_size=60,
        node_color="#64748b",
        edgecolors="white",
        linewidths=1.0,
        ax=ax,
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    if coordinates.size:
        mins = coordinates.min(axis=0)
        maxs = coordinates.max(axis=0)
        center = (mins + maxs) / 2
        span = np.maximum(maxs - mins, 1.05)
        padded_span = span * 1.28

        ax.set_xlim(center[0] - padded_span[0] / 2, center[0] + padded_span[0] / 2)
        ax.set_ylim(center[1] - padded_span[1] / 2, center[1] + padded_span[1] / 2)


def one_node_two_dangling_edges():
    G = nx.MultiGraph()
    G.add_node("A")
    add_dangling_edge(G, "A")
    add_dangling_edge(G, "A")
    set_rank_from_degree(G)
    return G


def square_with_double_and_single_edges():
    G = nx.MultiGraph()
    G.add_nodes_from(["A", "B", "C", "D"])
    G.add_edge("A", "B")
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    G.add_edge("C", "D")
    G.add_edge("D", "A")
    set_rank_from_degree(G)
    return G


def triangle_with_double_edge_and_one_dangling_edge():
    G = nx.MultiGraph()
    G.add_nodes_from(["A", "B", "C"])
    G.add_edges_from([
        ("A", "B"),
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
    ])
    add_dangling_edge(G, "C")
    set_rank_from_degree(G)
    return G


def plot_demo_examples():
    examples = [
        ("One node, two dangling edges", one_node_two_dangling_edges()),
        ("Square: double and single edges", square_with_double_and_single_edges()),
        ("Triangle: double edge and dangling edge", triangle_with_double_edge_and_one_dangling_edge()),
    ]

    fig, axes = plt.subplots(1, len(examples), figsize=(12, 4))

    for ax, (title, graph) in zip(axes, examples):
        draw_graph_example(ax, graph, title)

    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    plot_demo_examples()
    plt.show()
