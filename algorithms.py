"""
algorithms.py - Core graph algorithms and helpers for the DSA-II Case Study:
'Fastest Routes for Emergency Vehicles: Dijkstra's Algorithm vs Bellman-Ford on a City Road Map'
"""

import heapq
import random
from typing import Dict, List, Tuple, Set, Optional, Any

# Canonical Places and Labels as specified in the case study
PLACES: List[str] = ['H', 'F', 'A', 'B', 'C', 'D', 'E', 'X']

PLACE_LABELS: Dict[str, str] = {
    'H': 'Hospital (H)',
    'F': 'Fire Station (F)',
    'A': 'Junction A',
    'B': 'Junction B',
    'C': 'Junction C',
    'D': 'Junction D',
    'E': 'Junction E',
    'X': 'Accident Site (X)',
}

# Undirected road network with travel times in minutes
DEFAULT_ROADS: List[Tuple[str, str, int]] = [
    ('H', 'A', 4),
    ('H', 'B', 2),
    ('A', 'B', 1),
    ('A', 'C', 5),
    ('B', 'D', 8),
    ('B', 'F', 6),
    ('F', 'D', 3),
    ('C', 'D', 2),
    ('C', 'E', 6),
    ('D', 'E', 3),
    ('D', 'X', 9),
    ('E', 'X', 2),
]

# Fixed coordinates matching the report layout:
# Hospital left, A & C top, B & D middle, E right, Accident far right, Fire Station bottom
FIXED_LAYOUT: Dict[str, Tuple[float, float]] = {
    'H': (0.8, 2.0),
    'A': (2.4, 3.4),
    'B': (2.4, 1.8),
    'F': (3.6, 0.5),
    'C': (4.8, 3.4),
    'D': (4.8, 1.8),
    'E': (6.6, 2.9),
    'X': (8.4, 1.9),
}

# Color palette adhering to specifications
NODE_THEME: Dict[str, str] = {
    'H': '#00897B',  # Teal (Hospital)
    'F': '#8E24AA',  # Purple (Fire Station)
    'X': '#E53935',  # Red (Accident Site)
    'A': '#455A64',  # Slate Grey (Junctions)
    'B': '#455A64',
    'C': '#455A64',
    'D': '#455A64',
    'E': '#455A64',
}

PATH_COLOR = '#FFB300'       # Amber for shortest path
EDGE_COLOR = '#B0BEC5'       # Neutral soft grey for regular roads
CLOSED_ROAD_COLOR = '#D32F2F' # Red for blocked roads


def make_graph(
    roads: List[Tuple[str, str, int]],
    blocked: Optional[Set[Tuple[str, str]]] = None,
    places: Optional[List[str]] = None
) -> Dict[str, Dict[str, int]]:
    """
    Constructs an undirected adjacency dict representation of the city graph,
    ignoring any road present in the `blocked` set.
    """
    blocked_normalized = set()
    if blocked:
        for b in blocked:
            if isinstance(b, (tuple, list)) and len(b) >= 2:
                blocked_normalized.add((b[0], b[1]))
                blocked_normalized.add((b[1], b[0]))

    # Use standard place ordering if given, otherwise collect from roads
    node_order = list(places) if places is not None else list(PLACES)
    for u, v, _ in roads:
        if u not in node_order:
            node_order.append(u)
        if v not in node_order:
            node_order.append(v)

    city: Dict[str, Dict[str, int]] = {n: {} for n in node_order}

    for u, v, w in roads:
        if (u, v) not in blocked_normalized and (v, u) not in blocked_normalized:
            city[u][v] = w
            city[v][u] = w

    return city


def dijkstra_search(
    city: Dict[str, Dict[str, int]],
    start: str,
    keep_log: bool = False
) -> Tuple[Dict[str, float], Dict[str, Optional[str]], int, List[Dict[str, Any]]]:
    """
    Dijkstra's Single-Source Shortest Path using a min-heap (heapq).
    Skips settled nodes when relaxing adjacent edges.
    Logs the distance table after each settled node if keep_log=True.

    Returns:
        best: dict mapping node -> minimum distance from start
        parent: dict mapping node -> previous node on optimal path
        relax_count: total edge relaxation attempts
        log: list of step snapshots containing distance table state
    """
    best: Dict[str, float] = {node: float('inf') for node in city}
    parent: Dict[str, Optional[str]] = {node: None for node in city}
    best[start] = 0.0

    # Min-heap stores tuples of (tentative_distance, node)
    pq: List[Tuple[float, str]] = [(0.0, start)]
    settled: Set[str] = set()
    relax_count: int = 0
    log: List[Dict[str, Any]] = []
    step: int = 0

    while pq:
        d, u = heapq.heappop(pq)
        if u in settled:
            continue
        settled.add(u)
        step += 1

        for v, w in city[u].items():
            if v in settled:
                # Per case study rules, do not relax edges to already settled vertices
                continue
            relax_count += 1
            if d + w < best[v]:
                best[v] = d + w
                parent[v] = u
                heapq.heappush(pq, (best[v], v))

        if keep_log:
            snapshot = {
                'Step': step,
                'Settled Node': u,
                'Settled Dist': d,
                'Settled Nodes Set': set(settled),
            }
            for n in city:
                snapshot[n] = best[n] if best[n] != float('inf') else float('inf')
            log.append(snapshot)

    return best, parent, relax_count, log


def bellman_ford_search(
    city: Dict[str, Dict[str, int]],
    start: str
) -> Tuple[Dict[str, float], Dict[str, Optional[str]], int]:
    """
    Bellman-Ford algorithm with early-exit and a final negative-cycle check.
    Supports both undirected and directed graphs.

    Returns:
        best: dict mapping node -> minimum distance from start
        parent: dict mapping node -> previous node on optimal path
        relax_count: total edge relaxation attempts
    """
    best: Dict[str, float] = {node: float('inf') for node in city}
    parent: Dict[str, Optional[str]] = {node: None for node in city}
    best[start] = 0.0
    relax_count: int = 0

    # Flatten edges in stable node iteration order
    edges: List[Tuple[str, str, int]] = []
    for u in city:
        for v, w in city[u].items():
            edges.append((u, v, w))

    num_v = len(city)
    early_exit = False

    # Perform up to |V| - 1 relaxation sweeps
    for _ in range(max(1, num_v - 1)):
        updated = False
        for u, v, w in edges:
            relax_count += 1
            if best[u] != float('inf') and best[u] + w < best[v]:
                best[v] = best[u] + w
                parent[v] = u
                updated = True
        if not updated:
            early_exit = True
            break

    # Final check for negative-weight cycles
    if not early_exit:
        for u, v, w in edges:
            if best[u] != float('inf') and best[u] + w < best[v]:
                raise ValueError("Graph contains a negative-weight cycle reachable from start node.")

    return best, parent, relax_count


def rebuild_path(
    parent: Dict[str, Optional[str]],
    goal: str,
    start: Optional[str] = None
) -> List[str]:
    """
    Reconstructs the shortest path from start to goal using the parent map.
    Returns empty list if goal is unreachable or not in parent.
    """
    if goal not in parent:
        return []

    # If start is specified and goal is not reachable from start
    if start is not None and goal != start and parent.get(goal) is None:
        return []

    path = []
    curr: Optional[str] = goal
    visited = set()

    while curr is not None:
        if curr in visited:  # Protect against cyclic references
            return []
        visited.add(curr)
        path.append(curr)
        curr = parent.get(curr)

    reconstructed = path[::-1]
    if start is not None and (not reconstructed or reconstructed[0] != start):
        return []
    return reconstructed


def calculate_path_time(path: List[str], city: Dict[str, Dict[str, int]]) -> Optional[int]:
    """
    Calculates total travel time along a reconstructed path.
    """
    if not path or len(path) < 2:
        return 0 if len(path) == 1 else None
    total = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if v not in city.get(u, {}):
            return None
        total += city[u][v]
    return total


def generate_random_connected_graph(
    num_nodes: int,
    edge_factor: float = 1.5,
    min_weight: int = 1,
    max_weight: int = 20,
    seed: Optional[int] = None
) -> Dict[int, Dict[int, int]]:
    """
    Generates a random connected undirected graph with `num_nodes` vertices
    and approximately `edge_factor * num_nodes` edges, with weights in [min_weight, max_weight].
    """
    if seed is not None:
        random.seed(seed)

    nodes = list(range(num_nodes))
    adj: Dict[int, Dict[int, int]] = {i: {} for i in nodes}

    # 1. Random Spanning Tree to guarantee connectedness
    connected = [0]
    unconnected = set(nodes[1:])
    while unconnected:
        u = random.choice(connected)
        v = random.choice(list(unconnected))
        unconnected.remove(v)
        connected.append(v)
        w = random.randint(min_weight, max_weight)
        adj[u][v] = w
        adj[v][u] = w

    # 2. Add extra edges up to target count
    target_edges = int(edge_factor * num_nodes)
    existing_edges = set((min(u, v), max(u, v)) for u in adj for v in adj[u])

    attempts = 0
    max_attempts = max(100, target_edges * 10)
    while len(existing_edges) < target_edges and attempts < max_attempts:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            edge_key = (min(u, v), max(u, v))
            if edge_key not in existing_edges:
                w = random.randint(min_weight, max_weight)
                adj[u][v] = w
                adj[v][u] = w
                existing_edges.add(edge_key)
        attempts += 1

    return adj
