"""
test_algorithms.py - Verification test suite for algorithms.py
"""

import unittest
from algorithms import (
    DEFAULT_ROADS,
    PLACES,
    make_graph,
    dijkstra_search,
    bellman_ford_search,
    rebuild_path,
    calculate_path_time,
    generate_random_connected_graph
)

class TestEmergencyRouting(unittest.TestCase):
    def setUp(self):
        self.city = make_graph(DEFAULT_ROADS)

    def test_dijkstra_default(self):
        """Expected: H->X is H-B-D-E-X in 15 min; Dijkstra uses 12 relaxations."""
        best, parent, relax_count, log = dijkstra_search(self.city, 'H', keep_log=True)
        path = rebuild_path(parent, 'X')
        
        self.assertEqual(path, ['H', 'B', 'D', 'E', 'X'])
        self.assertEqual(best['X'], 15)
        self.assertEqual(calculate_path_time(path, self.city), 15)
        self.assertEqual(relax_count, 12)
        self.assertEqual(len(log), 8)  # All 8 vertices settled

    def test_bellman_ford_default(self):
        """Expected: H->X is H-B-D-E-X in 15 min; Bellman-Ford uses 72 relaxations."""
        best, parent, relax_count = bellman_ford_search(self.city, 'H')
        path = rebuild_path(parent, 'X')

        self.assertEqual(path, ['H', 'B', 'D', 'E', 'X'])
        self.assertEqual(best['X'], 15)
        self.assertEqual(relax_count, 72)

    def test_d_e_closed_road(self):
        """Expected with D-E closed: H-B-A-C-E-X in 16 min."""
        city_closed = make_graph(DEFAULT_ROADS, blocked=[('D', 'E')])
        
        # Dijkstra
        d_best, d_parent, _, _ = dijkstra_search(city_closed, 'H')
        d_path = rebuild_path(d_parent, 'X')
        self.assertEqual(d_path, ['H', 'B', 'A', 'C', 'E', 'X'])
        self.assertEqual(d_best['X'], 16)
        
        # Bellman-Ford
        bf_best, bf_parent, _ = bellman_ford_search(city_closed, 'H')
        bf_path = rebuild_path(bf_parent, 'X')
        self.assertEqual(bf_path, ['H', 'B', 'A', 'C', 'E', 'X'])
        self.assertEqual(bf_best['X'], 16)

    def test_negative_weight_demo(self):
        """Test directed graph A->B 2, A->C 5, C->B -4."""
        neg_graph = {
            'A': {'B': 2, 'C': 5},
            'B': {},
            'C': {'B': -4}
        }
        d_best, _, _, _ = dijkstra_search(neg_graph, 'A')
        bf_best, _, _ = bellman_ford_search(neg_graph, 'A')
        
        # Dijkstra gives 2 (fails to update settled B)
        self.assertEqual(d_best['B'], 2)
        # Bellman-Ford gives 1 (5 + (-4) = 1)
        self.assertEqual(bf_best['B'], 1)

    def test_negative_cycle_detection(self):
        """Bellman-Ford must detect negative cycle reachable from source."""
        cycle_graph = {
            'A': {'B': 1},
            'B': {'C': -2},
            'C': {'A': -1}
        }
        with self.assertRaises(ValueError):
            bellman_ford_search(cycle_graph, 'A')

    def test_unreachable_node(self):
        """Test path reconstruction when destination is disconnected."""
        isolated_graph = {
            'H': {'A': 4},
            'A': {'H': 4},
            'X': {}
        }
        best, parent, _, _ = dijkstra_search(isolated_graph, 'H')
        self.assertEqual(best['X'], float('inf'))
        path = rebuild_path(parent, 'X', start='H')
        self.assertEqual(path, [])
        self.assertIsNone(calculate_path_time(path, isolated_graph))

    def test_random_graph_generation(self):
        """Check random graph generator produces valid connected graphs."""
        g = generate_random_connected_graph(50, edge_factor=1.5, seed=42)
        self.assertEqual(len(g), 50)
        best, _, _, _ = dijkstra_search(g, 0)
        # All reachable because connected
        for d in best.values():
            self.assertNotEqual(d, float('inf'))

if __name__ == '__main__':
    unittest.main()
