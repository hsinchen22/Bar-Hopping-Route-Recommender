import numpy as np
import sqlite3
from itertools import combinations
from typing import List, Tuple, Dict
from barhopping.logger import logger

class PathFinder:
    def __init__(self, db_path: str):
        """Initialize the path finder.
        
        Args:
            db_path: Path to the SQLite database containing bar information
        """
        self.db_path = db_path
        
    def get_bar_addresses(self, bar_ids: List[int]) -> List[str]:
        """Get addresses for a list of bar IDs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Get all bars first
                cursor.execute("SELECT id, name, address FROM bars")
                all_bars = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
                
                # Map indices to actual bar IDs
                addresses = []
                for idx in bar_ids:
                    if idx in all_bars:
                        name, addr = all_bars[idx]
                        addresses.append(f"{name}, {addr}")
                    else:
                        logger.warning(f"Bar ID {idx} not found in database")
                        addresses.append("Unknown Location")
                        
                return addresses
        except Exception as e:
            logger.error(f"Error getting bar addresses: {str(e)}")
            return ["Unknown Location"] * len(bar_ids)
        
    def hamiltonian_path(self, dist_matrix: np.ndarray, n_bars: int, start: int) -> Tuple[float, List[int]]:
        """Find the shortest Hamiltonian path starting from a given point.
        
        Args:
            dist_matrix: Distance matrix between bars
            n_bars: Number of bars
            start: Index of the starting bar
            
        Returns:
            Tuple of (minimum distance, path)
        """
        dp = {}
        dp[(1 << start, start)] = (0, -1)

        for subset_len in range(2, n_bars + 1):
            for subset in combinations(range(n_bars), subset_len):
                if start not in subset:
                    continue  # ensure start is always included
                mask = sum(1 << k for k in subset)
                for k in subset:
                    prev_mask = mask & ~(1 << k)
                    res = []
                    for m in subset:
                        if m == k:
                            continue
                        if (prev_mask, m) in dp:
                            dist = dp[(prev_mask, m)][0] + dist_matrix[m][k]
                            res.append((dist, m))
                    if res:
                        dp[(mask, k)] = min(res)

        full_mask = (1 << n_bars) - 1
        candidates = [(dp[(full_mask, k)][0], k) for k in range(n_bars) if (full_mask, k) in dp]
        if not candidates:
            return float("inf"), []

        min_dist, u = min(candidates)

        # Reconstruct path
        path = [u]
        mask = full_mask
        while True:
            last = path[-1]
            _, v = dp[(mask, last)]
            if v == -1:
                break
            mask &= ~(1 << last)
            path.append(v)

        path.reverse()
        return min_dist, path
        
    def find_optimal_path(self, bar_ids: List[int], distances: Dict[Tuple[int, int], float]) -> Tuple[float, List[int]]:
        """Find the optimal path through a set of bars.
        
        Args:
            bar_ids: List of bar IDs to visit
            distances: Dictionary mapping (bar_id1, bar_id2) to distance
            
        Returns:
            Tuple of (total distance, list of bar IDs in optimal order)
        """
        n_bars = len(bar_ids)
        dist_matrix = np.zeros((n_bars, n_bars))
        
        # Fill distance matrix
        for i, j in combinations(range(n_bars), 2):
            id1, id2 = bar_ids[i], bar_ids[j]
            dist = distances.get((id1, id2), float("inf"))
            dist_matrix[i][j] = dist_matrix[j][i] = dist
            
        # Find optimal path starting from first bar
        min_dist, path = self.hamiltonian_path(dist_matrix, n_bars, 0)
        
        # Convert path indices back to bar IDs
        path_ids = [bar_ids[i] for i in path]
        
        return min_dist, path_ids 