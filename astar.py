"""
astar.py - Implémentation de A* et variantes (UCS, Greedy, A*) sur grille 2D
"""

import heapq
import time
import sys
from typing import List, Tuple, Dict, Optional, Callable


def manhattan(p: Tuple[int,int], goal: Tuple[int,int]) -> float:
    """Heuristique Manhattan (admissible pour coûts uniformes = 1)."""
    return abs(p[0] - goal[0]) + abs(p[1] - goal[1])


def euclidean(p: Tuple[int,int], goal: Tuple[int,int]) -> float:
    """Heuristique Euclidienne (admissible mais moins informée que Manhattan pour grilles 4-voisins)."""
    return ((p[0] - goal[0])**2 + (p[1] - goal[1])**2) ** 0.5


def zero_heuristic(p: Tuple[int,int], goal: Tuple[int,int]) -> float:
    """Heuristique nulle (équivalent à UCS)."""
    return 0.0


def get_neighbors(state: Tuple[int,int], grid: List[List[int]]) -> List[Tuple[Tuple[int,int], float]]:
    """Retourne les voisins accessibles (4-voisins) avec leur coût de transition."""
    rows, cols = len(grid), len(grid[0])
    x, y = state
    neighbors = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:
            cost = grid[nx][ny] if grid[nx][ny] > 1 else 1
            neighbors.append(((nx, ny), float(cost)))
    return neighbors


def reconstruct_path(came_from: Dict, current: Tuple[int,int]) -> List[Tuple[int,int]]:
    """Reconstruit le chemin depuis le dictionnaire came_from."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


def search(
    grid: List[List[int]],
    start: Tuple[int,int],
    goal: Tuple[int,int],
    heuristic: Callable = manhattan,
    weight_g: float = 1.0,
    weight_h: float = 1.0,
    algorithm_name: str = "A*"
) -> Dict:
    """
    Algorithme de recherche générique sur grille.
    
    - weight_g=1, weight_h=1  -> A*
    - weight_g=1, weight_h=0  -> UCS
    - weight_g=0, weight_h=1  -> Greedy Best-First
    - weight_g=1, weight_h=w  -> Weighted A* (w > 1)
    
    Retourne un dict avec: path, cost, nodes_expanded, open_max_size, time_ms
    """
    t_start = time.perf_counter()

    # OPEN: (f, g, state)
    open_heap = []
    h0 = heuristic(start, goal)
    heapq.heappush(open_heap, (weight_h * h0, 0.0, start))

    came_from: Dict = {}
    g_score: Dict[Tuple[int,int], float] = {start: 0.0}
    closed: set = set()

    nodes_expanded = 0
    open_max_size = 1

    while open_heap:
        open_max_size = max(open_max_size, len(open_heap))
        f, g, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            t_end = time.perf_counter()
            return {
                "found": True,
                "path": path,
                "cost": g_score[goal],
                "nodes_expanded": nodes_expanded,
                "open_max_size": open_max_size,
                "time_ms": (t_end - t_start) * 1000,
                "algorithm": algorithm_name
            }

        for neighbor, step_cost in get_neighbors(current, grid):
            if neighbor in closed:
                continue
            tentative_g = g_score[current] + step_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = heuristic(neighbor, goal)
                f_new = weight_g * tentative_g + weight_h * h
                heapq.heappush(open_heap, (f_new, tentative_g, neighbor))

    t_end = time.perf_counter()
    return {
        "found": False,
        "path": [],
        "cost": float('inf'),
        "nodes_expanded": nodes_expanded,
        "open_max_size": open_max_size,
        "time_ms": (t_end - t_start) * 1000,
        "algorithm": algorithm_name
    }


def astar(grid, start, goal, heuristic=manhattan):
    return search(grid, start, goal, heuristic=heuristic,
                  weight_g=1.0, weight_h=1.0, algorithm_name="A*")

def ucs(grid, start, goal):
    return search(grid, start, goal, heuristic=zero_heuristic,
                  weight_g=1.0, weight_h=0.0, algorithm_name="UCS")

def greedy(grid, start, goal, heuristic=manhattan):
    return search(grid, start, goal, heuristic=heuristic,
                  weight_g=0.0, weight_h=1.0, algorithm_name="Greedy")

def weighted_astar(grid, start, goal, w=2.0, heuristic=manhattan):
    return search(grid, start, goal, heuristic=heuristic,
                  weight_g=1.0, weight_h=w, algorithm_name=f"Weighted A* (w={w})")


def compare_algorithms(grid, start, goal):
    """Compare UCS, Greedy, A* et Weighted A* sur la même grille."""
    results = []
    results.append(ucs(grid, start, goal))
    results.append(greedy(grid, start, goal, heuristic=manhattan))
    results.append(astar(grid, start, goal, heuristic=zero_heuristic))
    results.append(astar(grid, start, goal, heuristic=manhattan))
    results.append(weighted_astar(grid, start, goal, w=2.0))
    return results
