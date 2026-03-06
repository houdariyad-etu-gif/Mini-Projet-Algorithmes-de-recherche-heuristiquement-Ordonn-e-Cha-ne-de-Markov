"""
markov.py - Construction et analyse de la chaîne de Markov induite par une politique
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


# États spéciaux
GOAL_STATE = "GOAL"
FAIL_STATE = "FAIL"

DIRECTIONS = {
    (-1, 0): "UP",
    (1, 0):  "DOWN",
    (0, -1): "LEFT",
    (0, 1):  "RIGHT"
}

DIR_LATERAL = {
    (-1, 0): [(0, -1), (0, 1)],   # UP -> latéraux: LEFT, RIGHT
    (1, 0):  [(0, -1), (0, 1)],   # DOWN -> latéraux: LEFT, RIGHT
    (0, -1): [(-1, 0), (1, 0)],   # LEFT -> latéraux: UP, DOWN
    (0, 1):  [(-1, 0), (1, 0)],   # RIGHT -> latéraux: UP, DOWN
}


def build_policy_from_path(path: List[Tuple[int,int]]) -> Dict[Tuple[int,int], Tuple[int,int]]:
    """
    Construit la politique (état -> action) à partir du chemin planifié par A*.
    Chaque état du chemin (sauf le but) est associé au mouvement vers le prochain état.
    """
    policy = {}
    for i in range(len(path) - 1):
        state = path[i]
        next_state = path[i+1]
        action = (next_state[0] - state[0], next_state[1] - state[1])
        policy[state] = action
    return policy


def is_valid(state: Tuple[int,int], grid: List[List[int]]) -> bool:
    rows, cols = len(grid), len(grid[0])
    r, c = state
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 1


def build_transition_matrix(
    grid: List[List[int]],
    path: List[Tuple[int,int]],
    goal: Tuple[int,int],
    epsilon: float = 0.1,
    include_fail: bool = True
) -> Tuple[np.ndarray, List]:
    """
    Construit la matrice de transition P de la chaîne de Markov.
    
    - États: tous les états du chemin + GOAL + (optionnel) FAIL
    - Transition stochastique avec paramètre epsilon:
        * action voulue: probabilité 1 - epsilon
        * déviation latérale: epsilon/2 chacune
        * si collision/obstacle: reste sur place
    
    Retourne: (P, states_list)
    """
    policy = build_policy_from_path(path)
    
    # Construction de la liste d'états
    path_states = list(dict.fromkeys(path))  # chemin sans doublons, ordre préservé
    
    states = path_states.copy()
    if goal not in states:
        states.append(goal)
    
    goal_idx = states.index(goal)
    
    fail_idx = None
    if include_fail:
        states.append(FAIL_STATE)
        fail_idx = len(states) - 1
    
    n = len(states)
    state_idx = {s: i for i, s in enumerate(states)}
    
    P = np.zeros((n, n))
    
    for i, s in enumerate(states):
        # État absorbant GOAL
        if s == goal:
            P[i, i] = 1.0
            continue
        # État absorbant FAIL
        if s == FAIL_STATE:
            P[i, i] = 1.0
            continue
        
        # État dans le chemin avec politique définie
        if s in policy:
            action = policy[s]
            laterals = DIR_LATERAL[action]
            
            # Action voulue
            next_intended = (s[0] + action[0], s[1] + action[1])
            
            # Transitions possibles: (état_cible, probabilité_brute)
            transitions = [(next_intended, 1.0 - epsilon)]
            for lat in laterals:
                next_lat = (s[0] + lat[0], s[1] + lat[1])
                transitions.append((next_lat, epsilon / 2.0))
            
            for (ns, prob) in transitions:
                if prob == 0:
                    continue
                if ns == goal:
                    P[i, goal_idx] += prob
                elif is_valid(ns, grid):
                    if ns in state_idx:
                        P[i, state_idx[ns]] += prob
                    else:
                        # État valide mais hors chemin -> reste sur place (ou FAIL)
                        if include_fail and epsilon > 0:
                            P[i, fail_idx] += prob
                        else:
                            P[i, i] += prob
                else:
                    # Collision/obstacle -> reste sur place
                    P[i, i] += prob
        else:
            # État sans politique (fin de chemin mais pas goal) -> reste sur place
            P[i, i] = 1.0
        
        # Normalisation (sécurité numérique)
        row_sum = P[i].sum()
        if row_sum > 0:
            P[i] /= row_sum
    
    return P, states


def verify_stochastic(P: np.ndarray, tol: float = 1e-8) -> bool:
    """Vérifie que P est bien stochastique (lignes somment à 1)."""
    row_sums = P.sum(axis=1)
    return np.allclose(row_sums, 1.0, atol=tol)


def compute_distribution(pi0: np.ndarray, P: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Calcule la distribution au temps n: π(n) = π(0) @ P^n
    Utilise la multiplication itérative pour éviter les erreurs numériques.
    """
    pi = pi0.copy()
    for _ in range(n_steps):
        pi = pi @ P
    return pi


def compute_pn(P: np.ndarray, n: int) -> np.ndarray:
    """Calcule P^n par exponentiation matricielle."""
    return np.linalg.matrix_power(P, n)


def goal_probability_over_time(pi0: np.ndarray, P: np.ndarray, goal_idx: int, max_steps: int = 50):
    """Retourne P(Xn = GOAL) pour n = 0, ..., max_steps."""
    probs = []
    pi = pi0.copy()
    for _ in range(max_steps + 1):
        probs.append(pi[goal_idx])
        pi = pi @ P
    return np.array(probs)


def compute_absorption(P: np.ndarray, states: List, goal_idx: int, fail_idx: Optional[int] = None):
    """
    Calcule les probabilités d'absorption et le temps moyen d'absorption.
    
    Décomposition canonique: P = [[I, 0], [R, Q]]
    N = (I - Q)^{-1} : matrice fondamentale
    B = N @ R : probabilités d'absorption par état absorbant
    t = N @ 1 : temps moyen avant absorption
    """
    n = len(states)
    absorbing = [goal_idx]
    if fail_idx is not None:
        absorbing.append(fail_idx)
    
    transient = [i for i in range(n) if i not in absorbing]
    
    if not transient:
        return None
    
    # Sous-matrices
    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]
    
    # Matrice fondamentale N = (I - Q)^{-1}
    I_Q = np.eye(len(transient)) - Q
    try:
        N = np.linalg.inv(I_Q)
    except np.linalg.LinAlgError:
        return None
    
    # Probabilités d'absorption
    B = N @ R
    
    # Temps moyen avant absorption
    t_mean = N @ np.ones(len(transient))
    
    absorbing_names = [states[i] if not isinstance(states[i], tuple) else f"({states[i][0]},{states[i][1]})"
                       for i in absorbing]
    transient_names = [f"({states[i][0]},{states[i][1]})" for i in transient]
    
    return {
        "N": N,
        "B": B,
        "t_mean": t_mean,
        "transient_states": transient_names,
        "absorbing_states": absorbing_names,
        "transient_indices": transient,
        "absorbing_indices": absorbing
    }


def identify_classes(P: np.ndarray, states: List):
    """
    Identifie les classes de communication par DFS sur le graphe des transitions.
    Retourne: (classes, transient_states, recurrent_states)
    """
    n = len(states)
    
    # Construction du graphe d'accessibilité
    def reachable(i):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for j in range(n):
                if P[node, j] > 1e-10 and j not in visited:
                    stack.append(j)
        return visited
    
    # Composantes fortement connexes (algorithme de Kosaraju simplifié)
    visited_global = set()
    finish_order = []
    
    def dfs_forward(i):
        stack = [(i, False)]
        while stack:
            node, returning = stack.pop()
            if returning:
                finish_order.append(node)
                continue
            if node in visited_global:
                continue
            visited_global.add(node)
            stack.append((node, True))
            for j in range(n):
                if P[node, j] > 1e-10 and j not in visited_global:
                    stack.append((j, False))
    
    for i in range(n):
        if i not in visited_global:
            dfs_forward(i)
    
    # Graphe transposé
    PT = P.T
    visited2 = set()
    sccs = []
    
    def dfs_reverse(i):
        comp = []
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited2:
                continue
            visited2.add(node)
            comp.append(node)
            for j in range(n):
                if PT[node, j] > 1e-10 and j not in visited2:
                    stack.append(j)
        return comp
    
    for i in reversed(finish_order):
        if i not in visited2:
            comp = dfs_reverse(i)
            sccs.append(comp)
    
    # Classification: récurrent vs transitoire
    classes_info = []
    for comp in sccs:
        comp_set = set(comp)
        is_closed = True
        for i in comp:
            for j in range(n):
                if P[i, j] > 1e-10 and j not in comp_set:
                    is_closed = False
                    break
        
        state_names = []
        for i in comp:
            s = states[i]
            if isinstance(s, tuple):
                state_names.append(f"({s[0]},{s[1]})")
            else:
                state_names.append(str(s))
        
        classes_info.append({
            "states": state_names,
            "indices": comp,
            "is_recurrent": is_closed,
            "type": "Récurrent (persistant)" if is_closed else "Transitoire"
        })
    
    return classes_info
