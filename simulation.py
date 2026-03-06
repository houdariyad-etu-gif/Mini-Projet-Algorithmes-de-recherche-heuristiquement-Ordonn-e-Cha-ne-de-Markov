"""
simulation.py - Simulation Monte-Carlo des trajectoires de la chaîne de Markov
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def simulate_trajectory(
    P: np.ndarray,
    states: List,
    start_idx: int,
    goal_idx: int,
    fail_idx: Optional[int] = None,
    max_steps: int = 200,
    rng: Optional[np.random.Generator] = None
) -> Dict:
    """
    Simule une trajectoire Markov depuis l'état de départ.
    Retourne: {reached_goal, reached_fail, steps, trajectory}
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(states)
    current = start_idx
    trajectory = [current]
    
    for step in range(max_steps):
        if current == goal_idx:
            return {"reached_goal": True, "reached_fail": False, "steps": step, "trajectory": trajectory}
        if fail_idx is not None and current == fail_idx:
            return {"reached_goal": False, "reached_fail": True, "steps": step, "trajectory": trajectory}
        
        # Tirage de la prochaine cellule selon P[current]
        next_state = rng.choice(n, p=P[current])
        current = next_state
        trajectory.append(current)
    
    # Pas absorbé dans max_steps
    reached_goal = (current == goal_idx)
    reached_fail = (fail_idx is not None and current == fail_idx)
    return {
        "reached_goal": reached_goal,
        "reached_fail": reached_fail,
        "steps": max_steps,
        "trajectory": trajectory
    }


def monte_carlo(
    P: np.ndarray,
    states: List,
    start_idx: int,
    goal_idx: int,
    fail_idx: Optional[int] = None,
    N: int = 5000,
    max_steps: int = 200,
    seed: int = 42
) -> Dict:
    """
    Simulation Monte-Carlo de N trajectoires.
    
    Retourne:
    - prob_goal: probabilité empirique d'atteindre GOAL
    - prob_fail: probabilité empirique d'atteindre FAIL
    - mean_time_goal: temps moyen d'atteinte de GOAL (sur les trajectoires réussies)
    - std_time_goal: écart-type du temps d'atteinte
    - steps_distribution: distribution des temps d'atteinte
    """
    rng = np.random.default_rng(seed)
    
    n_goal = 0
    n_fail = 0
    steps_goal = []
    steps_fail = []
    
    for _ in range(N):
        result = simulate_trajectory(P, states, start_idx, goal_idx, fail_idx, max_steps, rng)
        if result["reached_goal"]:
            n_goal += 1
            steps_goal.append(result["steps"])
        elif result["reached_fail"]:
            n_fail += 1
            steps_fail.append(result["steps"])
    
    prob_goal = n_goal / N
    prob_fail = n_fail / N
    prob_timeout = 1.0 - prob_goal - prob_fail
    
    mean_time_goal = np.mean(steps_goal) if steps_goal else float('nan')
    std_time_goal = np.std(steps_goal) if steps_goal else float('nan')
    
    return {
        "N": N,
        "prob_goal": prob_goal,
        "prob_fail": prob_fail,
        "prob_timeout": prob_timeout,
        "mean_time_goal": mean_time_goal,
        "std_time_goal": std_time_goal,
        "steps_goal": np.array(steps_goal),
        "steps_fail": np.array(steps_fail),
        "n_goal": n_goal,
        "n_fail": n_fail
    }


def compare_mc_vs_matrix(
    P: np.ndarray,
    pi0: np.ndarray,
    goal_idx: int,
    max_steps: int = 50,
    N: int = 5000,
    seed: int = 42
) -> Dict:
    """
    Compare simulation Monte-Carlo vs calcul matriciel π(n) = π(0) P^n.
    Retourne les probabilités P(Xn=GOAL) pour les deux méthodes.
    """
    from markov import goal_probability_over_time
    
    # Calcul matriciel
    matrix_probs = goal_probability_over_time(pi0, P, goal_idx, max_steps)
    
    # Simulation MC
    states = list(range(len(pi0)))
    rng = np.random.default_rng(seed)
    start_idx = np.argmax(pi0)
    
    mc_probs = []
    for step in range(max_steps + 1):
        count = 0
        for _ in range(N):
            result = simulate_trajectory(P, states, start_idx, goal_idx, None, step, rng)
            if result["reached_goal"] or (step > 0 and result["steps"] < step):
                count += 1
        mc_probs.append(count / N)
    
    return {
        "matrix_probs": matrix_probs,
        "mc_probs": np.array(mc_probs),
        "steps": np.arange(max_steps + 1)
    }
