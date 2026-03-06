"""
experiments.py - Expériences comparatives et génération des figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from grids import GRIDS
from astar import astar, ucs, greedy, weighted_astar, manhattan, euclidean, zero_heuristic, compare_algorithms
from markov import (build_transition_matrix, verify_stochastic, compute_distribution,
                     compute_pn, goal_probability_over_time, compute_absorption, identify_classes)
from simulation import monte_carlo, simulate_trajectory

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Utilitaire visualisation grille
# ──────────────────────────────────────────────────────────────

def plot_grid(grid, path, start, goal, title="", filename=None, ax=None):
    rows, cols = len(grid), len(grid[0])
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, cols*0.5), max(6, rows*0.5)))
    
    # Fond
    img = np.array([[grid[r][c] for c in range(cols)] for r in range(rows)], dtype=float)
    cmap = ListedColormap(['white', '#333333', '#FFE0B2'])
    img_display = np.where(img == 1, 1, 0)
    ax.imshow(img_display, cmap=ListedColormap(['white', '#444444']), vmin=0, vmax=1)
    
    # Grille
    for r in range(rows+1):
        ax.axhline(r-0.5, color='#CCCCCC', lw=0.5)
    for c in range(cols+1):
        ax.axvline(c-0.5, color='#CCCCCC', lw=0.5)
    
    # Chemin
    if path:
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, 'b-', lw=2.5, alpha=0.7, zorder=3)
        for p in path[1:-1]:
            ax.plot(p[1], p[0], 'o', color='#4FC3F7', ms=6, zorder=4)
    
    # Start et Goal
    ax.plot(start[1], start[0], 's', color='#43A047', ms=12, zorder=5, label='Départ')
    ax.plot(goal[1], goal[0], '*', color='#E53935', ms=14, zorder=5, label='But')
    
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols), fontsize=7)
    ax.set_yticklabels(range(rows), fontsize=7)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    
    if show and filename:
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120, bbox_inches='tight')
        plt.close()
        return os.path.join(OUTPUT_DIR, filename)


# ──────────────────────────────────────────────────────────────
# Expérience E1 : Comparaison algorithmes sur 3 grilles
# ──────────────────────────────────────────────────────────────

def experiment_E1():
    """Compare UCS vs Greedy vs A* sur les 3 grilles."""
    print("\n" + "="*60)
    print("EXPÉRIENCE E1 : Comparaison UCS / Greedy / A*")
    print("="*60)
    
    all_results = {}
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    
    for row_idx, (gname, (grid, start, goal)) in enumerate(GRIDS.items()):
        results = compare_algorithms(grid, start, goal)
        all_results[gname] = results
        
        algos = ["UCS", "Greedy (h=Manhattan)", "A* (h=0=UCS)", "A* (h=Manhattan)", "Weighted A* (w=2)"]
        
        print(f"\n  Grille {gname.upper()}:")
        print(f"  {'Algo':<22} {'Coût':>8} {'Nœuds':>8} {'Taille OPEN':>12} {'Temps(ms)':>10} {'Trouvé':>8}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*12} {'-'*10} {'-'*8}")
        for r in results:
            found_str = "Oui" if r["found"] else "Non"
            print(f"  {r['algorithm']:<22} {r['cost']:>8.1f} {r['nodes_expanded']:>8} {r['open_max_size']:>12} {r['time_ms']:>10.3f} {found_str:>8}")
        
        # Figures: colonnes = UCS, Greedy, A*
        for col_idx, algo_idx in enumerate([0, 1, 3]):  # UCS, Greedy, A* Manhattan
            r = results[algo_idx]
            plot_grid(grid, r["path"], start, goal,
                      title=f"{r['algorithm']}\nCoût={r['cost']:.0f} | Nœuds={r['nodes_expanded']}",
                      ax=axes[row_idx][col_idx])
    
    axes[0][0].set_ylabel("FACILE", fontsize=12, fontweight='bold', labelpad=15)
    axes[1][0].set_ylabel("MOYENNE", fontsize=12, fontweight='bold', labelpad=15)
    axes[2][0].set_ylabel("DIFFICILE", fontsize=12, fontweight='bold', labelpad=15)
    
    plt.suptitle("Comparaison UCS / Greedy / A* sur 3 grilles", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path_fig = os.path.join(OUTPUT_DIR, "E1_comparaison_algos.png")
    plt.savefig(path_fig, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure sauvegardée!")
    
    # Graphique barres comparatif
    fig, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for col_idx, (gname, results) in enumerate(all_results.items()):
        display_results = [results[i] for i in [0, 1, 3, 4]]  
        algos_labels = [r["algorithm"][:14] for r in display_results]
        nodes = [r["nodes_expanded"] for r in display_results]
        costs = [r["cost"] if r["found"] else 0 for r in display_results]

        colors = ['#EF5350', '#FFA726', '#42A5F5', '#AB47BC']
        bars = axes2[col_idx].bar(algos_labels, nodes, color=colors, alpha=0.85, edgecolor='black')
        axes2[col_idx].set_title(f"Grille {gname}", fontweight='bold')
        axes2[col_idx].set_ylabel("Nœuds développés")
        axes2[col_idx].tick_params(axis='x', rotation=30, labelsize=8)
        for bar, c in zip(bars, costs):
            if c > 0:
                axes2[col_idx].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                                    f'c={c:.0f}', ha='center', va='bottom', fontsize=7)
    
    plt.suptitle("Nœuds développés par algorithme", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path_fig2 = os.path.join(OUTPUT_DIR, "E1_noeuds_barres.png")
    plt.savefig(path_fig2, dpi=120, bbox_inches='tight')
    plt.close()
    
    return all_results


# ──────────────────────────────────────────────────────────────
# Expérience E2 : Impact de epsilon sur la robustesse Markov
# ──────────────────────────────────────────────────────────────

def experiment_E2():
    """Fixe A* sur la grille moyenne, varie epsilon, mesure P(GOAL)."""
    print("\n" + "="*60)
    print("EXPÉRIENCE E2 : Impact de ε sur la robustesse Markov")
    print("="*60)
    
    grid, start, goal = GRIDS["moyenne"]
    result_astar = astar(grid, start, goal, heuristic=manhattan)
    
    if not result_astar["found"]:
        print("  A* n'a pas trouvé de chemin!")
        return
    
    path = result_astar["path"]
    print(f"  Chemin A* : longueur={len(path)}, coût={result_astar['cost']}")
    
    epsilons = [0.0, 0.1, 0.2, 0.3]
    mc_results = {}
    matrix_probs_all = {}
    N_SIM = 3000
    MAX_STEPS = 80
    
    print(f"\n  {'ε':>6} {'P(GOAL) MC':>12} {'P(GOAL) Matrice':>18} {'Temps moyen':>14} {'P(FAIL)':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*18} {'-'*14} {'-'*10}")
    
    for eps in epsilons:
        P, states = build_transition_matrix(grid, path, goal, epsilon=eps, include_fail=True)
        assert verify_stochastic(P), f"Matrice non stochastique pour ε={eps}"
        
        goal_idx = states.index(goal)
        fail_idx = len(states) - 1
        start_idx = states.index(start)
        
        pi0 = np.zeros(len(states))
        pi0[start_idx] = 1.0
        
        # Calcul matriciel P^n
        prob_matrix = goal_probability_over_time(pi0, P, goal_idx, MAX_STEPS)
        matrix_probs_all[eps] = prob_matrix
        
        # Simulation MC
        mc = monte_carlo(P, states, start_idx, goal_idx, fail_idx, N=N_SIM, max_steps=MAX_STEPS, seed=42)
        mc_results[eps] = mc
        
        print(f"  {eps:>6.1f} {mc['prob_goal']:>12.4f} {prob_matrix[-1]:>18.4f} "
              f"{mc['mean_time_goal']:>14.2f} {mc['prob_fail']:>10.4f}")
    
    # Figure 1: P(GOAL en n étapes) pour chaque epsilon
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for eps, col in zip(epsilons, colors):
        steps = np.arange(MAX_STEPS + 1)
        ax.plot(steps, matrix_probs_all[eps], '-', color=col, lw=2, label=f'ε={eps} (matriciel)')
        mc = mc_results[eps]
        # Courbe MC approx (cumulative)
        if len(mc['steps_goal']) > 0:
            sorted_steps = np.sort(mc['steps_goal'])
            cdf = np.arange(1, len(sorted_steps)+1) / mc['N']
            ax.plot(sorted_steps, cdf, '--', color=col, lw=1.5, alpha=0.6, label=f'ε={eps} (MC)')
    
    ax.set_xlabel("Nombre d'étapes n", fontsize=12)
    ax.set_ylabel("P(Xₙ = GOAL)", fontsize=12)
    ax.set_title("Probabilité d'atteindre GOAL selon ε\n(Calcul matriciel vs Monte-Carlo)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_STEPS)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path_fig = os.path.join(OUTPUT_DIR, "E2_epsilon_impact.png")
    plt.savefig(path_fig, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Distribution temps atteinte GOAL
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, eps in zip(axes.flat, epsilons):
        mc = mc_results[eps]
        if len(mc['steps_goal']) > 0:
            ax.hist(mc['steps_goal'], bins=20, color='#42A5F5', alpha=0.8, edgecolor='black')
            ax.axvline(mc['mean_time_goal'], color='red', lw=2, linestyle='--',
                       label=f"Moyenne={mc['mean_time_goal']:.1f}")
            ax.legend(fontsize=9)
        ax.set_title(f"ε = {eps} | P(GOAL)={mc['prob_goal']:.3f}", fontweight='bold')
        ax.set_xlabel("Nombre d'étapes")
        ax.set_ylabel("Fréquence")
    
    plt.suptitle("Distribution du temps d'atteinte de GOAL", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path_fig2 = os.path.join(OUTPUT_DIR, "E2_distribution_temps.png")
    plt.savefig(path_fig2, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Figures sauvegardées!")
    return mc_results, matrix_probs_all


# ──────────────────────────────────────────────────────────────
# Expérience E3 : Comparaison heuristiques admissibles
# ──────────────────────────────────────────────────────────────

def experiment_E3():
    """Compare h=0 (UCS) vs h=Manhattan vs h=Euclidienne."""
    print("\n" + "="*60)
    print("EXPÉRIENCE E3 : Comparaison heuristiques admissibles")
    print("="*60)
    
    all_results = {}
    
    for gname, (grid, start, goal) in GRIDS.items():
        r_h0 = astar(grid, start, goal, heuristic=zero_heuristic)
        r_h0["algorithm"] = "A* h=0 (UCS)"
        
        r_manhattan = astar(grid, start, goal, heuristic=manhattan)
        r_manhattan["algorithm"] = "A* h=Manhattan"
        
        r_euclidean = astar(grid, start, goal, heuristic=euclidean)
        r_euclidean["algorithm"] = "A* h=Euclidienne"
        
        results = [r_h0, r_manhattan, r_euclidean]
        all_results[gname] = results
        
        print(f"\n  Grille {gname.upper()}:")
        print(f"  {'Heuristique':<22} {'Coût':>8} {'Nœuds':>8} {'Admissible':>12}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*12}")
        for r in results:
            print(f"  {r['algorithm']:<22} {r['cost']:>8.1f} {r['nodes_expanded']:>8} {'Oui':>12}")
    
    # Figure comparative
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    gnames = list(GRIDS.keys())
    
    for col_idx, gname in enumerate(gnames):
        results = all_results[gname]
        labels = [r["algorithm"].replace("A* ", "").replace("(UCS)", "\n(UCS)") for r in results]
        nodes = [r["nodes_expanded"] for r in results]
        costs = [r["cost"] for r in results]
        
        x = np.arange(len(labels))
        bars = axes[col_idx].bar(x, nodes, color=['#EF5350', '#42A5F5', '#66BB6A'], alpha=0.85, edgecolor='black', width=0.5)
        axes[col_idx].set_xticks(x)
        axes[col_idx].set_xticklabels(labels, fontsize=9)
        axes[col_idx].set_title(f"Grille {gname}", fontweight='bold')
        axes[col_idx].set_ylabel("Nœuds développés")
        for bar, c in zip(bars, costs):
            axes[col_idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                               f'c={c:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle("Comparaison heuristiques admissibles\n(h=0 vs Manhattan vs Euclidienne)", 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path_fig = os.path.join(OUTPUT_DIR, "E3_heuristiques.png")
    plt.savefig(path_fig, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure sauvegardée!")
    
    return all_results


# ──────────────────────────────────────────────────────────────
# Expérience E4 : Weighted A* (vitesse vs optimalité)
# ──────────────────────────────────────────────────────────────

def experiment_E4():
    """Weighted A* : compromis vitesse vs optimalité."""
    print("\n" + "="*60)
    print("EXPÉRIENCE E4 (Option) : Weighted A* — vitesse vs optimalité")
    print("="*60)
    
    grid, start, goal = GRIDS["difficile"]
    weights = [1.0, 1.5, 2.0, 3.0, 5.0]
    
    results = []
    r_optimal = astar(grid, start, goal, heuristic=manhattan)
    optimal_cost = r_optimal["cost"]
    
    print(f"\n  Coût optimal (A*, w=1): {optimal_cost}")
    print(f"\n  {'Poids w':>10} {'Coût':>8} {'Ratio':>8} {'Nœuds':>8} {'Temps(ms)':>12}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    
    for w in weights:
        r = weighted_astar(grid, start, goal, w=w, heuristic=manhattan)
        ratio = r["cost"] / optimal_cost if optimal_cost > 0 else 1.0
        results.append({"w": w, **r, "ratio": ratio})
        print(f"  {w:>10.1f} {r['cost']:>8.1f} {ratio:>8.3f} {r['nodes_expanded']:>8} {r['time_ms']:>12.3f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ws = [r["w"] for r in results]
    ax1.plot(ws, [r["nodes_expanded"] for r in results], 'o-', color='#EF5350', lw=2, ms=8, label="Nœuds")
    ax1.set_xlabel("Poids w", fontsize=12)
    ax1.set_ylabel("Nœuds développés", fontsize=12)
    ax1.set_title("Efficacité de la recherche", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(ws, [r["ratio"] for r in results], 's-', color='#42A5F5', lw=2, ms=8, label="Ratio coût/optimal")
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.7, label="Optimal (ratio=1)")
    ax2.set_xlabel("Poids w", fontsize=12)
    ax2.set_ylabel("Ratio coût / coût optimal", fontsize=12)
    ax2.set_title("Dégradation de l'optimalité", fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Weighted A* : compromis vitesse/optimalité\n(Grille difficile)", 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path_fig = os.path.join(OUTPUT_DIR, "E4_weighted_astar.png")
    plt.savefig(path_fig, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure sauvegardée!")
    
    return results


# ──────────────────────────────────────────────────────────────
# Expérience E5 : Analyse Markov complète (classes, absorption)
# ──────────────────────────────────────────────────────────────

def experiment_E5():
    """Analyse complète Markov: classes, absorption, simulation vs matriciel."""
    print("\n" + "="*60)
    print("EXPÉRIENCE E5 : Analyse Markov — Classes, Absorption, Validation")
    print("="*60)
    
    grid, start, goal = GRIDS["facile"]
    result_astar = astar(grid, start, goal, heuristic=manhattan)
    path = result_astar["path"]
    eps = 0.15
    
    P, states = build_transition_matrix(grid, path, goal, epsilon=eps, include_fail=True)
    
    print(f"\n  Grille facile | ε={eps} | |états|={len(states)}")
    print(f"  Matrice P stochastique: {verify_stochastic(P)}")
    
    goal_idx = states.index(goal)
    fail_idx = len(states) - 1
    start_idx = states.index(start)
    
    # Classes de communication
    classes = identify_classes(P, states)
    print(f"\n  Classes de communication ({len(classes)} classes):")
    for i, cls in enumerate(classes):
        print(f"    Classe {i+1}: {cls['type']} | États: {cls['states'][:5]}{'...' if len(cls['states'])>5 else ''}")
    
    # Absorption
    absorption = compute_absorption(P, states, goal_idx, fail_idx)
    if absorption:
        print(f"\n  Absorption (états transitoires: {len(absorption['transient_states'])}):")
        print(f"  Temps moyen avant absorption (depuis départ): {absorption['t_mean'][0]:.2f} étapes")
        print(f"  P(GOAL depuis départ)  = {absorption['B'][0, 0]:.4f}")
        print(f"  P(FAIL depuis départ)  = {absorption['B'][0, 1]:.4f}")
    
    # Simulation MC vs Matriciel
    pi0 = np.zeros(len(states))
    pi0[start_idx] = 1.0
    MAX_STEPS = 60
    N_SIM = 5000
    
    probs_matrix = goal_probability_over_time(pi0, P, goal_idx, MAX_STEPS)
    mc = monte_carlo(P, states, start_idx, goal_idx, fail_idx, N=N_SIM, max_steps=MAX_STEPS, seed=0)
    
    print(f"\n  Simulation MC ({N_SIM} trajectoires, ε={eps}):")
    print(f"  P(GOAL) MC={mc['prob_goal']:.4f} | Matrice={probs_matrix[-1]:.4f}")
    print(f"  P(FAIL) MC={mc['prob_fail']:.4f}")
    print(f"  Temps moyen GOAL: {mc['mean_time_goal']:.2f} ± {mc['std_time_goal']:.2f}")
    
    # Figure comparaison MC vs Matriciel
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    steps = np.arange(MAX_STEPS + 1)
    axes[0].plot(steps, probs_matrix, 'b-', lw=2.5, label="Calcul matriciel π⁽ⁿ⁾=π⁽⁰⁾Pⁿ")
    
    # CDF empirique MC
    if len(mc['steps_goal']) > 0:
        sorted_steps = np.sort(mc['steps_goal'])
        cdf = np.arange(1, len(sorted_steps)+1) / N_SIM
        axes[0].step(sorted_steps, cdf, 'r--', lw=2, alpha=0.8, label=f"Monte-Carlo (N={N_SIM})")
    
    axes[0].set_xlabel("Étapes n", fontsize=12)
    axes[0].set_ylabel("P(atteindre GOAL)", fontsize=12)
    axes[0].set_title("Validation : Matrice vs Monte-Carlo", fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    
    # Heatmap des premières lignes de P^n
    Pn = compute_pn(P, 20)
    im = axes[1].imshow(Pn[:min(15, len(states)), :min(15, len(states))], 
                         cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title("P²⁰ (premières 15 lignes/colonnes)", fontweight='bold')
    axes[1].set_xlabel("État destination j")
    axes[1].set_ylabel("État source i")
    
    plt.suptitle("Analyse Markov complète (Grille facile, ε=0.15)", 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path_fig = os.path.join(OUTPUT_DIR, "E5_markov_analyse.png")
    plt.savefig(path_fig, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure sauvegardée!")
    
    return {"absorption": absorption, "mc": mc, "probs_matrix": probs_matrix}


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Mini-Projet IA : A* + Chaînes de Markov                ║")
    print("║  Planification robuste sur grille                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    e1 = experiment_E1()
    e2 = experiment_E2()
    e3 = experiment_E3()
    e4 = experiment_E4()
    e5 = experiment_E5()
    
    print("\n" + "="*60)
    print("Toutes les expériences terminées.")
    print("="*60)
