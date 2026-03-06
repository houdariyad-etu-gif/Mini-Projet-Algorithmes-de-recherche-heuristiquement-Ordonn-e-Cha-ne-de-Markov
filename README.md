# 🤖 Planification Robuste Sur Grille — A* + Chaînes de Markov

> Mini-projet académique — ENSET Mohammedia | Département Informatique & IA  
> Filière SDIA | Année universitaire 2025–2026  
> **Réalisé par :** Houda RIYAD | **Encadrant :** Pr. Mohamed MESTARI

---

## 📌 Description

Ce projet explore la planification d'un agent sur une grille 2D sous deux angles complémentaires :

- **A\*** : recherche du chemin optimal (déterministe) avec comparaison d'heuristiques
- **Chaînes de Markov** : modélisation probabiliste de l'exécution réelle avec incertitude ε

> *Question centrale :* le plan A\* reste-t-il performant quand l'agent peut dévier à chaque pas ?

---

## 🗂️ Structure du projet
```
.
├── astar.py          # Recherche heuristique : A*, UCS, Greedy, Weighted A*
├── markov.py         # Modèle stochastique : matrice P, absorption, classes
├── simulation.py     # Simulations Monte-Carlo
├── experiments.py    # Expériences et génération des figures (E1 à E5)
└── grids.py          # Définition des grilles (facile, moyenne, difficile)
```

## 🚀 Utilisation
```python
# Lancer toutes les expériences
python experiments.py

# Exemple : A* sur grille moyenne
from astar import search
from grids import GRID_MEDIUM

path, cost, nodes = search(GRID_MEDIUM, weight_g=1, weight_h=1)
print(f"Coût : {cost} | Nœuds développés : {nodes}")
```

---

## 🧪 Expériences

| # | Expérience | Description |
|---|-----------|-------------|
| E1 | Comparaison algorithmes | UCS vs Greedy vs A\* sur 3 grilles |
| E2 | Impact de ε | P(GOAL) matriciel vs Monte-Carlo pour ε ∈ {0.0, 0.1, 0.2, 0.3} |
| E3 | Heuristiques admissibles | h=0, Manhattan, Euclidienne |
| E4 | Weighted A\* | Compromis vitesse/optimalité selon w |
| E5 | Analyse Markov complète | Classes, absorption, périodicité, heatmap P²⁰ |

---

## 📊 Résultats clés

### Comparaison algorithmes (grille 15×15)
| Algorithme | Coût | Nœuds développés |
|-----------|------|-----------------|
| UCS | 28 | 136 |
| Greedy | 28 | 31 |
| A\* (Manhattan) | 28 | **41** |

### Impact de l'incertitude ε (grille moyenne)
| ε | P(GOAL) Matriciel | P(GOAL) Monte-Carlo | P(FAIL) |
|---|------------------|---------------------|---------|
| 0.0 | 1.0000 | 1.0000 | 0.0000 |
| 0.1 | 0.6470 | 0.6353 | 0.3647 |
| 0.2 | 0.3844 | 0.3963 | 0.6037 |
| 0.3 | 0.2039 | 0.2027 | **0.7973** |

> ⚠️ Avec ε = 0.3, un plan A\* optimal n'a que **~20% de chance de succès**.

---

## 🧮 Modélisation mathématique

**Matrice de transition :**
- `p(n → n_voulu) = 1 − ε`
- `p(n → voisin_latéral) = ε/2`

**Distribution à l'instant n :**
```
π⁽ⁿ⁾ = π⁽⁰⁾ · Pⁿ
```

**Matrice fondamentale (états absorbants) :**
```
N = (I − Q)⁻¹
B = N · R   → probabilités d'absorption
t = N · 1   → temps moyen avant absorption
```

---

## 📐 Heuristiques comparées

| Heuristique | Admissible | Cohérente | Nœuds (15×15) |
|------------|-----------|----------|--------------|
| h = 0 (UCS) | ✅ | ✅ | 136 |
| Euclidienne | ✅ | ❌ | 65 |
| **Manhattan** | ✅ | ✅ | **41** |

---

## 🔍 Points notables

- **A\*** garantit l'optimalité déterministe — mais ignore ε
- **Markov** révèle la fragilité du plan : P(GOAL) chute de 100% → 20% quand ε passe de 0 à 0.3
- Validation croisée : écart matriciel/Monte-Carlo < 1.2% sur toutes les expériences
- La politique A\* est **statique** — une vraie planification robuste nécessiterait D\* Lite ou une formulation MDP

---

## 📚 Références

- Russell & Norvig, *AI: A Modern Approach* (4e éd., 2020)
- Hart, Nilsson & Raphael (1968) — algorithme A\*
- Puterman (1994) — *Markov Decision Processes*
- Norris (1997) — *Markov Chains*
- Koenig & Likhachev (2002) — D\* Lite
- Cours ENSET — Pr. Mohamed MESTARI (2025–2026)

---

