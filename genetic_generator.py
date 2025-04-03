import random
from collections import Counter
from typing import List, Dict
import pandas as pd
import numpy as np

# Configuration
POP_SIZE = 200
N_GEN = 100
MUTATION_RATE = 0.15

class GeneticGenerator:
    def __init__(self):
        self.df = pd.read_csv("data/euromillions_clean.csv")
        self._init_frequencies()
        self.last_numbers = self._get_last_numbers()
        
    def _init_frequencies(self):
        """Initialise les compteurs de fréquences"""
        boules = np.concatenate([self.df[f'boule_{i}'] for i in range(1,6)])
        etoiles = np.concatenate([self.df[f'etoile_{i}'] for i in range(1,3)])
        self.boule_freq = Counter(boules)
        self.etoile_freq = Counter(etoiles)
        self.all_boules = list(range(1, 51))
        self.all_etoiles = list(range(1, 13))
    
    def _get_last_numbers(self) -> set:
        """Récupère les numéros des 5 derniers tirages"""
        last_boules = set(self.df[[f"boule_{i}" for i in range(1,6)]].tail(5).values.ravel())
        last_etoiles = set(self.df[[f"etoile_{i}" for i in range(1,3)]].tail(5).values.ravel())
        return last_boules.union(last_etoiles)
    
    def score_grid(self, grid: Dict[str, list]) -> float:
        """Score basé sur la fréquence et la répartition"""
        # Score de fréquence
        b_score = sum(self.boule_freq[n] for n in grid["boules"])
        e_score = sum(self.etoile_freq[n] for n in grid["etoiles"])
        
        # Pénality pour numéros récents
        recency_penalty = sum(1 for n in grid["boules"] + grid["etoiles"] if n in self.last_numbers)
        
        # Bonus pour répartition uniforme
        range_bonus = (max(grid["boules"]) - min(grid["boules"])) / 49
        return (b_score + e_score) * 0.8 - (recency_penalty * 0.5) + (range_bonus * 20)
    
    def generate_grids(self) -> List[Dict[str, list]]:
        """Génère les meilleures grilles via algorithme génétique"""
        population = [self._random_valid_grid() for _ in range(POP_SIZE)]
        
        for generation in range(N_GEN):
            population = self._evolve_population(population)
            
        # Retourne les 5 meilleures grilles uniques
        return self._get_unique_best_grids(population)
    
    def _evolve_population(self, population: List[Dict]) -> List[Dict]:
        """Effectue une génération d'évolution"""
        # Sélection des elites
        scored = [(g, self.score_grid(g)) for g in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        elites = [g for g,_ in scored[:POP_SIZE//4]]
        
        # Reproduction
        children = []
        while len(children) < POP_SIZE - len(elites):
            parent1, parent2 = random.choices(elites, k=2)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            if self._is_valid_grid(child):
                children.append(child)
                
        return elites + children
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Croisement intelligent de deux grilles parentes"""
        # Boules: prend 3 du parent1 et 2 du parent2 (sans doublons)
        boules = list(set(parent1["boules"][:3] + parent2["boules"][-2:]))
        if len(boules) < 5:
            boules += random.sample([n for n in self.all_boules if n not in boules], 5 - len(boules))
        
        # Étoiles: prend 1 de chaque parent
        etoiles = list(set(parent1["etoiles"][:1] + parent2["etoiles"][-1:]))
        if len(etoiles) < 2:
            etoiles += random.sample([n for n in self.all_etoiles if n not in etoiles], 2 - len(etoiles))
            
        return {
            "boules": sorted(boules),
            "etoiles": sorted(etoiles)
        }
    
    def _mutate(self, grid: Dict) -> Dict:
        """Mutation contrôlée"""
        new_grid = grid.copy()
        
        # Mutation des boules
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, 4)
            available = [n for n in self.all_boules if n not in new_grid["boules"]]
            if available:
                new_grid["boules"][idx] = random.choice(available)
                new_grid["boules"].sort()
        
        # Mutation des étoiles
        if random.random() < MUTATION_RATE:
            idx = random.randint(0, 1)
            available = [n for n in self.all_etoiles if n not in new_grid["etoiles"]]
            if available:
                new_grid["etoiles"][idx] = random.choice(available)
                new_grid["etoiles"].sort()
                
        return new_grid
    
    def _random_valid_grid(self) -> Dict:
        """Génère une grille aléatoire valide sans doublons"""
        return {
            "boules": sorted(random.sample(self.all_boules, 5)),
            "etoiles": sorted(random.sample(self.all_etoiles, 2))
        }
    
    def _is_valid_grid(self, grid: Dict) -> bool:
        """Vérifie qu'une grille est valide"""
        return (len(set(grid["boules"])) == 5 and 
                len(set(grid["etoiles"])) == 2 and
                all(1 <= n <= 50 for n in grid["boules"]) and
                all(1 <= n <= 12 for n in grid["etoiles"]))
    
    def _get_unique_best_grids(self, population: List[Dict], n: int = 5) -> List[Dict]:
        """Retourne les meilleures grilles uniques"""
        scored = [(g, self.score_grid(g)) for g in population if self._is_valid_grid(g)]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        unique_grids = []
        seen = set()
        
        for grid, score in scored:
            grid_tuple = (tuple(grid["boules"]), tuple(grid["etoiles"]))
            if grid_tuple not in seen:
                seen.add(grid_tuple)
                unique_grids.append(grid)
                if len(unique_grids) >= n:
                    break
                    
        return unique_grids