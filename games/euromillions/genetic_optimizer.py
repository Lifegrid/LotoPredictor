import random
import logging
import pandas as pd  # 🔧 Manquait ici !
from games.euromillions.data_loader import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeneticOptimizer:
    def __init__(self, population_size=100, generations=30):
        self.population_size = population_size
        self.generations = generations

    def generate_optimized_grid(self, history=None):
        try:
            data = load_data()
            if data.empty:
                logger.warning("Aucune donnée disponible pour l'optimisation.")
                return self._generate_random_grid()

            past_draws = data[["boule_1", "boule_2", "boule_3", "boule_4", "boule_5", "etoile_1", "etoile_2"]].tail(30)
            past_draws = past_draws.to_dict('records')
            past_draws = [
                {
                    "boules": [d["boule_1"], d["boule_2"], d["boule_3"], d["boule_4"], d["boule_5"]],
                    "etoiles": [d["etoile_1"], d["etoile_2"]]
                } for d in past_draws if all(pd.notna(d[k]) for k in ["boule_1", "etoile_1"])
            ]

            if not past_draws:
                return self._generate_random_grid()

            population = [self._generate_random_grid() for _ in range(self.population_size)]

            def fitness(candidate, draws):
                score = 0
                for draw in draws:
                    score -= len(set(candidate["boules"]) & set(draw["boules"]))
                    score -= len(set(candidate["etoiles"]) & set(draw["etoiles"]))
                return score

            for _ in range(self.generations):
                population = sorted(population, key=lambda x: fitness(x, past_draws))
                survivors = population[:self.population_size // 2]
                children = []

                while len(children) < self.population_size // 2:
                    p1, p2 = random.sample(survivors, 2)
                    child_boules = sorted(random.sample(list(set(p1["boules"] + p2["boules"])), 5))
                    child_etoiles = sorted(random.sample(list(set(p1["etoiles"] + p2["etoiles"])), 2))
                    children.append({"boules": child_boules, "etoiles": child_etoiles})

                population = survivors + children

            best = min(population, key=lambda x: fitness(x, past_draws))
            return {
                "date": None,
                "boules": best["boules"],
                "etoiles": best["etoiles"]
            }

        except Exception as e:
            logger.error(f"Erreur Optimiseur Génétique : {e}")
            return self._generate_random_grid()

    def _generate_random_grid(self):
        return {
            "boules": sorted(random.sample(range(1, 51), 5)),
            "etoiles": sorted(random.sample(range(1, 13), 2))
        }
