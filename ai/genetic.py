from deap import base, creator, tools, algorithms
import random
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class GeneticOptimizer:
    def __init__(self):
        self.pop_size = 150
        self.ngen = 50
        self.cxpb = 0.7
        self.mutpb = 0.3
        self.setup_evolution()

    def setup_evolution(self):
        """Configuration DEAP"""
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_int", random.randint, 1, 50)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_int, n=5)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    def evaluate_individual(self, individual, frequency_map):
        try:
            score = sum(frequency_map.get(num, 0) for num in individual)
            return (score,)
        except Exception as e:
            logger.error(f"Erreur d'évaluation: {e}")
            return (0,)

    def generate_optimized_grid(self, history):
        try:
            all_numbers = []
            for draw in history:
                nums = draw.get('boules') or draw.get('results', {}).get('boules', [])
                all_numbers.extend(nums)

            if not all_numbers:
                logger.warning("Historique vide — génération aléatoire")
                return self.generate_random_grid()

            frequency_map = Counter(all_numbers)

            self.toolbox.register("evaluate", lambda ind: self.evaluate_individual(ind, frequency_map))
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", tools.mutUniformInt, low=1, up=50, indpb=0.2)
            self.toolbox.register("select", tools.selTournament, tournsize=3)

            population = self.toolbox.population(n=self.pop_size)
            algorithms.eaSimple(population, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen, verbose=False)

            best = tools.selBest(population, k=1)[0]
            unique_numbers = list(set(best))
            while len(unique_numbers) < 5:
                n = random.randint(1, 50)
                if n not in unique_numbers:
                    unique_numbers.append(n)

            return {
                "boules": sorted(unique_numbers[:5]),
                "etoiles": sorted(random.sample(range(1, 13), 2))
            }

        except Exception as e:
            logger.error(f"Erreur majeure algorithme génétique: {e}")
            return self.generate_random_grid()

    def generate_random_grid(self):
        return {
            "boules": sorted(random.sample(range(1, 51), 5)),
            "etoiles": sorted(random.sample(range(1, 13), 2))
        }
