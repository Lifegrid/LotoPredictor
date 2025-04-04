import random
from collections import Counter
from deap import base, creator, tools, algorithms
from games.euromillions.data_loader import load_training_data

class GeneticOptimizer:
    def __init__(self):
        self.pop_size = 150
        self.ngen = 50
        self.cxpb = 0.7
        self.mutpb = 0.3
        self._setup()

    def _setup(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_int", random.randint, 1, 50)
        self.toolbox.register("individual", tools.initRepeat,
                              creator.Individual, self.toolbox.attr_int, n=5)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=1, up=50, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual, frequency_map):
        return (sum(frequency_map.get(num, 0) for num in individual),)

    def generate_optimized_grid(self, history=None):
        history = history or load_training_data()
        all_numbers = [n for draw in history for n in draw['boules']]
        if not all_numbers:
            return self.generate_random()

        freq_map = Counter(all_numbers)
        self.toolbox.register("evaluate", lambda ind: self.evaluate(ind, freq_map))

        pop = self.toolbox.population(n=self.pop_size)
        algorithms.eaSimple(pop, self.toolbox, self.cxpb, self.mutpb, self.ngen, verbose=False)
        best = tools.selBest(pop, k=1)[0]

        unique = list(set(best))
        while len(unique) < 5:
            n = random.randint(1, 50)
            if n not in unique:
                unique.append(n)

        return {
            "boules": sorted(unique[:5]),
            "etoiles": sorted(random.sample(range(1, 13), 2))
        }

    def generate_random(self):
        return {
            "boules": sorted(random.sample(range(1, 51), 5)),
            "etoiles": sorted(random.sample(range(1, 13), 2))
        }
