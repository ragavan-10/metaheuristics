import numpy as np
import copy

class Individual:
    def __init__(self, dimensions, bounds=None, ints=None):
        """
        Parameters:
            dimensions (int): Number of dimensions
            bounds (tuple): (lower_bound, upper_bound) for each dimension
        """
        self.dim = dimensions
        self.bounds = bounds
        self.ints = ints

        self.position = np.array([np.random.uniform(lower, upper) for (lower, upper) in bounds])
        self.velocity = np.zeros(dimensions)

        self.fitness = float('inf')
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        
    def evaluate(self, objective_function):
        """Evaluate fitness using the provided objective function"""
        self.fitness = objective_function(self.position)

        # Update personal best for PSO
        if self.fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness
        return self.fitness

    def clip_to_bounds(self):
        """Clip each element of x to its corresponding (min, max) bounds"""
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            self.position[i] = np.clip(self.position[i], lower, upper)
            if self.ints[i]:
                self.position[i] = int(round(self.position[i]))
        
    def clone(self):
        """Create a deep copy of the individual"""
        return copy.deepcopy(self)


class Population:
    def __init__(self, size, objective_func, dimensions):
        """
        Parameters:
            size (int): Population size
            objective_function (class ObjectiveFunction)
                Function to minimize the value of
        """
        self.individuals = []  # List of individuals
        self.size = size
        
        self.objective_function = objective_func
        self.dim = dimensions
        self.bounds = objective_func.bounds
        self.ints = objective_func.ints

        self.best_individual = None  # Global best individual
        self.best_fitness = float('inf')  # Global best fitness
        self.diversity = 0



    def initialize(self):
        """Initialize the population"""
        self.individuals = [
            Individual(dimensions=self.dim, bounds=self.bounds, ints=self.ints)
            for _ in range(self.size)]

        
    def evaluate(self):
        """Evaluate all individuals"""
        interation_best = float('inf')
        for ind in self.individuals:
            fitness = ind.evaluate(self.objective_function.evaluate)
            if fitness < self.best_fitness:
                self.best_fitness = ind.fitness
                self.best_individual = ind.clone()
        
    def update_best(self):
        """Update global best individual"""
        for ind in self.individuals:
            if ind.fitness < self.best_fitness:
                self.best_fitness = ind.fitness
                self.best_individual = ind.clone()

    def calculate_diversity(self):
        
        n = len(self.individuals)
        if n < 2:
            self.diversity = 0
            return

        total_distance = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.individuals[i].position - self.individuals[j].position)
                total_distance += dist
                count += 1

        self.diversity = total_distance / count
    