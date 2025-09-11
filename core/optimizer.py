from core.population import Population
import numpy as np

class Optimizer:
    def __init__(self, objective_function, population_size):
        """
        Parameters:
            objective_function (callable): Function to be optimized
            dimensions (int): Number of dimensions
            bounds (tuple): (lower_bound, upper_bound) for each dimension
            population_size (int): Size of population
        """
        self.objective_function = objective_function
        self.dimensions = objective_function.get_dim()
        self.bounds = objective_function.bounds
        self.ints = objective_function.ints
        self.population_size = population_size
        self.population = None
        self.iteration = 0
        self.metrics_history = {
            'best_fitness': [],
            'mean_fitness': [],
            'diversity': []
        }
        
    def initialize(self):
        self.population = Population(self.population_size, self.objective_function, self.dimensions)
        self.population.initialize()
        self.population.evaluate()
        self.iteration = 0
        
    def step(self):
        """Perform one iteration"""
        
    def run(self, max_iterations, termination_criteria=None):
        """Run optimization for specified iterations or until termination"""
        
    def set_metrics(self):
        mean_fitness = np.mean([ind.fitness for ind in self.population.individuals])
        self.population.calculate_diversity()
        self.metrics_history['best_fitness'].append(self.population.best_fitness)
        self.metrics_history['mean_fitness'].append(mean_fitness)
        self.metrics_history['diversity'].append(self.population.diversity)
