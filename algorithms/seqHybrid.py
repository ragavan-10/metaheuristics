import numpy as np
from core.population import Population
from core.optimizer import Optimizer
from algorithms.simple_ga import SimpleGA
from algorithms.simple_pso import SimplePSO

class SequentialHybrid(Optimizer):
    def __init__(self, objective_function, population_size, switch_interval=10, **kwargs):
        super().__init__(objective_function, population_size)
        self.ga = SimpleGA(objective_function, population_size, **kwargs.get('ga', {}))
        self.pso = SimplePSO(objective_function, population_size, **kwargs.get('pso', {}))
        self.switch_interval = switch_interval
        self.use_ga = True

    def initialize(self):
        self.population = Population(self.population_size, self.objective_function, self.dimensions)
        self.population.initialize()
        self.pso.population = self.population
        self.ga.population = self.population
        self.population.evaluate()
        self.iteration = 0

    def step(self):
        if self.use_ga:
            self.ga.step()
            self.pso.population = self.ga.population  # sync
        else:
            self.pso.step()
            self.ga.population = self.pso.population  # sync

        if self.iteration % self.switch_interval == 0:
            self.use_ga = not self.use_ga
        self.iteration += 1
        self.set_metrics()
        return self.population.best_fitness

    def run(self, max_iterations, termination_criteria=None):
        self.initialize()
        best_fitness_history = []
        for i in range(max_iterations):
            current_best = self.step()
            best_fitness_history.append(current_best)
            if max_iterations < 10 or i % (max_iterations // 10) == 0:
                print(f"Iteration {i}/{max_iterations}, Best Fitness: {current_best}")
        print("Optimization completed")
        print(f"Best solution: {self.population.best_individual.position}")
        print(f"Best fitness: {self.population.best_fitness}")
        return self.population.best_individual, best_fitness_history
