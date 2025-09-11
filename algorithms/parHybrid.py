import numpy as np
from core.population import Population
from core.optimizer import Optimizer
from algorithms.simple_ga import SimpleGA
from algorithms.simple_pso import SimplePSO

class ParallelHybrid(Optimizer):
    def __init__(self, objective_function, population_size, exchange_interval=10, **kwargs):
        super().__init__(objective_function, population_size)
        half_size = population_size // 2
        self.ga = SimpleGA(objective_function, half_size, **kwargs.get('ga', {}))
        self.pso = SimplePSO(objective_function, half_size, **kwargs.get('pso', {}))
        self.exchange_interval = exchange_interval

    def initialize(self):
        self.population = Population(self.population_size, self.objective_function, self.dimensions)
        self.pso.population = Population(self.population_size, self.objective_function, self.dimensions)
        self.pso.population.initialize()
        self.pso.population.evaluate()
        self.ga.population = Population(self.population_size, self.objective_function, self.dimensions)
        self.ga.population.initialize()
        self.ga.population.evaluate()        
        self.population.individuals = self.ga.population.individuals + self.pso.population.individuals
        self.iteration = 0

    def exchange_individuals(self):
        # Sort individuals by fitness
        self.ga.population.individuals.sort(key=lambda ind: ind.fitness)
        self.pso.population.individuals.sort(key=lambda ind: ind.fitness)
        
        # Exchange top individuals (e.g., top 10%)
        n_exchange = max(1, len(self.ga.population.individuals) // 10)
        for i in range(n_exchange):
            ga_ind = self.ga.population.individuals[i].clone()
            pso_ind = self.pso.population.individuals[i].clone()
            self.ga.population.individuals[-1 - i] = pso_ind
            self.pso.population.individuals[-1 - i] = ga_ind

    def step(self):
        self.ga.step()
        self.pso.step()
        if self.iteration % self.exchange_interval == 0:
            self.exchange_individuals()
        self.iteration += 1
        self.set_metrics()
        # Return the better best fitness
        return min(self.ga.population.best_fitness, self.pso.population.best_fitness)

    def run(self, max_iterations, termination_criteria=None):
        self.initialize()
        best_fitness_history = []
        for i in range(max_iterations):
            current_best = self.step()
            best_fitness_history.append(current_best)
            if max_iterations < 10 or i % (max_iterations // 10) == 0:
                print(f"Iteration {i}/{max_iterations}, Best Fitness: {current_best}")

        best_algo = self.ga if self.ga.population.best_fitness < self.pso.population.best_fitness else self.pso
        print("Optimization completed")
        print(f"Best solution: {best_algo.population.best_individual.position}")
        print(f"Best fitness: {best_algo.population.best_fitness}")
        return best_algo.population.best_individual, best_fitness_history
