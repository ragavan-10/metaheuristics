import numpy as np
from core.optimizer import Optimizer
from core.population import Population

class ArithmeticGA(Optimizer):
    def __init__(self, objective_function, population_size, crossover_prob=0.8, mutation_prob=0.01, mutation_std=2, tournament_size=3):
        super().__init__(objective_function, population_size)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_std = mutation_std
        self.tournament_size = tournament_size

    def tournament_selection(self):
        n_individuals = len(self.population.individuals)
        tournament_indices = np.random.choice(n_individuals, self.tournament_size, replace=False)
        tournament_fitness = [self.population.individuals[i].fitness for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population.individuals[winner_idx]

    def arithmetic_crossover(self, parent1, parent2):
        if np.random.rand() > self.crossover_prob:
            return parent1.clone(), parent2.clone()
        offspring1 = parent1.clone()
        offspring2 = parent2.clone()  
        alpha = np.random.rand()
        offspring1.position = alpha * parent1.position + (1 - alpha) * parent2.position
        offspring2.position = (1 - alpha) * parent1.position + alpha * parent2.position
        offspring1.clip_to_bounds()
        offspring2.clip_to_bounds()
        return offspring1, offspring2

    def gaussian_mutation(self, individual):
        offspring = individual.clone()
        mutation_mask = np.random.rand(self.dimensions) < self.mutation_prob
        noise = np.random.normal(0, self.mutation_std, self.dimensions)
        offspring.position += mutation_mask * noise
        offspring.clip_to_bounds()
        return offspring

    def step(self):
        
        new_individuals = []
        while len(new_individuals) < self.population_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            offspring1, offspring2 = self.arithmetic_crossover(p1, p2)
            offspring1 = self.gaussian_mutation(offspring1)
            offspring2 = self.gaussian_mutation(offspring2)
            new_individuals.extend([offspring1, offspring2])

        self.population.individuals = new_individuals[:self.population_size]
        self.population.evaluate()
        self.set_metrics()
        self.iteration += 1
        return self.population.best_fitness

    def run(self, max_iterations, termination_criteria=None):
        if self.population is None:
            self.initialize()
        best_fitness_history = []
        for i in range(max_iterations):
            current_best = self.step()
            best_fitness_history.append(current_best)
            if (i) % (max_iterations//10) == 0:
                print(f"Iteration {i}/{max_iterations}, Best Fitness: {current_best}")

        print(f"Optimization completed after {self.iteration} iterations")
        print(f"Best solution: {self.population.best_individual.position}")
        print(f"Best fitness: {self.population.best_fitness}")
        return self.population.best_individual, best_fitness_history