import numpy as np
from core.population import Population
from core.optimizer import Optimizer


class SimpleGA(Optimizer):
    def __init__(self, objective_function, population_size, crossover_prob=0.9, mutation_prob=0.01, tournament_size=8):
        super().__init__(objective_function, population_size)
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = int(tournament_size)
        


    def tournament_selection(self):
        n_individuals = len(self.population.individuals)
        tournament_indices = np.random.choice(n_individuals, self.tournament_size, replace=False)
        tournament_fitness = [self.population.individuals[i].fitness for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population.individuals[winner_idx]
    
    def uniform_crossover(self, parent1, parent2):
        if np.random.random() > self.crossover_prob:
            return parent1.clone(), parent2.clone()
        offspring1 = parent1.clone()
        offspring2 = parent2.clone()
        for i in range(len(parent1.position)):
            if np.random.random() < 0.5: 
                offspring1.position[i] = parent2.position[i]
                offspring2.position[i] = parent1.position[i]
        return offspring1, offspring2

    
    def mutation(self, individual):
        offspring = individual.clone()        
        for i in range(self.dimensions):
            if np.random.rand() < self.mutation_prob:
                lower, upper = self.bounds[i]
                offspring.position[i] = np.random.uniform(lower, upper)
                
        return offspring


    def step(self):

        new_individuals = []
        while len(new_individuals) < self.population_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            offspring1, offspring2 = self.uniform_crossover(p1, p2)
            offspring1 = self.mutation(offspring1)
            offspring2 = self.mutation(offspring2)
            new_individuals.extend([offspring1, offspring2])

        self.population.individuals = new_individuals[:self.population_size]
        
        self.population.evaluate()
        self.set_metrics()
        self.iteration+=1
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
