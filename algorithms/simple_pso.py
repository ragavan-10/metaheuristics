import numpy as np
from core.population import Population
from core.optimizer import Optimizer

class SimplePSO(Optimizer):
    def __init__(self, objective_function, population_size, inertia_weight=0.75, c1=1.5, c2=1.7, noise_std=0.01):
        super().__init__(objective_function, population_size)
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2
        self.noise_std = noise_std

    def update_velocity(self, individual):
        r1 = np.random.uniform(0, 1, self.dimensions)
        r2 = np.random.uniform(0, 1, self.dimensions)
        inertial = self.inertia_weight * individual.velocity
        cognitive = self.c1 * r1 * (individual.best_position - individual.position)
        social = self.c2 * r2 * (self.population.best_individual.position - individual.position)
        individual.velocity = inertial + cognitive + social
    
    def update_position(self, individual):
        noise = np.random.normal(0, self.noise_std, size=self.dimensions)
        individual.position = individual.position + individual.velocity + noise
        individual.clip_to_bounds()


    def step(self):

        for individual in self.population.individuals:
            self.update_velocity(individual)
            self.update_position(individual)
        
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
