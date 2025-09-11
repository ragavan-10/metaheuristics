import numpy as np
from core.population import Population
from core.optimizer import Optimizer

class CLPSO(Optimizer):
    def __init__(self, objective_function, population_size, inertia=0.4, c=1.5, learning_prob=0.1):
        super().__init__(objective_function, population_size)
        self.inertia = inertia
        self.c = c
        self.learning_prob = learning_prob


    def comprehensive_learning(self, individual):
        D = len(individual.position)
        exemplar = np.copy(individual.best_position)
        for d in range(D):
            if np.random.rand() < self.learning_prob:
                peer = self.population.individuals[np.random.randint(self.population.size)]
                exemplar[d] = peer.best_position[d]      
        individual.velocity = individual.velocity * self.inertia + self.c * np.random.rand(D) * (exemplar - individual.position)
    
    def position_update(self, individual):
        individual.position += individual.velocity
        individual.clip_to_bounds()


    def step(self):

        for individual in self.population.individuals:
            old_fitness = individual.fitness
            self.comprehensive_learning(individual)
            self.position_update(individual)
            individual.evaluate(self.objective_function.evaluate)

        self.population.update_best()
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
