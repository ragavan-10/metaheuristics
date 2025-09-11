import numpy as np
from core.population import Population, Individual
from core.optimizer import Optimizer
from util.fuzzy import FuzzySystem
from collections import deque

class FHPSOE(Optimizer):
    
    def __init__(self, objective_function, population_size, inertia_weight=0.76, c1=1.5, c2=1.75, influence_factor=0.5, crossover_prob=0.3, mutation_prob=0.9, tournament_size=7, window=30):
    
        super().__init__(objective_function, population_size)
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2
        self.window = window
        self.influence_factor = influence_factor
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.pso_eff_buffer = deque(maxlen=window)
        self.ga_eff_buffer = deque(maxlen=window)
        self.fuzzy_system = FuzzySystem()
        
        # Metrics tracking
        self.metrics_history = {
            'best_fitness': [],
            'mean_fitness': [],
            'diversity': [],
            'pso_efficiency': [],
            'ga_efficiency': []
        }
    
    def tournament_selection(self):

        n_individuals = len(self.population.individuals)
        tournament_indices = np.random.choice(n_individuals, self.tournament_size, replace=False)
        tournament_fitness = [self.population.individuals[i].fitness for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population.individuals[winner_idx]
    
    def crossover(self, parent1, parent2):
        
        # Create copies to avoid modifying originals
        offspring1 = parent1.clone()
        offspring2 = parent2.clone()
        
        # Randomly select genes to be exchanged
        n_genes = np.random.randint(1, self.dimensions + 1)
        gene_indices = np.random.choice(self.dimensions, n_genes, replace=False)
        
        # Exchange the selected genes
        for i in gene_indices:
            offspring1.position[i], offspring2.position[i] = offspring2.position[i], offspring1.position[i]
            
        return offspring1, offspring2
    
    def mutation(self, individual):

        offspring = individual.clone()
        
        n_genes = np.random.randint(1, self.dimensions + 1)
        gene_indices = np.random.choice(self.dimensions, n_genes, replace=False)
        
        for i in gene_indices:
            lower, upper = self.bounds[i][0], self.bounds[i][1]
            phi = np.random.uniform(0, 1)
            offspring.position[i] = phi * (upper - lower) + lower
            
        return offspring
    
    def update_velocity(self, individual):
        
        r1 = np.random.uniform(0, 1, self.dimensions)
        r2 = np.random.uniform(0, 1, self.dimensions)

        inertial = self.inertia_weight * individual.velocity

        cognitive = self.c1 * r1 * (individual.best_position - individual.position)

        social = self.c2 * r2 * (self.population.best_individual.position - individual.position)
        
        individual.velocity = inertial + cognitive + social
    
    def update_position(self, individual):

        individual.position = individual.position + individual.velocity
        individual.clip_to_bounds()

    def compute_efficiencies(self, E_GA, CH_size, E_PSO):
        N = self.population_size
        w_o = self.window
    
        # Update rolling buffers
        self.ga_eff_buffer.append((E_GA, CH_size))
        self.pso_eff_buffer.append(E_PSO)
    
        # Compute windowed averages
        total_E_GA = sum(e for e, _ in self.ga_eff_buffer)
        total_CH = sum(c for _, c in self.ga_eff_buffer)
        total_E_PSO = sum(self.pso_eff_buffer)
        window_size = len(self.pso_eff_buffer)
    
        delta_E_prime_GA = total_E_GA / total_CH if total_CH > 0 else 0
        delta_E_prime_PSO = total_E_PSO / (window_size * N) if window_size > 0 else 0
    
        # Normalized efficiencies
        total = delta_E_prime_GA + delta_E_prime_PSO
        delta_EN_GA = delta_E_prime_GA / total if total > 0 else 0
        return delta_EN_GA
    
    
    def step(self):

        E_GA = 0
        E_PSO = 0
        CH_size = 0
        N = self.population_size


        # Particle Swarm Optimization phase
        for individual in self.population.individuals:

            # Update velocity and position
            self.update_velocity(individual)
            self.update_position(individual)

            individual.evaluate(self.objective_function.evaluate)
            if individual.fitness < self.population.best_fitness:
                E_PSO += (self.population.best_fitness - individual.fitness)
        
        # Genetic algorithm phase
        temporary_population = []        
        num_genetic_ops = int(self.influence_factor * self.population_size)
        
        for _ in range(num_genetic_ops):
            # Mutation
            if np.random.random() < self.mutation_prob:
                parent = self.tournament_selection()
                offspring = self.mutation(parent)

                offspring.evaluate(self.objective_function.evaluate)
                if offspring.fitness < self.population.best_fitness:
                    E_GA += self.population.best_fitness - offspring.fitness
                CH_size += 1
                offspring.parent_index = self.population.individuals.index(parent)
                temporary_population.append(offspring)
            
            # Crossover
            if np.random.random() < self.crossover_prob:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                offspring1.evaluate(self.objective_function.evaluate)
                offspring2.evaluate(self.objective_function.evaluate)
                
                if offspring1.fitness < self.population.best_fitness:
                    E_GA += self.population.best_fitness - offspring1.fitness
                if offspring2.fitness < self.population.best_fitness:
                    E_GA += self.population.best_fitness - offspring2.fitness

                CH_size += 2
                offspring1.parent_index = self.population.individuals.index(parent1)
                offspring2.parent_index = self.population.individuals.index(parent2)
                temporary_population.extend([offspring1, offspring2])
        
        # Merge temporary population with main population 
        for offspring in temporary_population:
            parent_idx = offspring.parent_index
            parent = self.population.individuals[parent_idx]
            
            # Replace parent with offspring if offspring has better fitness than parent's personal best
            if offspring.fitness < parent.best_fitness:
                self.population.individuals[parent_idx] = offspring
                # Update personal best
                offspring.best_position = offspring.position.copy()
                offspring.best_fitness = offspring.fitness
        
        # Update global best
        self.population.update_best()
        
        # Efficiency calculation
        delta_EN_GA = self.compute_efficiencies(E_GA, CH_size, E_PSO)



        # Calculate population statistics
        mean_fitness = np.mean([ind.fitness for ind in self.population.individuals])
        self.population.calculate_diversity()
        
        # Store metrics
        self.metrics_history['best_fitness'].append(self.population.best_fitness)
        self.metrics_history['mean_fitness'].append(mean_fitness)
        self.metrics_history['diversity'].append(self.population.diversity)
        self.metrics_history['pso_efficiency'].append(E_PSO)
        self.metrics_history['ga_efficiency'].append(E_GA)
        
        self.iteration += 1
        
        return self.population.best_fitness, delta_EN_GA
    
    def run(self, max_iterations):
        """Run the optimization process for a specified number of iterations"""
        if self.population is None:
            self.initialize()
        
        best_fitness_history = []
        
        for i in range(max_iterations):
            current_best, delta_en_ga = self.step()
            best_fitness_history.append(current_best)
            
            # Print progress
            if max_iterations < 10 or  (i) % (max_iterations//10) == 0:
                print(f"Iteration {i}/{max_iterations}, Best Fitness: {current_best}")
                # print(f"Fuzzy system Pe = {self.influence_factor} Ef = {delta_en_ga}")
            
            if (i + 1) % 10 == 0:
                delta_pe = self.fuzzy_system.infer(self.influence_factor, delta_en_ga)
                self.influence_factor = self.influence_factor + delta_pe

            

        
        print(f"Optimization completed after {self.iteration} iterations")
        print(f"Best solution: {self.population.best_individual.position}")
        print(f"Best fitness: {self.population.best_fitness}")
        
        return self.population.best_individual, best_fitness_history