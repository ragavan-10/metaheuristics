import numpy as np
from core.population import Population, Individual
from core.optimizer import Optimizer

class HPSOE(Optimizer):
    
    def __init__(self, objective_function, population_size, inertia_weight=0.76, c1=1.5, c2=1.75, influence_factor=0.1, crossover_prob=0.3, mutation_prob=0.9, tournament_size=9):
    
        super().__init__(objective_function, population_size)
        self.inertia_weight = inertia_weight
        self.c1 = c1
        self.c2 = c2
        self.influence_factor = influence_factor
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        
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
    
    def step(self):

        # Particle update phase (PSO part)
        PSO_efficiency = 0
        for individual in self.population.individuals:
            old_fitness = individual.fitness
            
            # Update velocity and position
            self.update_velocity(individual)
            self.update_position(individual)
            
            # Evaluate new position
            individual.evaluate(self.objective_function.evaluate)
            
            # Track PSO improvement
            if individual.fitness < old_fitness:
                PSO_efficiency += (old_fitness - individual.fitness)
        
        # Genetic algorithm phase
        temporary_population = []
        GA_efficiency = 0
        
        # Number of genetic operations based on influence factor
        num_genetic_ops = int(self.influence_factor * self.population_size)
        
        for _ in range(num_genetic_ops):
            # Mutation
            if np.random.random() < self.mutation_prob:
                parent = self.tournament_selection()
                offspring = self.mutation(parent)
                offspring.evaluate(self.objective_function.evaluate)
                
                # Track improvement from mutation
                old_fitness = parent.fitness
                if offspring.fitness < old_fitness:
                    GA_efficiency += (old_fitness - offspring.fitness)
                
                # Store offspring with reference to parent
                offspring.parent_index = self.population.individuals.index(parent)
                temporary_population.append(offspring)
            
            # Crossover
            if np.random.random() < self.crossover_prob:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                offspring1.evaluate(self.objective_function.evaluate)
                offspring2.evaluate(self.objective_function.evaluate)
                
                # Track improvement from crossover
                if offspring1.fitness < parent1.fitness:
                    GA_efficiency += (parent1.fitness - offspring1.fitness)
                if offspring2.fitness < parent2.fitness:
                    GA_efficiency += (parent2.fitness - offspring2.fitness)
                
                # Store offspring with reference to parents
                offspring1.parent_index = self.population.individuals.index(parent1)
                offspring2.parent_index = self.population.individuals.index(parent2)
                temporary_population.append(offspring1)
                temporary_population.append(offspring2)
        
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
        
        # Calculate population statistics
        mean_fitness = np.mean([ind.fitness for ind in self.population.individuals])
        self.population.calculate_diversity()
        
        # Store metrics
        self.metrics_history['best_fitness'].append(self.population.best_fitness)
        self.metrics_history['mean_fitness'].append(mean_fitness)
        self.metrics_history['diversity'].append(self.population.diversity)
        self.metrics_history['pso_efficiency'].append(PSO_efficiency)
        self.metrics_history['ga_efficiency'].append(GA_efficiency)
        
        self.iteration += 1        
        return self.population.best_fitness
    
    def run(self, max_iterations):
        """Run the optimization process for a specified number of iterations"""
        if self.population is None:
            self.initialize()
        
        best_fitness_history = []
        
        for i in range(max_iterations):
            current_best = self.step()
            best_fitness_history.append(current_best)
            
            # Print progress
            if (i) % (max_iterations//10) == 0:
                print(f"Iteration {i}/{max_iterations}, Best Fitness: {current_best}")

        
        print(f"Optimization completed after {self.iteration} iterations")
        print(f"Best solution: {self.population.best_individual.position}")
        print(f"Best fitness: {self.population.best_fitness}")
        
        return self.population.best_individual, best_fitness_history