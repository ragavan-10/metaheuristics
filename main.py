import numpy as np
import matplotlib.pyplot as plt
from core.population import Population
from core.optimizer import Optimizer
from algorithms.hpsoe import HPSOE
from algorithms.fhpsoe import FHPSOE
from algorithms.arithga import ArithmeticGA
from algorithms.simple_pso import SimplePSO
from algorithms.simple_ga import SimpleGA
from algorithms.clpso import CLPSO
from algorithms.seqHybrid import SequentialHybrid
from algorithms.parHybrid import ParallelHybrid
from util.benchmark import benchmark_functions
# from visual.parallel_plot import add_ga_frame, add_pso_frame, reset_frames, get_frames, plot_background
from matplotlib.widgets import Slider


def run_algorithm(alg_class, obj_func, max_iterations=100, population_size=30):

    algorithm = alg_class(obj_func, population_size)
    _, hist = algorithm.run(max_iterations)
    
    results = hist[-1]
    diversity_data = (
        algorithm.metrics_history['mean_fitness'],
        algorithm.metrics_history['diversity']
    )

    return hist, diversity_data


def plot_fitness_curves(all_histories):
    for func_name, func_histories in all_histories.items():
        plt.figure(figsize=(10, 6))
        for alg_name, history in func_histories.items():
            plt.semilogy(history, label=alg_name)
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (log scale)')
        plt.title(f'{func_name} Function Optimization')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_mean_diversity(diversity_data):
    for func_name, data in diversity_data.items():
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        for alg_name, (mean_fit, div) in data.items():
            ax1.plot(mean_fit, label=f'{alg_name} - Mean Fit')
            ax2.plot(div, '--', label=f'{alg_name} - Diversity')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean Fitness')
        ax2.set_ylabel('Diversity')
        plt.title(f'{func_name} - Mean Fitness vs Diversity')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_heatmap(obj_func, bounds, resolution=200):
    X, Y, Z = plot_background(bounds, obj_func, resolution)
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.title("Objective Function Heatmap")
    plt.colorbar(label='Fitness')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dimensions = 30
    max_iterations = 20
    population_size = 50
    results_summary = {}

    # List of algorithms to run with their display names
    algorithms = [
        (HPSOE, 'HPSO-E'),
        (FHPSOE, 'FHPSO-E'),
        (SimplePSO, 'PSO'),
        (SimpleGA, 'GA'),
        (CLPSO, 'CLPSO'),
        (ArithmeticGA, 'Arithmetic GA'),
        (SequentialHybrid, 'SequentialHybrid'),
        (ParallelHybrid, 'ParallelHybrid'),
    ]

    results = {}
    all_histories = {}
    diversity_data = {}
    benchmark_registry = benchmark_functions(dimensions)
    for func_name, obj_func in list(benchmark_registry.items()):
        print("-" * 100)
        print(f"Running for {func_name}...")
        results[func_name] = {}
        all_histories[func_name] = {}
        diversity_data[func_name] = {}
        for alg_class, alg_name in algorithms:
            print("-" * 100)
            print(f"Running Algorithm {alg_name}...")
            # Run algorithms for current function
            hist, diversity = run_algorithm(
                alg_class, benchmark_registry[func_name], max_iterations, population_size)
            results[func_name][alg_name] = hist[-1]
            all_histories[func_name][alg_name] = hist
            diversity_data[func_name][alg_name] = diversity
    plot_fitness_curves(all_histories)
    plot_mean_diversity(diversity_data)

    print("\nFinal Results Summary:")
    print("-" * 100)
    header = f"{'Function':<20}"
    for _, alg_name in algorithms:
        header += f"{alg_name:<15}"
    print(header)
    print("-" * 100)
