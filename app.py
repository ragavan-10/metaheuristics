import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
import time

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
from util.benchmark import benchmark_registry




def run_algorithm(alg_class, obj_func, max_iterations=100, population_size=30):

    algorithm = alg_class(obj_func, population_size)
    _, hist = algorithm.run(max_iterations)
    
    results = hist[-1]
    diversity_data = (
        algorithm.metrics_history['mean_fitness'],
        algorithm.metrics_history['diversity']
    )

    return hist, diversity_data


def plot_fitness_curves(all_histories, function_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in all_histories.items():
        ax.semilogy(history, label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Fitness (log scale)')
    ax.set_title(f'{function_name} Function Optimization')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_mean_diversity(diversity_data, function_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(diversity_data)))
    
    for i, (name, (mean_fit, div)) in enumerate(diversity_data.items()):
        ax1.plot(mean_fit, label=f'{name} - Mean Fit', color=colors[i])
        ax2.plot(div, label=f'{name} - Diversity', color=colors[i])
    
    ax1.set_ylabel('Mean Fitness')
    ax1.set_title(f'{function_name} - Mean Fitness Over Iterations')
    ax1.grid(True)
    ax1.legend(loc='upper right')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Diversity')
    ax2.set_title(f'{function_name} - Diversity Over Iterations')
    ax2.grid(True)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    return fig



def display_results_table(results, function_name):
    # Convert results to DataFrame for better display
    df = pd.DataFrame({algo: [val] for algo, val in results.items()})
    df.index = [function_name]
    return df


# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Optimization Algorithms Comparison",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Hybrid PSO, GA with Fuzzy logic")

    # Sidebar for parameters
    st.sidebar.header("Parameters")
    
    # Define benchmark functions (would be replaced with your actual benchmark registry)
    benchmark_functions = list(benchmark_registry.keys())

    # List of algorithms to run with their display names
    algorithms = [
        (HPSOE, 'HPSO-E'),
        (FHPSOE, 'FHPSO-E'),
        (SimplePSO, 'PSO'),
        (SimpleGA, 'GA'),
        (CLPSO, 'CLPSO'),
        (ArithmeticGA, 'Arithmetic GA'),
        (SequentialHybrid, 'Sequential Hybrid'),
        (ParallelHybrid, 'Parallel Hybrid')
    ]

    
    selected_function = st.sidebar.selectbox("Benchmark Function", benchmark_functions)
    
    dimensions = st.sidebar.slider("Dimensions", 2, 100, 30)
    custom_steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    max_iterations = st.sidebar.select_slider(
        "Max Iterations",
        options=custom_steps,
        value=100
    )
    population_size = st.sidebar.slider("Population Size", 10, 200, 50)
    
    # Buttons to run the algorithms
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        run_single = st.button("Run Single Function")
    
    with col2:
        run_all = st.button("Run All Functions")
    
    if run_single:
        with st.spinner(f"Running optimization algorithms on {selected_function} function..."):
            # Progress bar for visualization
            progress_bar = st.progress(0)
            
            results = {}
            all_histories = {}
            diversity_data = {}
            for j, (alg_class, alg_name) in enumerate(algorithms):
            # This would be replaced with your actual optimization run
                hist, diversity = run_algorithm(
                    alg_class, benchmark_registry[selected_function],  max_iterations, population_size)
                results[alg_name] = hist[-1]
                all_histories[alg_name] = hist
                diversity_data[alg_name] = diversity
            # Simulating computation time for better UX
            for i in range(10):
                time.sleep(0.1)
                progress_bar.progress((i + 1) * 10)
            
            # Store results for the selected function
            if 'all_function_results' not in st.session_state:
                st.session_state['all_function_results'] = {}
                
            st.session_state['all_function_results'][selected_function] = {
                'results': results,
                'all_histories': all_histories,
                'diversity_data': diversity_data
            }
            
            st.session_state['last_run_function'] = selected_function
            
            # Clear progress bar
            progress_bar.empty()
            
            # Force rerun to update the page
            st.rerun()
    
    elif run_all:
        # Initialize container for all results if not exists
        if 'all_function_results' not in st.session_state:
            st.session_state['all_function_results'] = {}
            
        # Run optimization for all benchmark functions
        with st.spinner(f"Running optimization algorithms on all {len(benchmark_functions)} functions..."):
            progress_bar = st.progress(0)
            
            for i, func_name in enumerate(benchmark_functions):
                results = {}
                all_histories = {}
                diversity_data = {}
                for j, (alg_class, alg_name) in enumerate(algorithms):
                    # Run algorithms for current function
                    hist, diversity = run_algorithm(
                        alg_class, benchmark_registry[func_name], max_iterations, population_size)
                    results[alg_name] = hist[-1]
                    all_histories[alg_name] = hist
                    diversity_data[alg_name] = diversity

                    # Update progress
                    progress_bar.progress(((j+1)+(len(algorithms))*(i)) / (len(benchmark_functions)*len(algorithms)))
                    
                # Store results
                st.session_state['all_function_results'][func_name] = {
                    'results': results,
                    'all_histories': all_histories,
                    'diversity_data': diversity_data
                }
                

            
            st.session_state['run_all_completed'] = True
            
            # Clear progress bar
            progress_bar.empty()
            
            # Force rerun to update the page
            st.rerun()
    
    # Display results if available
    if 'all_function_results' in st.session_state and st.session_state['all_function_results']:
        # Create tabs for each function
        function_tabs = st.tabs(list(st.session_state['all_function_results'].keys()))
        
        # Display results for each function in its own tab
        for idx, func_name in enumerate(st.session_state['all_function_results'].keys()):
            with function_tabs[idx]:
                func_data = st.session_state['all_function_results'][func_name]
                
                st.subheader("Convergence Plot")
                fig = plot_fitness_curves(func_data['all_histories'], func_name)
                st.pyplot(fig)
                
                st.subheader("Diversity Analysis")
                fig = plot_mean_diversity(func_data['diversity_data'], func_name)
                st.pyplot(fig)
                
                
                # Results table
                st.subheader(f"Final Results")
                df = display_results_table(func_data['results'], func_name)
                st.dataframe(df.style.format("{:.6e}"))
                
                # Display algorithm ranking
                st.subheader("Algorithm Ranking (Best to Worst)")
                sorted_results = sorted(func_data['results'].items(), key=lambda x: x[1])
                
                # Create a ranking table
                ranking_data = [(i+1, algo, value) for i, (algo, value) in enumerate(sorted_results)]
                ranking_df = pd.DataFrame(ranking_data, columns=["Rank", "Algorithm", "Best Fitness"])
                st.table(ranking_df.set_index("Rank").style.format({"Best Fitness": "{:.6e}"}))
                
        # Add a summary tab with comparative analysis if all functions have been run
        if 'run_all_completed' in st.session_state and st.session_state['run_all_completed']:
            with st.expander("Comparative Analysis Across All Functions", expanded=True):
                st.subheader("Algorithm Performance Comparison")
                
                # Create a summary dataframe with all results
                summary_data = {}
                for func_name, func_data in st.session_state['all_function_results'].items():
                    for algo, result in func_data['results'].items():
                        if algo not in summary_data:
                            summary_data[algo] = {}
                        summary_data[algo][func_name] = result
                
                # Create summary dataframe
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df.style.format("{:.6e}"))
                
                # Visualization of summary data (optional)
                st.subheader("Overall Performance Ranking")
                
                # Calculate average ranks for each algorithm
                avg_ranks = {}
                for func_name, func_data in st.session_state['all_function_results'].items():
                    sorted_results = sorted(func_data['results'].items(), key=lambda x: x[1])
                    for rank, (algo, _) in enumerate(sorted_results):
                        if algo not in avg_ranks:
                            avg_ranks[algo] = []
                        avg_ranks[algo].append(rank + 1)  # Add 1 since rank starts at 0
                
                # Calculate average rank for each algorithm
                avg_rank_data = [(algo, np.mean(ranks)) for algo, ranks in avg_ranks.items()]
                avg_rank_df = pd.DataFrame(sorted(avg_rank_data, key=lambda x: x[1]), 
                                          columns=["Algorithm", "Average Rank"])
                
                st.table(avg_rank_df.set_index("Algorithm").style.format({"Average Rank": "{:.2f}"}))
                
    else:
        # Initial instructions
        st.info("👈 Set parameters in the sidebar and click 'Run Single Function' or 'Run All Functions' to start")
        st.image("https://via.placeholder.com/800x400?text=Select+Parameters+and+Run+Optimization", use_container_width=True)


if __name__ == "__main__":
    main()