# Metaheuristics

A python based utility for some common varaitions of Genetic Algorithm and Particle Swarm Optimizer

This project implements and benchmarks several evolutionary and swarm optimization algorithms, including hybrid variants such as FHPSOE. It provides visual and statistical comparisons on standard benchmark functions like Ackley, Rastrigin, Rosenbrock, and Sphere.

## Project Structure

``` bash
project/
├── main.py                   # Main script to run benchmarks
├── app.py                    # Auxiliary application logic
├── algorithms/               # Contains all optimization algorithm implementations
├── core/                     # Core components like population and optimizer classes
│   ├─ optimizer.py
│   ├─ population.py
├── util/                     # Utility functions and benchmark definitions
│   ├─ fuzzy.py
│   ├─ benchmark.py
├── Output/                   # Visualization outputs of algorithm performance
```

## How to Run

### Requirements

Ensure you have Python 3.7+ installed, then install the dependencies:

```bash
pip install numpy matplotlib streamlit
````

### Running the Benchmark

To run the benchmark across all algorithms on predefined objective functions:

```bash
python main.py
```

This will:

1. Execute multiple optimization algorithms on benchmark functions.
2. Record the best solution found and diversity metrics over generations.
3. Generate performance plots and display them.

### GUI

To run the GUI web application built using Streamlit:

```bash
streamlit run app.py
```

This will:

1. Create a web server on localhost, listening on port 8501.
2. The UI lets you customize the optimization parameters like population size, dimensions, and max iterations.
3. It also allows you to select which benchmark functions to run the algorithms on from a predefined set.

## Algorithms Compared

The following algorithms are included:

* **FHPSOE**: Feature-level Hybrid PSO-GA (main focus)
* **SimpleGA**: Basic Genetic Algorithm
* **SimplePSO**: Basic Particle Swarm Optimization
* **ArithmeticGA**: Genetic Algorithm with arithmetic crossover
* **CLPSO**: Comprehensive Learning PSO
* **HPSOE**: Hybrid PSO with embedded GA features
* **SequentialHybrid**: Sequential GA → PSO hybrid
* **ParallelHybrid**: GA and PSO run in parallel with periodic exchange

## Benchmarks Used

The following benchmark functions are considered:

* **Sphere**: A simple convex function used to test exploitation, defined as the sum of squared variables.
* **Step**: A discontinuous function with flat plateaus, used to test precision and robustness to non-smoothness.
* **Rastrigin**: A highly multimodal function with a large search space and frequent local minima.
* **Rosenbrock**: A non-convex valley-shaped function used to test convergence along curved paths.
* **Ackley**: A multimodal function with a nearly flat outer region and a large number of local minima.
* **Griewank**: A function combining polynomial and oscillatory components, testing complex variable interaction.

## Results

The output includes plots like:

* Convergence graph over iterations
* Mean & diversity metrics
* Overall results in tabular format

## Key Insight

From the results, **FHPSOE** consistently outperforms other algorithms, achieving better convergence rates and maintaining higher population diversity throughout the optimization process.
