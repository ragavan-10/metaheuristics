import numpy as np

class ObjectiveFunction:
    def __init__(self, function, dimensions, bounds, ints):
        """
        Parameters:
            bounds (list of tuples): List of (min, max) bounds for each dimension.
            function (callable): The function to evaluate.
        """
        self.bounds = bounds
        self.function = function
        self.dim = dimensions
        self.ints = ints

    def evaluate(self, x):
        """
        Evaluate the function at the given point 'x', ensuring it stays within bounds.
        """
        x = self.clip_to_bounds(x)
        return self.function(*x)

    def clip_to_bounds(self, x):
        for i in range(self.dim):
            lower, upper = self.bounds[i]
            x[i] = max(lower, min(upper, x[i]))
            if self.ints[i]:
                x[i] = int(round(x[i]))
        return x

    def get_dim(self):
        return self.dim


def sphere(*x):
    x = np.asarray(x)
    return np.sum(x ** 2)

def rastrigin(*x):
    x = np.asarray(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

def ackley(*x):
    x = np.asarray(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) \
           - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def rosenbrock(*x):
    x = np.asarray(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

def griewank(*x):
    x = np.asarray(x)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1)))) + 1

def step(*x):
    x = np.asarray(x)
    return np.sum(np.floor(x + 0.5) ** 2)


def benchmark_functions(dim):
    # Instantiate the ObjectiveFunction objects for each benchmark
    sphere_benchmark = ObjectiveFunction(sphere, dim, [(-100, 100)]*dim, [False]*dim)
    rastrigin_benchmark = ObjectiveFunction(rastrigin, dim, [(-5, 5)]*dim, [False]*dim)
    ackley_benchmark = ObjectiveFunction(ackley, dim, [(-32, 32)]*dim, [False]*dim)
    rosenbrock_benchmark = ObjectiveFunction(rosenbrock, dim,[(-2, 2)]*dim, [False]*dim)
    griewank_benchmark = ObjectiveFunction(griewank, dim, [(-600, 600)]*dim, [False]*dim)
    step_benchmark = ObjectiveFunction(step, dim, [(-100, 100)]*dim, [False]*dim)


    # Store the benchmarks in a registry
    benchmark_registry = {
        "Sphere": sphere_benchmark,
        "Rastrigin": rastrigin_benchmark,
        "Ackley": ackley_benchmark,
        "Rosenbrock": rosenbrock_benchmark,
        "Griewank": griewank_benchmark,
        "Step": step_benchmark
    }

    return benchmark_registry

benchmark_registry = benchmark_functions(30)