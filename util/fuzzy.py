import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FuzzySystem:
    def __init__(self):
        # Initialize the fuzzy system with membership functions from the paper
        # Specifically from Figure 4 which shows the final trained system

        # Input 1: pe (influence factor)
        # Membership functions for small, medium, large
        self.pe_small = {"type": "L", "params": [0.35, 0.1]}  # L-function with a=0.35, b=0.1
        self.pe_medium = {"type": "triangle", "params": [0.1, 0.5, 0.8]}  # Triangle with a=0.1, b=0.5, c=0.8
        self.pe_large = {"type": "gamma", "params": [0.6, 0.95]}  # Gamma-function with a=0.6, b=0.95

        # Input 2: delta_en_ga (normalized efficiency)
        # Membership functions for small, medium, large
        self.en_small = {"type": "L", "params": [0.5, 0.1]}  # L-function with a=0.5, b=0.1
        self.en_medium = {"type": "triangle", "params": [0.2, 0.5, 0.8]}  # Triangle with a=0.2, b=0.5, c=0.8
        self.en_large = {"type": "gamma", "params": [0.5, 0.9]}  # Gamma-function with a=0.5, b=0.9

        # Output: delta_pe (change in influence factor)
        # Singleton membership functions
        self.delta_pe_outputs = [-0.1, -0.05, 0, 0.05, 0.1]  # Values from Figure 4

        # The rule base from Table I
        # Format: [pe_membership, en_membership, output_index]
        self.rules = [
            ["small", "small", 3],  # If pe is small and ΔEN_GA is small then Δpe is 0.05
            ["small", "medium", 4],  # If pe is small and ΔEN_GA is medium then Δpe is 0.1
            ["small", "large", 4],   # If pe is small and ΔEN_GA is large then Δpe is 0.1
            ["medium", "small", 2],  # If pe is medium and ΔEN_GA is small then Δpe is 0
            ["medium", "medium", 3], # If pe is medium and ΔEN_GA is medium then Δpe is 0.05
            ["medium", "large", 4],  # If pe is medium and ΔEN_GA is large then Δpe is 0.1
            ["large", "small", 0],   # If pe is large and ΔEN_GA is small then Δpe is -0.1
            ["large", "medium", 1],  # If pe is large and ΔEN_GA is medium then Δpe is -0.05
            ["large", "large", 2],   # If pe is large and ΔEN_GA is large then Δpe is 0
        ]

    def _l_membership(self, x, params):
        """Calculate L-function membership"""
        a, b = params
        if x <= b:
            return 1.0
        elif x >= a:
            return 0.0
        else:
            return (a - x) / (a - b)

    def _triangle_membership(self, x, params):
        """Calculate triangle membership function"""
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)

    def _gamma_membership(self, x, params):
        """Calculate gamma membership function"""
        a, b = params
        if x <= a:
            return 0.0
        elif x >= b:
            return 1.0
        else:
            return (x - a) / (b - a)

    def get_membership_value(self, x, fuzzy_set, input_type):
        """Calculate membership value for a given input and fuzzy set"""
        if input_type == "pe":
            if fuzzy_set == "small":
                return self._l_membership(x, self.pe_small["params"])
            elif fuzzy_set == "medium":
                return self._triangle_membership(x, self.pe_medium["params"])
            elif fuzzy_set == "large":
                return self._gamma_membership(x, self.pe_large["params"])
        elif input_type == "en":
            if fuzzy_set == "small":
                return self._l_membership(x, self.en_small["params"])
            elif fuzzy_set == "medium":
                return self._triangle_membership(x, self.en_medium["params"])
            elif fuzzy_set == "large":
                return self._gamma_membership(x, self.en_large["params"])
        return 0.0

    def infer(self, pe, delta_en_ga):
        """Apply fuzzy inference to get delta_pe"""
        # Calculate rule firing strengths
        firing_strengths = []
        for rule in self.rules:
            pe_set, en_set, output_idx = rule
            pe_membership = self.get_membership_value(pe, pe_set, "pe")
            en_membership = self.get_membership_value(delta_en_ga, en_set, "en")
            
            # Use min operator for AND
            rule_strength = min(pe_membership, en_membership)
            firing_strengths.append((rule_strength, output_idx))

        # Apply center of averages defuzzification
        numerator = 0.0
        denominator = 0.0
        
        for strength, output_idx in firing_strengths:
            if strength > 0:
                y_k = self.delta_pe_outputs[output_idx]
                numerator += y_k * strength
                denominator += strength
        
        if denominator == 0:
            return 0.0  # Default value if no rules fire
        
        delta_pe = numerator / denominator
        return delta_pe

    def plot_membership_functions(self):
        """Plot the membership functions for visualization"""
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot pe membership functions
        x_pe = np.linspace(0, 1, 100)
        small_pe = [self.get_membership_value(x, "small", "pe") for x in x_pe]
        medium_pe = [self.get_membership_value(x, "medium", "pe") for x in x_pe]
        large_pe = [self.get_membership_value(x, "large", "pe") for x in x_pe]
        
        axs[0].plot(x_pe, small_pe, 'r-', label='Small')
        axs[0].plot(x_pe, medium_pe, 'g-', label='Medium')
        axs[0].plot(x_pe, large_pe, 'b-', label='Large')
        axs[0].set_title('Influence Factor (pe)')
        axs[0].legend()
        
        # Plot delta_en_ga membership functions
        x_en = np.linspace(0, 1, 100)
        small_en = [self.get_membership_value(x, "small", "en") for x in x_en]
        medium_en = [self.get_membership_value(x, "medium", "en") for x in x_en]
        large_en = [self.get_membership_value(x, "large", "en") for x in x_en]
        
        axs[1].plot(x_en, small_en, 'r-', label='Small')
        axs[1].plot(x_en, medium_en, 'g-', label='Medium')
        axs[1].plot(x_en, large_en, 'b-', label='Large')
        axs[1].set_title('Normalized Efficiency (delta_en_ga)')
        axs[1].legend()
        
        # Plot output singletons
        outputs = self.delta_pe_outputs
        axs[2].stem(range(len(outputs)), outputs)
        axs[2].set_xticks(range(len(outputs)))
        axs[2].set_xticklabels(['B1', 'B2', 'B3', 'B4', 'B5'])
        axs[2].set_title('Output Singletons (delta_pe)')
        
        plt.tight_layout()
        plt.show()
        
    def plot_decision_surface(self):
        """Plot the decision surface of the fuzzy system"""
        pe_range = np.linspace(0, 1, 50)
        en_range = np.linspace(0, 1, 50)
        pe_grid, en_grid = np.meshgrid(pe_range, en_range)
        
        delta_pe_grid = np.zeros_like(pe_grid)
        
        for i in range(len(pe_range)):
            for j in range(len(en_range)):
                delta_pe_grid[j, i] = self.infer(pe_grid[j, i], en_grid[j, i])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(pe_grid, en_grid, delta_pe_grid, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Influence Factor (pe)')
        ax.set_ylabel('Normalized Efficiency (delta_en_ga)')
        ax.set_zlabel('Change in Influence Factor (delta_pe)')
        ax.set_title('Fuzzy System Decision Surface')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()
