from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
import numpy as np

class HundredsDiscreteSampling(Sampling):
    """
    A custom sampling method that generates values discretized to hundreds.
    This samples continuous values and then rounds them to the nearest hundredth.
    """
    
    def __init__(self, initial_population=None):
        super().__init__()
        self.float_sampling = FloatRandomSampling()
        self.initial_population = initial_population
        
    def _do(self, problem, n_samples, **kwargs):
        if self.initial_population is not None:
            X = np.array(self.initial_population)
            self.initial_population = None  # Reset after first use
            # Ensure initial population respects bounds BEFORE discretization
            X = np.clip(X, problem.xl, problem.xu)
            # Discretize to hundreds
            X = np.round(X * 100) / 100
            # CRITICAL: Enforce bounds AFTER discretization to handle rounding errors
            X = np.clip(X, problem.xl, problem.xu)
        else:
            X = self.float_sampling._do(problem, n_samples, **kwargs)
            # Enforce bounds BEFORE discretization
            X = np.clip(X, problem.xl, problem.xu)
            # Discretize to hundreds
            X = np.round(X * 100) / 100
            # CRITICAL: Enforce bounds AFTER discretization to handle rounding errors
            X = np.clip(X, problem.xl, problem.xu)
        
        return X

class HundredsDiscreteMutation(Mutation):
    """
    A custom mutation operator that ensures mutated values are discretized to hundreds.
    """
    
    def __init__(self, eta=3, prob=0.95):
        super().__init__()
        # Use the polynomial mutation as a base
        self.pm = PM(eta=eta, prob=prob)
        
    def _do(self, problem, X, **kwargs):
        # Apply standard polynomial mutation
        Y = self.pm._do(problem, X, **kwargs)
        
        # Enforce bounds BEFORE discretization
        Y = np.clip(Y, problem.xl, problem.xu)
        
        # Discretize to hundreds
        Y = np.round(Y * 100) / 100
        
        # CRITICAL: Enforce bounds AFTER discretization to handle rounding errors
        Y = np.clip(Y, problem.xl, problem.xu)
        
        return Y

class HundredsDiscreteCrossover(Crossover):
    """
    A custom crossover operator that ensures offspring values are discretized to hundreds.
    """
    
    def __init__(self, eta=1, prob=0.80):
        super().__init__(2, 2)  # two parents, two offspring
        # Use the simulated binary crossover as a base
        self.sbx = SBX(eta=eta, prob=prob)
        
    def _do(self, problem, X, **kwargs):
        # Apply standard SBX crossover
        Y = self.sbx._do(problem, X, **kwargs)
        
        # Enforce bounds BEFORE discretization
        Y = np.clip(Y, problem.xl, problem.xu)
        
        # Discretize to hundreds
        Y = np.round(Y * 100) / 100
        
        # CRITICAL: Enforce bounds AFTER discretization to handle rounding errors
        Y = np.clip(Y, problem.xl, problem.xu)
        
        return Y
