"""Stand-in for DADApy in host tests (no jax): Data.compute_id_2NN returns the mean row norm as 'd', so tests can see what the wrapper passed in."""
import numpy as np
class Data:
    """Records its input; mimics the DADApy 0.3.4 TWO-NN signatures."""
    def __init__(self, coordinates=None, distances=None, maxk=None, period=None, verbose=False, n_jobs=8, working_memory=1024):
        self.X, self.N = np.asarray(coordinates), len(coordinates)
    def compute_id_2NN(self, mu_fraction=0.9):
        return float(np.linalg.norm(self.X, axis=1).mean()), 0.0, 0.5
    def return_id_scaling_2NN(self, n_min=10, mu_fraction=0.9):
        n = [int(round(self.N / 2 ** i)) for i in range(int(np.log2(self.N)) - 1)]
        n = [x for x in n if x > n_min]
        return np.full(len(n), 3.0), np.full(len(n), 0.1), np.linspace(0.1, 1.0, len(n))
