import copy
import numpy as np
import math
import numpy.random
import org


class FourierFitness:
    def __init__(self, gene_sizes, alpha=2, tau=1, func_num=6):
        self.gene_sizes = copy.deepcopy(gene_sizes)

        self.a_xi = np.array([[[k ** (-alpha) * np.random.uniform(-1, 1) for k in range(1, func_num + 1)]
                               for _ in range(gene_sizes[0])]
                              for _ in range(gene_sizes[1])])
        self.lin_c = np.array([[[2 * math.pi / tau * k for k in range(1, func_num + 1)]
                                for _ in range(gene_sizes[0])]
                               for _ in range(gene_sizes[1])])
        self.const_c = np.array([[[2 * math.pi * np.random.random() for k in range(1, func_num + 1)]
                                  for _ in range(gene_sizes[0])]
                                 for _ in range(gene_sizes[1])])

    def run(self, vec):
        return np.sum(np.cos(self.lin_c * vec[:, np.newaxis] + self.const_c) * self.a_xi, axis=(1, 2))
