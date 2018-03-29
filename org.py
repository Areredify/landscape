import copy
import numpy as np


# class describing organism with single gene as floating point vector
class FGeneOrg:
    def __init__(self, gene):
        self.gene = copy.deepcopy(gene)

    def mutated_gene(self, average_mutation_num=0.1, scale=0.02):
        new_gene = copy.deepcopy(self.gene)
        prob = average_mutation_num
        new_gene += [np.random.normal(0, scale) if np.random.random() < prob else 0 for _ in range(len(new_gene))]

        return new_gene


class Bacteria(FGeneOrg):
    mut_v = 0.01
    mut_m = 0.1

    def __init__(self, gene):
        super().__init__(gene)

    def offspring(self):
        return Bacteria(super().mutated_gene(scale=self.mut_v, average_mutation_num=self.mut_m))


class Plant(FGeneOrg):
    mut_v = 0.01
    mut_m = 0.1

    def __init__(self, gene):
        super().__init__(gene)

    def offspring(self):
        return Plant(super().mutated_gene(scale=self.mut_v, average_mutation_num=self.mut_m))
