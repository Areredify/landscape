import copy
import numpy as np

class FGeneOrg:
    def __init__(self, gene):
        self.gene = copy.deepcopy(gene)

    def mutated_gene(self, average_mutation_num=0.1, scale=0.02):
        new_gene = copy.deepcopy(self.gene)
        prob = average_mutation_num
        flag = False
        for i in range(len(new_gene)):
            if np.random.random() < prob:
                new_gene[i] += np.random.normal(0, scale)
                flag = True

        return new_gene, flag


class Bacteria(FGeneOrg):
    mut_v = 0.01
    mut_m = 0.1
    last_id = 0
    dist_epochs = 10

    def __init__(self, gene, parent_ids, g_id = None):
        super().__init__(gene)
        if g_id is None:
            Bacteria.last_id += 1
            g_id = Bacteria.last_id
        self.g_id = g_id
        self.parent_ids = parent_ids

    def offspring(self):
        new_gene, changed = super().mutated_gene(scale=self.mut_v, average_mutation_num=self.mut_m)
        if (len(self.parent_ids) < Bacteria.dist_epochs):
            parent_ids = self.parent_ids + [self.g_id]
        else:
            parent_ids = self.parent_ids[1:] + [self.g_id]

        if not changed:
            return Bacteria(new_gene, parent_ids, self.g_id)
        else:
            return Bacteria(new_gene, parent_ids)


class Plant(FGeneOrg):
    mut_v = 0.01
    mut_m = 0.1
    last_id = 0
    dist_epochs = 10

    def __init__(self, gene, parent_ids, g_id = None):
        super().__init__(gene)
        if g_id is None:
            Plant.last_id += 1
            g_id = Plant.last_id
        self.g_id = g_id
        self.parent_ids = parent_ids

    def offspring(self):
        new_gene, changed = super().mutated_gene(scale=self.mut_v, average_mutation_num=self.mut_m)

        if (len(self.parent_ids) < Plant.dist_epochs):
            parent_ids = self.parent_ids + [self.g_id]
        else:
            parent_ids = self.parent_ids[1:] + [self.g_id]

        if not changed:
            return Plant(new_gene, parent_ids, self.g_id)
        else:
            return Plant(new_gene, parent_ids)


