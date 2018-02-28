import numpy as np
import copy
import sklearn.metrics
import sklearn.manifold
import matplotlib.pyplot as plt
import operator
from mpl_toolkits.mplot3d import Axes3D

import nnet

""" simulates population
    arguments:
        [int, int] pop_sizes - population sizes
        [int, int] gene_sizes - number of genes per organism
        string distrib - initial genome distribution
                         "uniform" or "normal" 
        [string, string] repr_type - sexual or asexual reproduction
                         "y" for sexual, "n" otherwise
        int gener_num - number of generations
        int N_e - get dynamics every N_e generations, -1 for none
    returns:
        two lists of numpy arrays - evolution dynamics 
"""


class FooOrg:
    def __init__(self, genes):
        self.genes = genes

    def offspring(self):
        new_genes = copy.deepcopy(self.genes)
        if np.random.random() < 0.05:
            new_genes = new_genes + 0.05 * np.random.normal(size=len(new_genes))
        else:
            new_genes = new_genes + 0.003 * np.random.normal(size=len(new_genes))
        return FooOrg(new_genes)


class BarOrg:
    def __init__(self, genes):
        self.genes = genes

    def offspring(self):
        new_genes = copy.deepcopy(self.genes)
        if np.random.random() < 0.05:
            new_genes = new_genes + 0.05 * np.random.random(size=len(new_genes)) - 0.025
        else:
            new_genes = new_genes + 0.0001 * np.random.random(size=len(new_genes)) - 0.00005
        return BarOrg(new_genes)


def can_connect(foo_org, bar_org, neural_net, threshold):
    dist = np.linalg.norm(nnet.run_neural_net(foo_org.genes, neural_net) - np.array(bar_org.genes))
    prob = np.exp(1 - dist / threshold)
    if np.random.random() < prob:
        return True
    return False


def pop_to_distances(pop):
    if type(pop[0]) is list or type(pop[0]) is tuple or type(pop[0]) is np.ndarray:
        return sklearn.metrics.pairwise.euclidean_distances(np.array([np.array(x) for x in pop]))
    else:
        return sklearn.metrics.pairwise.euclidean_distances(np.array([np.array(x.genes) for x in pop]))


def print_debug_info(foo_pop, bar_pop, generation, gener_num, neural_net, threshold):
    print(str(int(generation / gener_num * 100)) + "%")
    foo_pop_neural = []
    close = 0
    av = 0
    for t in foo_pop:
        vec = nnet.run_neural_net(t.genes, neural_net)
        foo_pop_neural.append(vec)
        dists = [np.linalg.norm(vec - k.genes) for k in bar_pop]
        min_index, min_value = min(enumerate(dists), key=operator.itemgetter(1))
        close += int(min_value < threshold)
        av += min_value / len(foo_pop)
    print(close / len(foo_pop), av)


def simulate(pop_sizes=(800, 60), gene_sizes=(10, 10),
             repr_type=("n", "y"), threshold=0.09, gener_num=10000):
    FooOrg.repr_type = repr_type[0]
    BarOrg.repr_type = repr_type[1]
    foo_pop = [FooOrg(i) for i in np.random.uniform(-1, 1, (pop_sizes[0], gene_sizes[0]))]
    bar_pop = [BarOrg(i) for i in np.random.uniform(0, 1, (pop_sizes[1], gene_sizes[1]))]
    neural_net = nnet.create_neural_net(*gene_sizes)
    for i in range(gener_num):
        if int(i / gener_num * 50) > int((i - 1) / gener_num * 50) or i == 0:
            print_debug_info(foo_pop, bar_pop, i, gener_num, neural_net, threshold)
        new_foo_pop = copy.deepcopy(foo_pop)
        new_bar_pop = copy.deepcopy(bar_pop)
        new_bar_pop = np.append(new_bar_pop, [bar_org.offspring() for bar_org in bar_pop])
        succ_connections = 0
        for foo_org in foo_pop:
            samples = np.random.choice(bar_pop, 3)
            for sample in samples:
                if can_connect(foo_org, sample, neural_net, threshold):
                    new_foo_pop = np.append(new_foo_pop, [sample.offspring() for i in range(7)])
                    new_bar_pop = np.append(new_bar_pop, [sample.offspring()])
                    succ_connections += 1
                    break
            else:
                new_foo_pop = np.append(new_foo_pop, [foo_org.offspring() for i in range(5)])
        if int(i / gener_num * 1000) > int((i - 1) / gener_num * 1000) or i == 0:
            ttt = succ_connections / len(foo_pop)
            print(ttt)

        foo_pop = np.random.choice(new_foo_pop, pop_sizes[0], False)
        bar_pop = np.random.choice(new_bar_pop, pop_sizes[1], False)

    foo_pop_neural = []
    colors = []
    close = 0
    for t in foo_pop:
        vec = nnet.run_neural_net(t.genes, neural_net)
        foo_pop_neural.append(vec)
        dists = [np.linalg.norm(vec - k.genes) for k in bar_pop]
        min_index, min_value = min(enumerate(dists), key=operator.itemgetter(1))
        close += int(min_value < threshold)
        colors.append([min_index, min_value < threshold])
    print(close / pop_sizes[0])
    return [[pop_to_distances(foo_pop), pop_to_distances(foo_pop_neural), pop_to_distances(bar_pop)],
            colors]

np.random.seed(8)
distances = simulate()
colors = distances[1]
distances = distances[0]
print("*")
mds = sklearn.manifold.MDS(n_components=3, max_iter=3000, eps=1e-4,
                           dissimilarity="precomputed", n_jobs=1)

color_map = plt.cm.get_cmap('hsv', len(distances[2]) + 1)
print(color_map(0))
foo_col = [color_map((i[0] + 1) * int(i[1])) for i in colors]
bar_col = [color_map(i + 1) for i in range(len(distances[2]))]

pop_colors = [foo_col, foo_col, bar_col]

for i in range(len(distances)):
    print(i)
    res = mds.fit_transform(distances[i])
    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')

    xx = res[:, 0]
    yy = res[:, 1]
    zz = res[:, 2]
    ax.scatter(xx, yy, zz, color=pop_colors[i])

plt.show()
