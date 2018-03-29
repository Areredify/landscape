from org import *
import multiprocessing
import itertools
from statistics import mean


def can_connect(bacteria, plant, neural_net, threshold):
    dist = np.linalg.norm(neural_net.run(bacteria.gene) - np.array(plant.gene))
    prob = np.exp(1 - dist / threshold)
    if np.random.random() < prob:
        return True
    return False


def ev_basic(bacteria_pop, plant_pop, neural_net, threshold):
    new_bacteria_pop = copy.deepcopy(bacteria_pop)
    new_plant_pop = copy.deepcopy(plant_pop)
    new_plant_pop.extend([plant.offspring() for plant in plant_pop])
    new_plant_pop.extend([plant.offspring() for plant in plant_pop])
    new_plant_pop.extend([plant.offspring() for plant in plant_pop])
    plants_repr_prob = [0] * len(plant_pop)
    for bacteria in bacteria_pop:
        samples = np.random.randint(len(plant_pop), size=3)
        for sample in samples:
            if can_connect(bacteria, plant_pop[sample], neural_net, threshold):
                new_bacteria_pop.extend([bacteria.offspring() for _ in range(4)])
                plants_repr_prob[sample] += 0.2
                break
        else:
            new_bacteria_pop.extend([bacteria.offspring() for _ in range(3)])
    for j in range(len(plant_pop)):
        if np.random.random() < plants_repr_prob[j]:
            new_plant_pop.append(plant_pop[j].offspring())

    new_bacteria_pop = list(np.random.choice(new_bacteria_pop, len(bacteria_pop), False))
    new_plant_pop = list(np.random.choice(new_plant_pop, len(plant_pop), False))
    return [new_bacteria_pop, new_plant_pop]


def ev_pooling(bacteria_pop, plant_pop, transformed_genes):
    new_bacteria_pop = copy.deepcopy(bacteria_pop)
    new_plant_pop = []
    # copy.deepcopy(plant_pop)
    new_bacteria_pop.extend([bacteria.offspring() for bacteria in bacteria_pop])
    new_plant_pop.extend([plant.offspring() for plant in plant_pop])

    pool_size = len(bacteria_pop) // len(plant_pop)
    mean_dists = []
    for i in range(len(plant_pop)):
        slc = [-np.linalg.norm(gene - plant_pop[i].gene)
               for gene in transformed_genes[pool_size * i:pool_size * (i + 1)]]
        ind = np.argpartition(slc, -4)[-4:]
        mean_dist = 0.0
        for x in ind:
            mean_dist += -slc[x]
        mean_dists.append(mean_dist / 4)
        # ind = np.random.randint(0, pool_size, 5)
        indices = [pool_size * i + j for j in ind]

        for j in indices:
            new_bacteria_pop.extend(bacteria_pop[j].offspring() for _ in range(4))

    new_bacteria_pop = list(np.random.choice(new_bacteria_pop, len(bacteria_pop), False))
    new_plant_pop = list(np.random.choice(new_plant_pop, len(plant_pop), False))
    return new_bacteria_pop, new_plant_pop, mean(mean_dists)
