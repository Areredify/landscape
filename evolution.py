from org import *
import multiprocessing
import itertools
from statistics import mean
from math import exp
import math


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

def get_probs(mean_dists):
    probs = np.array([(10 - 9 * (x ** 0.5)) if x < 1 else 1 for x in mean_dists])
    #probs = np.array([0.7 + exp(-x) for x in mean_dists])
    probs = probs / sum(probs)
    return probs

def ev_stochastic(bacteria_pop, plant_pop, t_bact_genes, t_plant_genes):
    new_bacteria_pop = copy.deepcopy(bacteria_pop)
    new_bacteria_pop.extend([bacteria.offspring() for bacteria in bacteria_pop])

    new_bacteria_pop = list(np.random.choice(new_bacteria_pop, len(bacteria_pop), False))
    new_plant_pop_ind = list(np.random.choice(range(len(plant_pop)), len(plant_pop), True))
    new_plant_pop = [plant_pop[i].offspring() for i in new_plant_pop_ind]

    return new_bacteria_pop, new_plant_pop, 0

def ev_pooling(bacteria_pop, plant_pop, t_bact_genes, t_plant_genes):
    new_bacteria_pop = copy.deepcopy(bacteria_pop)
    # new_plant_pop = copy.deepcopy(plant_pop)
    new_bacteria_pop.extend([bacteria.offspring() for bacteria in bacteria_pop])
    # new_plant_pop.extend([plant.offspring() for plant in plant_pop])

    close_bact_num = 3
    reproduction_bonus = 3

    pool_size = len(bacteria_pop) // len(plant_pop)
    mean_dists = []
    for i in range(len(plant_pop)):
        slc = [np.linalg.norm(gene - t_plant_genes[i])
               for gene in t_bact_genes[pool_size * i:pool_size * (i + 1)]]
        ind = list(np.argpartition(slc, close_bact_num)[:close_bact_num])
        mean_dist = 0.0
        for x in ind:
            mean_dist += -slc[x]
        mean_dists.append(mean_dist / len(ind))
        # ind = np.random.randint(0, pool_size, 5)
        indices = [pool_size * i + j for j in ind if slc[x] < 1] #remove

        for j in indices:
            new_bacteria_pop.extend(bacteria_pop[j].offspring() for _ in range(reproduction_bonus))

    probs = get_probs(mean_dists)
    new_bacteria_pop = list(np.random.choice(new_bacteria_pop, len(bacteria_pop), False))
    new_plant_pop_ind = list(np.random.choice(range(len(plant_pop)), len(plant_pop), True, p=probs))
    new_plant_pop = [plant_pop[i].offspring() for i in new_plant_pop_ind]

    return new_bacteria_pop, new_plant_pop, np.mean(mean_dists)

def glob_disaster(population, pop_genes, cls):
    org_ind = np.random.randint(0, len(population))
    surv_pop_size = int(0.1 * len(population))
    dists = [np.linalg.norm(pop_genes[i] - pop_genes[org_ind]) for i in range(len(population))]
    inds = list(np.argpartition(dists, surv_pop_size)[:surv_pop_size])
    new_pop = [population[i] for i in inds]

    inds = np.random.randint(0, surv_pop_size, len(population) - surv_pop_size)
    for i in range(len(population) - surv_pop_size):
        new_pop.append(new_pop[inds[i]].offspring())

    return new_pop


def local_surv(population, pop_genes, cls):
    org_ind = np.random.randint(0, len(population))
    surv_pop_size = int(0.5 * len(population))

    new_pop = list(np.random.choice(population, surv_pop_size))

    vec = np.random.random(len(pop_genes[0]))
    vec = vec / np.linalg.norm(vec) * np.random.normal(0, 0.001 * 20)

    new_gene = pop_genes[0] + vec

    new_pop = new_pop + [cls(new_gene, [], cls.last_id + 1) for _ in range(len(population) - surv_pop_size)]

    return new_pop