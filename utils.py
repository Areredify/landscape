import numpy as np
import numpy.random
import org
import numpy.linalg as npl


def gen_random(pop_sizes, fitness):
    bacteria_pop = [org.Bacteria(i) for i in np.random.uniform(-1, 1, (pop_sizes[0], fitness.gene_sizes[0]))]
    transformed_genes = [fitness.run(bacteria.gene) for bacteria in bacteria_pop]
    plant_pop = [org.Plant(transformed_genes[x] + np.random.uniform(-0.001, 0.001, fitness.gene_sizes[1]))
                 for x in np.random.choice(range(pop_sizes[0]), size=pop_sizes[1])]

    return bacteria_pop, plant_pop


def gen_same_bacteria(pop_sizes, fitness):
    haplotype = np.random.uniform(-1, 1, fitness.gene_sizes[0])
    bacteria_pop = [org.Bacteria(haplotype) for _ in range(pop_sizes[0])]
    plant_pop = [org.Plant(gene)
                 for gene in np.random.uniform(-1, 1, (pop_sizes[1], fitness.gene_sizes[1]))]
    return bacteria_pop, plant_pop


def stat_av_dist_to_closest(transformed_genes, plant_pop, epoch, **kwargs):
    sm = 0
    for trans_gene in transformed_genes:
        mn = 10000
        for plant in plant_pop:
            mn = min(mn, npl.norm(trans_gene - plant.gene))
        sm += mn
    return epoch, sm / len(transformed_genes)


def stat_av_dist_to_anc(bacteria_pop, ancestral, epoch, **kwargs):
    return epoch, np.mean([npl.norm(bacteria.gene - ancestral.gene) for bacteria in bacteria_pop])


def stat_plant_percentage(transformed_genes, plant_pop, epoch, **kwargs):
    res = [0] * len(plant_pop)
    for trans_gene in transformed_genes:
        mn = 10000
        ind = -1
        for i in range(len(plant_pop)):
            c = npl.norm(trans_gene - plant_pop[i].gene)
            if mn > c:
                mn = c
                ind = i
        res[ind] = 1
    return epoch, sum(res) / len(res)


def stat_mean_dist(mean_dist, epoch, **kwargs):
    return epoch, mean_dist
