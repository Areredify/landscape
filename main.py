import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import datetime
import time
import itertools
import nnet
from org import *
import evolution
import fourier
import utils
import multiprocessing
import os.path
import os
import matplotlib
import inspect
import collections
import scipy.spatial

def simulate_float(pop_sizes, gene_sizes, inter_size, generations, mut_params,
                   ev_alg=evolution.ev_basic, bact_fitness=None, plant_fitness=None,
                   generator=utils.gen_random,
                   init_seed=0, ev_seed=0, n_threads=1):
    Bacteria.mut_v = mut_params[0][0]
    Bacteria.mut_m = mut_params[0][1]
    Plant.mut_v = mut_params[1][0]
    Plant.mut_m = mut_params[1][1]

    dist_epochs = 100
    Bacteria.dist_epochs = dist_epochs
    Plant.dist_epochs = dist_epochs

    cnt = 0
    #try to generate populations with necesserary distance between them
    while (1):
        cnt += 1
        bacteria_pop, plant_pop = generator(pop_sizes, gene_sizes)
        d = np.linalg.norm(bact_fitness.run(bacteria_pop[0].gene) - plant_fitness.run(plant_pop[0].gene))
        if ((d > 1.26 and d < 1.36) or cnt > 100000):
            break
    np.random.seed(init_seed)
    if (generator == utils.gen_same_bacteria):
        ancestral = copy.deepcopy(bacteria_pop[0])

    time_sum = 0

    np.random.seed(ev_seed)
    picname = os.path.join("pic", str(mut_params) + "_" + str(ev_seed))
    if not os.path.exists(picname):
        os.mkdir(picname)
    print(picname)

    probs_lines = inspect.getsource(evolution.get_probs)

    bact_db, plant_db = dict(), dict()
    t_bact_db, t_plant_db = dict(), dict()

    fout = open(os.path.join(picname, "statistics.txt"), "w")
    print("bact_speed", "plant_speed", "inter_sim", "bact_div", "plant_div", "bact_ent", "plant_ent", file = fout)

    print(np.linalg.norm(bact_fitness.run(bacteria_pop[0].gene) - plant_fitness.run(plant_pop[0].gene)))

    dlog = open(os.path.join(picname, "log.txt"), "w")

    for epoch in range(generations):
        start_time = time.time()
        print(epoch)

        bact_db, plant_db = utils.update_db(bacteria_pop, plant_pop, bact_db, plant_db, epoch, dist_epochs)
        b_d, p_d = utils.normed_distance_db(bacteria_pop, plant_pop, bact_db, plant_db)
        print(b_d, file = fout, end = " ")
        print(p_d, file = fout, end = " ")

        bact_genes = np.array([bacteria.gene for bacteria in bacteria_pop])
        plant_genes = np.array([plant.gene for plant in plant_pop])
        t_bact_genes, t_bact_db = utils.update_t_db(bacteria_pop, t_bact_db, bact_fitness, epoch)
        t_plant_genes, t_plant_db = utils.update_t_db(plant_pop, t_plant_db, plant_fitness, epoch)

        t_b_probs, t_p_probs = utils.inter_labeling(t_bact_genes, t_plant_genes, 12)
        b_probs, p_probs = utils.intra_labeling(bact_genes, 12), utils.intra_labeling(plant_genes, 12)

        inter_dist = utils.hell_dist(t_b_probs, t_p_probs)
        print(inter_dist, file = fout, end = " ")

        bact_div, plant_div = utils.mc_diversity(bacteria_pop, 10000), utils.mc_diversity(plant_pop, 10000)
        print(bact_div, file = fout, end = " ")
        print(plant_div, file = fout, end = " ")

        bact_ent, plant_ent = utils.pop_entropy(b_probs), utils.pop_entropy(p_probs)
        print(bact_ent, file = fout, end = " ")
        print(plant_ent, file = fout)

        utils.flush_file(fout)

        if (inter_size == 2):
            utils.save_2d(t_bact_genes, t_plant_genes, epoch, picname, str(gene_sizes) + " " + str(pop_sizes) + " " + str(mut_params) + "\n" + probs_lines)

        bacteria_pop, plant_pop, mean_dist = ev_alg(bacteria_pop, plant_pop, t_bact_genes, t_plant_genes)

        if (epoch % 1000 == 0 and epoch != 0):
            orgType = np.random.randint(0, 2)
            funcType = np.random.randint(0, 2)

            funcTypes = [evolution.glob_disaster, evolution.local_surv]
            orgPops = [bacteria_pop, plant_pop]
            orgCls = [Bacteria, Plant]
            orgName = ["bacteria", "plant"]
            funcName = ["disaster", "local_surv"]

            print(epoch, funcName[funcType], orgName[orgType], file=dlog)
            if (orgType == 0):
                bacteria_pop = funcTypes[funcType](bacteria_pop, bact_genes, Bacteria)
            else:
                plant_pop = funcTypes[funcType](plant_pop, plant_genes, Plant)

        utils.flush_file(dlog)

        elapsed = time.time() - start_time
        time_sum += elapsed

        print(str(int(((epoch + 1) / generations * 100))) + str("%"), end=" ")
        print("%0.3f" % (time_sum / (epoch + 1) * (generations - epoch - 1)))

    t_bact_genes = np.array([bact_fitness.run(bacteria.gene) for bacteria in bacteria_pop])
    t_plant_genes = np.array([plant_fitness.run(plant.gene) for plant in plant_pop])


    return [np.array([bacteria.gene for bacteria in bacteria_pop]),
            t_bact_genes, t_plant_genes,
            np.array([plant.gene for plant in plant_pop])]

def print_to_csv(mut_params, statistics, ev_seed, statistics_desc):
    stat_transposed = [np.array(i).T for i in statistics]
    stat_to_csv = np.array([stat_transposed[0][0]] + [i[1] for i in stat_transposed])
    name = "Ba_m" + str(mut_params[0][1]) + "_v" + str(mut_params[0][0] * 10) + \
           "_Pd_m" + str(mut_params[1][1]) + "_v" + str(mut_params[1][0]) + "_id" + str(ev_seed) + ".csv"
    np.savetxt(name, stat_to_csv.T, delimiter=',', fmt="%.10f", header="epoch," + ",".join(statistics_desc))

def run_proc(b_mut_m, p_mut_m, id):
    mut = ((0.05, b_mut_m), (0.05, p_mut_m))
    bacteria_pop_size = 100000
    plant_pop_size = 1000
    gene_sizes = (5, 5)
    inter_size = 2
    generations = 10001

    seed = int(id + b_mut_m * 1000 + p_mut_m * 1000000)
    np.random.seed(seed)

    #pop_sizes - population sizes
    #mut_params - pair of (variation, probabilty) for both species
    #inter_size - common space dimension number
    #gene_size - gene space dimension sizes
    #ev_alg - function, that calculates next generations based on previous generation
    #bact_plant_fitness - fitness functions
    #generator - function that generates fiest generation
    #insit_seed - generator random seed
    #ev_seed - evolution random seed

    simulation_results = simulate_float(pop_sizes=(bacteria_pop_size, plant_pop_size),
                                        mut_params=mut, inter_size=inter_size,
                                        gene_sizes=gene_sizes, generations=generations, ev_alg=evolution.ev_pooling,
                                        bact_fitness=fourier.FourierFitness((gene_sizes[0], inter_size), alpha=1.2, tau=1, func_num=30),
                                        plant_fitness=fourier.FourierFitness((gene_sizes[1], inter_size), alpha=1.2, tau=1, func_num=30),
                                        generator=utils.gen_same_bacteria,
                                        init_seed=seed, ev_seed=seed)

if __name__ == "__main__":
    for i in range(30):
       f = fourier.FourierFitness((4, 2), alpha=1.2, tau=1, func_num=30)
       utils.fitness_vis(f, 4, 2, "fvis" + str(i) + ".svg")

    #multiprocessing support

    #hyperlul = list(itertools.product([0.08], [0.08], list(map(float, range(10, 16)))))

    #with multiprocessing.Pool(6) as workers:
    #    workers.starmap(run_proc, hyperlul)
    #run_proc(0.05, 0.05, 1)
