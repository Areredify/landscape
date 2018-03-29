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

def simulate_float(pop_sizes, gene_sizes, generations, mut_params,
                   ev_alg=evolution.ev_basic, fitness=None, generator=utils.gen_random,statistics_func=[],
                   init_seed=0, ev_seed=0, n_threads=1):
    Bacteria.mut_v = mut_params[0][0]
    Bacteria.mut_m = mut_params[0][1]
    Plant.mut_v = mut_params[1][0]
    Plant.mut_m = mut_params[1][1]

    if fitness is None:
        fitness = nnet.NeuralNetFitness(*gene_sizes)
    bacteria_pop, plant_pop = generator(pop_sizes, fitness)
    np.random.seed(init_seed)
    if (generator == utils.gen_same_bacteria):
        ancestral = copy.deepcopy(bacteria_pop[0])

    one_percent = max(1, generations // 100)
    time_sum = 0

    np.random.seed(ev_seed)
    stat_res = [[] for _ in range(len(statistics_func))]
    print(1)
    mean_dist = 8
    for epoch in range(generations):
        start_time = time.time()
        bacteria_genes = [bacteria.gene for bacteria in bacteria_pop]

        transformed_genes = [fitness.run(gene) for gene in bacteria_genes]

        if epoch % 50 == 0:
            for j in range(len(statistics_func)):
                stat_res[j].append(statistics_func[j](**locals()))
        print(Bacteria.mut_v, Bacteria.mut_m, Plant.mut_v, Plant.mut_m)

        bacteria_pop, plant_pop, mean_dist = ev_alg(bacteria_pop, plant_pop, transformed_genes)
        elapsed = time.time() - start_time
        time_sum += elapsed
        if epoch % one_percent == 0:
            print(str(int(((epoch + 1) / generations * 100))) + str("%"), end=" ")
            print("%0.3f" % (time_sum / (epoch + 1) * (generations - epoch - 1)))

    bacteria_genes = [bacteria.gene for bacteria in bacteria_pop]
    transformed_genes = [fitness.run(gene) for gene in bacteria_genes]
    for j in range(len(statistics_func)):
        stat_res[j].append(statistics_func[j](**locals()))

    return [np.array([bacteria.gene for bacteria in bacteria_pop]),
            np.array(transformed_genes),
            np.array([plant.gene for plant in plant_pop])], stat_res

def print_to_csv(mut_params, statistics, ev_seed, statistics_desc):
    stat_transposed = [np.array(i).T for i in statistics]
    stat_to_csv = np.array([stat_transposed[0][0]] + [i[1] for i in stat_transposed])
    name = "Ba_m" + str(mut_params[0][1]) + "_v" + str(mut_params[0][0] * 10) + \
           "_Pd_m" + str(mut_params[1][1]) + "_v" + str(mut_params[1][0]) + "_id" + str(ev_seed) + ".csv"
    np.savetxt(name, stat_to_csv.T, delimiter=',', fmt="%.10f", header="epoch," + ",".join(statistics_desc))

def run_proc(b_mut_m, p_mut_m, id):
    mut = ((0.004, b_mut_m), (0.04, p_mut_m))
    bacteria_pop_size = 10000
    plant_pop_size = 50
    gene_sizes = (20, 20)
    generations = 500
    statistics_func = [utils.stat_av_dist_to_anc, utils.stat_av_dist_to_closest, utils.stat_plant_percentage,
                       utils.stat_mean_dist]
    statistics_desc = ["mean dist to ancestor bacteria", "mean dist to closest plant", "percentage of reacting plants",
                       "mean dist to own bacteria"]
    statistics_ylabels = ["distance", "distance", "percentage", "distance"]
    np.random.seed(30)
    cnt = int(10 * b_mut_m + 10 * p_mut_m * id * 1000)
    print(cnt)
    simulation_results = simulate_float(pop_sizes=(bacteria_pop_size, plant_pop_size),
                                        mut_params=mut,
                                        gene_sizes=gene_sizes, generations=generations, ev_alg=evolution.ev_pooling,
                                        fitness=fourier.FourierFitness(gene_sizes, alpha=1.5, tau=1, func_num=30),
                                        generator=utils.gen_same_bacteria,
                                        statistics_func=statistics_func,
                                        init_seed=30, ev_seed=cnt)
    print_to_csv(mut, simulation_results[1], cnt, statistics_desc)

if __name__ == "__main__":
    hyperlul = list(itertools.product([0.1, 0.2, 0.4], [0.1, 0.2, 0.4], [6, 7]))
    with multiprocessing.Pool(2) as workers:
        workers.starmap(run_proc, hyperlul)

