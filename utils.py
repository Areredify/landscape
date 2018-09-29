import numpy as np
import numpy.random
import org
import numpy.linalg as npl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import os.path
import sklearn.cluster as sklc
import sklearn.metrics as sklm
import scipy.spatial
from org import Bacteria, Plant
import os
import sklearn.feature_selection as skfs

def sqlen(x):
    return np.inner(x, x)

#def gen_random(pop_sizes, fitness):
#    bacteria_pop = [org.Bacteria(i) for i in np.random.uniform(-1, 1, (pop_sizes[0], fitness.gene_sizes[0]))]
#    transformed_genes = [fitness.run(bacteria.gene) for bacteria in bacteria_pop]
#    plant_pop = [org.Plant(transformed_genes[x] + np.random.uniform(-0.001, 0.001, fitness.gene_sizes[1]))
#                 for x in np.random.choice(range(pop_sizes[0]), size=pop_sizes[1])]
#
#    return bacteria_pop, plant_pop


def gen_same_bacteria(pop_sizes, gene_sizes):
    haplotype_b = np.random.uniform(-1, 1, gene_sizes[0])
    haplotype_p = np.random.uniform(-1, 1, gene_sizes[1])

    # bacteria constructor accepts
    # 1. gene
    # 2. ancestor id list
    # 3. gene id
    # if no gene id is provided, new id is created

    bacteria_pop = [org.Bacteria(haplotype_b, [], 0) for _ in range(pop_sizes[0])]
    #plant_pop = [org.Plant(haplotype_p, [], 0) for _ in range(pop_sizes[1])]
    plant_pop = [org.Plant(gene, [], None)
                 for gene in np.random.uniform(-1, 1, (pop_sizes[1], gene_sizes[1]))]
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

def fitness_vis(fitness, s_in, s_out, path):
    if (s_out != 2):
        return
    fig = plt.figure()

    #dirs = np.random.random((3, s_in))
    dirs = []
    for i in range(s_in):
        vec = np.zeros(s_in)
        vec[i] += 1.0
        dirs.append(vec)

    out_x, out_y, out_c = [], [], []
    step = 0.001
    dist = 0.25
    nstep = int(dist / step)

    cc = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    cnt = 0
    for i in dirs:
        for j in range(nstep):
            lul = fitness.run(i * step * j)
            out_x.append(lul[0])
            out_y.append(lul[1])
            out_c.append(tuple(cc[cnt] * (j / nstep)))
        cnt += 1

    plt.scatter(out_x, out_y, c=out_c, s=2)
    zim = fitness.run(np.zeros(s_in))
    plt.scatter([zim[0]], [zim[1]], c=[(1.0, 0.0, 1.0)])
    plt.grid(True)
    plt.subplots_adjust(top=0.85)
    plt.savefig(path, dpi=100)
    plt.close(fig)


def stat_plant_disp(plant_pop, epoch, **kwargs):
    genes = [plant.gene for plant in plant_pop]
    mean_vec = np.mean(genes, axis=0)
    res = sum([sqlen(gene - mean_vec) for gene in genes]) / len(plant_pop)
    return epoch, res

def stat_bact_disp(transformed_genes, epoch, **kwargs):
    mean_vec = np.mean(transformed_genes, axis=0)
    res = sum([sqlen(gene - mean_vec) for gene in transformed_genes]) / len(transformed_genes)
    return epoch, res

def save_2d(t_bact_genes, t_plant_genes, epoch, name, title):
    fig = plt.figure()
    plt.scatter(*zip(*t_bact_genes), color=(0, 0, 0, 0.07))
    plt.scatter(*zip(*t_plant_genes), color=(1, 0, 0, 0.5))
    #plt.title(title)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.grid(True)
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(name, 'epoch_' + str(epoch) + '.png'), dpi=300)
    plt.close(fig)

def save_3d(t_bact_genes, t_plant_genes, epoch, name, title):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*zip(*t_bact_genes), color="black")
    ax.scatter(*zip(*t_plant_genes), color="red")
    #plt.figtext(title)
    plt.figtext(0.05, 0.95, title, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.savefig(os.path.join(name, 'epoch_' + str(epoch) + '.png'), dpi=100)

    if (epoch % 50 == 0):
        plt.show()
    else:
        plt.close()

def pop_distance(t_bact_genes, t_plant_genes):
    d = []

    for k in range(4, 15):
        bact_clustered = kmeans_cluster(t_bact_genes, k)
        plant_clustered = kmeans_cluster(t_plant_genes, k)
        mtx1, mtx2, disparity = scipy.spatial.procrustes(bact_clustered, plant_clustered)
        d.append(disparity)
    return min(d)

def kmeans_cluster(a, k):
    model = sklc.KMeans(k, n_init=10)
    model.fit(a)
    return model.cluster_centers_

def normed_distance_db(bact_pop, plant_pop, bact_db, plant_db):
    d1, d2 = distance_db(bact_pop, bact_db), distance_db(plant_pop, plant_db)
    return d1 / Bacteria.mut_v, d2 / Plant.mut_v

def distance_db(population, db):
    sum = 0

    for org in population:
        if len(org.parent_ids) != 0:
            sum += npl.norm(org.gene - db[org.parent_ids[0]]) / len(org.parent_ids)

    return sum / len(population)

def update_db(bact_pop, plant_pop, bact_db, plant_db, epoch, n):
    if epoch == 0 or epoch % n != 0:
        return update_simple_db(bact_pop, bact_db), update_simple_db(plant_pop, plant_db)
    else:
        return update_and_erase_db(bact_pop, bact_db), update_and_erase_db(plant_pop, plant_db)

def update_simple_db(population, db = dict()):
    for org in population:
        if org.g_id not in db:
            db[org.g_id] = org.gene

    return db

def update_and_erase_db(population, db):
    new_db = dict()
    for org in population:
        for db_id in org.parent_ids:
            new_db[db_id] = db[db_id]
        new_db[org.g_id] = org.gene
    return new_db

def flush_file(file):
    file.flush()
    os.fsync(file)

def uniform_labels(t_genes, k, mn, mx):
    siz = len(mn)
    probs = np.array([0 for _ in range(k ** siz)])
    for g in t_genes:
        coords = ((g - mn) / (mx - mn) * (k - 1e-4)).astype(int)

        label = 0
        for i in coords:
            label = label * k + i
        probs[label] += 1
    probs = probs / np.sum(probs)
    return probs

def update_t_db(population, db, fitness, epoch):
    if (epoch % 100 == 0):
        db = dict()

    res = []
    for org in population:
        if org.g_id not in db:
            db[org.g_id] = fitness.run(org.gene)
        res.append(db[org.g_id])

    return res, db

def intra_labeling(pop_genes, k):
    mn = np.min(pop_genes, axis=0)
    mx = np.max(pop_genes, axis=0)

    flag = False
    for i in range(len(mx)):
        if (mx[i] == mn[i]):
            flag = True

    if (flag):
        return ([1] + [0] * (len(pop_genes) - 1))
    probs = uniform_labels(pop_genes, k, mn, mx)

    return probs

def inter_labeling(t_bact_genes, t_plant_genes, k):
    p_mn = np.min(t_plant_genes, axis=0)
    b_mn = np.min(t_bact_genes, axis=0)
    mn = np.min(np.array([b_mn, p_mn]), axis = 0)

    p_mx = np.max(t_plant_genes, axis=0)
    b_mx = np.max(t_bact_genes, axis=0)
    mx = np.max(np.array([b_mx, p_mx]), axis=0)

    p_probs = uniform_labels(t_plant_genes, k, mn, mx)
    b_probs = uniform_labels(t_bact_genes, k, mn, mx)

    return b_probs, p_probs

def pop_entropy(probs):
    sm = 0
    for i in probs:
        if (i != 0):
            sm -= i * np.log(i)
    return sm

def mc_diversity(pop, samples):
    sm = 0
    for i in range(samples):
        ind = np.random.randint(0, len(pop), 2)
        sm += np.linalg.norm(pop[ind[0]].gene - pop[ind[1]].gene)

    return sm / samples

def hell_dist(b_probs, p_probs):
    score = 0
    for i in range(len(b_probs)):
        score += np.sqrt(p_probs[i] * b_probs[i]);

    return np.sqrt(1 - score);