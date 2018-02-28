import numpy as np


def create_neural_net(f_layer_size, l_layer_size):
    synapse_0 = 0.6 * np.random.random((f_layer_size, f_layer_size)) - 0.3
    synapse_1 = 0.6 * np.random.random((f_layer_size, l_layer_size)) - 0.3

    return np.array([synapse_0, synapse_1])


def sigmoid(x):
    return 1/(1+np.exp(-x))


def run_neural_net(vec, n_net):
    layer_0 = np.array(vec)
    layer_1 = sigmoid(np.dot(layer_0, n_net[0]))
    layer_2 = sigmoid(np.dot(layer_1, n_net[1]))
    return layer_2

