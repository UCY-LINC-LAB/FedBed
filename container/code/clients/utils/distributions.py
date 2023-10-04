import random
import sys
import traceback
from logging import DEBUG
from flwr.common.logger import log
import numpy as np
from scipy.stats import dirichlet

def gaussian(N=1000, nodes=10, scale=1, **kwargs):
    x = np.linspace(0, nodes - 1, nodes)
    dist = np.exp(-np.abs(x) / 2 / scale ** 2)
    dist /= dist.sum()
    return N * dist


def pareto(N=1000, nodes=10, scale=1, shape=1, **kwargs):
    x = np.linspace(scale, scale + nodes - 1, nodes)
    dist = (shape * (scale ** shape)) / (x ** (shape + 1))
    dist /= dist.sum()
    return N * dist

def flat(N=1000,nodes=10, **kwargs):
    return np.linspace(N/nodes, N/nodes, nodes)

def dirichlet_func(N=1000, nodes=10, random_state=None, **kwargs):
    alpha = np.ones(nodes)/nodes
    samples = dirichlet.rvs(alpha, size=1, random_state=random_state) * N
    res = list(samples.squeeze())
    res = [int(i) for i in res]
    res = [i if i>0 else 1 for i in res]
    res.sort(reverse=True)
    res = np.asarray(res)
    return res


_distributions = dict(
        gaussian=gaussian,
        pareto=pareto,
        flat=flat,
        dirichlet=dirichlet_func
    )

def shuffle_lists(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def dataset_resizing(data,
                     partition,
                     nodes,
                     distribution='flat',
                     distribution_parameters={}):
    if partition > nodes:
        partition = nodes
    dist = _distributions.get(distribution, flat)
    distribution_parameters['nodes'] = nodes
    x, y = data
    if len(x) == 2 and len(y) == 2: # sklearn
        (x_train, y_train), (x_test, y_test) = data
        x_N = len(x_train)
        distribution_parameters['N'] = x_N
        gen_dist = dist(**distribution_parameters).cumsum()
        _x_train, _y_train = apply_distribution(x_train, y_train, gen_dist, partition)
        x_N = len(x_test)
        distribution_parameters['N'] = x_N
        gen_dist = dist(**distribution_parameters).cumsum()
        _x_test, _y_test = apply_distribution(x_test, y_test, gen_dist, partition)
        return (_x_train, _y_train), (_x_test, _y_test)

    # others
    try:
        (x_train, y_train), (x_test, y_test) = (x.data, x.targets), (y.data, y.targets)
    except Exception as ex:
        ex_type, ex_value, ex_traceback = sys.exc_info()
        trace_back = traceback.extract_tb(ex_traceback)
        log(DEBUG, f"5.1 {dir(x)} {dir(y)} {ex_type.__name__}, {ex_value}, {trace_back}")
    x_N = len(x_train)
    distribution_parameters['N'] = x_N
    gen_dist = dist(**distribution_parameters).cumsum()
    x.data, x.targets = apply_distribution(x_train, y_train, gen_dist, partition)
    x_N = len(x_test)
    distribution_parameters['N'] = x_N
    gen_dist = dist(**distribution_parameters).cumsum()
    y.data, y.targets = apply_distribution(x_test, y_test, gen_dist, partition)

    return x, y



def apply_distribution(x, y, dist, partition):
    prev_partition = partition - 1
    start = 0
    if prev_partition >= 0:
        start = int(dist[prev_partition])
    end = int(dist[partition])
    x = x[start: end]
    y = y[start: end]
    log(DEBUG, f"-------------------------------------")
    log(DEBUG, f"Data distribution size: {end-start}. starting point: {start}, ending point: {end} ")
    log(DEBUG, f"-------------------------------------")

    return x, y

