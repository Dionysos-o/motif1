# coding utf-8
import numpy as np
from sklearn.manifold import MDS
import sim_seq2 as sim
from matplotlib import pyplot as plt
import random
import scipy
import os
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import SpectralEmbedding


def cal_pairwise_dist(x: np.ndarray) -> np.ndarray:
    """
    calculate pairwise distance for data by using (a-b)^2 = a^2 + b^2 - 2*a*b
    :param x: data matrix
    :param label: label of data
    :return: pairwise distance matrix of data x
    """
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)

    dist = np.abs(dist)  # float calculation may causes negative values
    return dist


def hd_distance_umap(hd_data: np.ndarray, beta: float) -> np.ndarray:
    """
    calculate the distance in high dimensional space in umap
    every pairwise distance need to be embedded to Gaussian kernel after 1-nearest connection
    :param hd_data: hd data
    :param beta: hyperparameter in Gaussian kernel
    :return: n x n matrix: hd distance matrix
    """
    hd_dist = np.sqrt(cal_pairwise_dist(hd_data))

    for i in range(len(hd_data)):
        hd_dist[i, :] = np.maximum(hd_dist[i, :] - np.min(hd_dist[i, hd_dist[i, :].nonzero()[0]]), 0)
        hd_dist[i, :] = np.exp(-hd_dist[i, :] / beta)

    return hd_dist


def func(x, a, b):
    """
    define the function for curve fit
    :param x: input data
    :param a: parameter1
    :param b: parameter2
    :return: the prediction values
    """
    return 1 / (1 + a * np.power(x, 2*b))


def ld_psai_function(ld_data: np.ndarray, min_dist: float) -> list:
    """
    :param ld_data: n x 2 matrix low dimensional data
    :param min_dist: hyperparameter to control the tightness of cluster
    :return:ld_dist_euclidean, ld_dist , n^2 x 1 matrix after psai function
    """
    ld_dist = []
    for i in range(len(ld_data)):
        if ld_data[i] <= min_dist:
            ld_dist.append(1)
        else:
            ld_dist.append(np.exp(-ld_data[i] - min_dist))
    return ld_dist


def cal_hd_prob_row(dist_row: np.ndarray, rho_i:float, sigma: float) -> np.ndarray:
    """
    For each row of Euclidean distance matrix (dist_row) compute
    probability in high dimensions (1D array)
    :param dist_row:
    :param rho_i:
    :param sigma:
    :return:
    """
    d = dist_row - rho_i
    d[d < 0] = 0
    return np.exp(- d / sigma)


def sigma_binary_search(row: np.ndarray, rho_i: float, k_neighbors: int) -> float:
    """
    Solve equation k_of_sigma(sigma) = fixed_k
    with respect to sigma by the binary search algorithm
    :param row:
    :param rho_i:
    :param k_neighbors:
    :return:
    """
    sigma_lower_limit = 0
    sigma_upper_limit = 1000
    approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
    for i in range(20):
        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
        entropy = np.power(2, np.sum(cal_hd_prob_row(row, rho_i, approx_sigma)))
        if entropy < k_neighbors:
            sigma_lower_limit = approx_sigma
        else:
            sigma_upper_limit = approx_sigma
        if np.abs(k_neighbors - entropy) <= 1e-5:
            break
    return approx_sigma


def cal_hd_prob(hd_data: np.ndarray, k_neighbors: int) -> np.ndarray:
    """
    :param hd_data:
    :param k_neighbors:
    :return:
    """
    n = len(hd_data)
    dist = np.square(euclidean_distances(hd_data, hd_data))
    rho = [sorted(dist[i])[1] for i in range(dist.shape[0])]
    hd_prob = np.zeros((n, n))
    for row_index in range(n):
        sigma = sigma_binary_search(dist[row_index], rho[row_index], k_neighbors)
        hd_prob[row_index] = cal_hd_prob_row(dist[row_index], rho[row_index], sigma)
    return hd_prob


def ce_gradient(p: np.ndarray, ld_data: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    :param p:
    :param ld_data:
    :param a:
    :param b:
    :return:
    """
    ld_data_diff = np.expand_dims(ld_data, 1) - np.expand_dims(ld_data, 0)
    inv_dist = np.power(1 + a * np.square(euclidean_distances(ld_data, ld_data)) ** b, -1)
    Q = np.dot(1 - p, np.power(0.001 + np.square(euclidean_distances(ld_data, ld_data)), -1))
    np.fill_diagonal(Q, 0)
    Q = Q / np.sum(Q, axis=1, keepdims=True)
    fact = np.expand_dims(a * p * (1e-8 + np.square(euclidean_distances(ld_data, ld_data))) ** (b - 1) - Q, 2)
    return 2 * b * np.sum(fact * ld_data_diff * np.expand_dims(inv_dist, 2), axis=1)


def ce(p: np.ndarray, ld_data: np.ndarray, a: float, b: float) -> float:
    """
    :param p:
    :param ld_data:
    :param a:
    :param b:
    :return:
    """
    Q = np.power(1 + a * np.square(euclidean_distances(ld_data, ld_data))**b, -1)
    return - p * np.log(Q + 0.01) - (1 - p) * np.log(1 - Q + 0.01)


def umap(x: np.ndarray, labels: np.ndarray, min_dist: float,  k_neigh: int,
         no_dims: int = 2, max_iter: int = 100) -> np.ndarray:
    """
    Runs umap on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    :param x: n x d high dimension data with
    :param labels: n x 1 the label of high dimensional data
    :param min_dist: float
    :param a: parameter in the family of curves in low dimensional distribution
    :param b: parameter in the family of curves in low dimensional distribution
    :param k_neigh: number of sampling neighbors
    :param no_dims: convert data to this dimension
    :param max_iter: number of steps of iteration
    :return: n x no_dims  low dimensional data
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # initialize parameters
    # x = pca(x, initial_dims).real
    path = 'figure/umap/min_dist = ' + str(min_dist) + ' k_neigh = ' + str(k_neigh)
    if not os.path.exists(path):
        os.makedirs(path)
    n = x.shape[0]
    learning_rate = 1

    # high dimensional distribution

    h_prob = cal_hd_prob(x, k_neigh)
    h_prob = (h_prob + np.transpose(h_prob)) / 2

    # low dimensional distribution
    test_x = np.linspace(0, 3, 300)
    para = scipy.optimize.curve_fit(func, test_x, ld_psai_function(test_x, min_dist))
    a = para[0][0]
    b = para[0][1]
    model = SpectralEmbedding(n_components=no_dims, n_neighbors=50)
    y = model.fit_transform(np.log(x + 1))

    # Run iterations
    ce_array = []
    for i in range(max_iter):
        y = y - learning_rate * ce_gradient(h_prob, y, a, b)
        ce_current = np.sum(ce(h_prob, y, a, b)) / 1e+5
        ce_array.append(ce_current)
        if i % 10 == 0:
            plt.figure(figsize=(20, 15))
            plt.scatter(y[:, 0], y[:, 1], c=labels.astype(int), cmap='tab10', s=50)
            plt.title("UMAP dimensional reduction for 3motif with random min_dist={}, k={}".format(str(min_dist), str(k_neigh)),
                      fontsize=20)
            plt.xlabel("UMAP1", fontsize=20)
            plt.ylabel("UMAP2", fontsize=20)
            plt.savefig(path + '/UMAP_iter' + str(i) + '.png')
            plt.close()
            print("Cross-Entropy = " + str(ce_current) + " after " + str(i) + " iterations")
    plt.figure(figsize=(20,15))
    plt.plot(ce_array)
    plt.title('cross-entropy', fontsize=20)
    plt.xlabel('ITERATION', fontsize=20)
    plt.ylabel("CROSS-ENTROPY", fontsize=20)
    plt.savefig(path + '/UMAP_iter' + '_Final' + '.png')
    return y


if __name__ == "__main__":
    # Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.
    # X = np.loadtxt("mnist2500_X.txt")
    # labels = np.loadtxt("mnist2500_labels.txt")
    data, label1, label2 = sim.simulation_run_1()
    X = data.numpy()
    label = label2.numpy()
    para_min_dist = np.linspace(0.1, 0.99, 10)
    para_k = np.linspace(1, 15, 15).astype(int)
    for min_dist_i in para_min_dist:
        for k_i in para_k:
            umap(X, label, min_dist_i, k_i)


