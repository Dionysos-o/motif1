# coding utf-8
import numpy as np
from sklearn.manifold import MDS
import sim_seq2 as sim
from matplotlib import pyplot as plt
import random
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties


def cal_pairwise_dist(x: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    calculate pairwise distance for data by using (a-b)^2 = a^w + b^2 - 2*a*b
    :param x: data matrix
    :param label: label of data
    :return: pairwise distance matrix of data x
    """
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    dist = dist / 2
    '''
    for i in range(len(x)):
        if label[i] % 2 == 1:
            pass
            dist[i, np.where(label-label[i] == 0)[0]] = 1
        if label[i] == 6:
            pass
            dist[i, np.where(label - label[i] != 0)[0]] = 100
            #dist[i, np.where(label - label[i] == 0)[0]] = 1
    '''
    return dist


def cal_pairwise_dist_sample(dist: np.ndarray, k_neighbors: int) -> np.ndarray:
    """
    calculate the distance from sampling 1-order neighbors
    :param dist: original pairwise distance matrix for high dimension data
    :param k_neighbors: number of 1-order neighbors
    :return: new pairwise distance matrix
    """

    new_dist = np.zeros((len(dist), len(dist)))
    for i in range(len(dist)):
        index_order_1 = np.where(dist[i, :] <= 1)[0]
        for j in range(len(dist)):
            sum_dist = 0
            if len(index_order_1) >= k_neighbors:
                sample_index = random.sample(range(0, len(index_order_1)), k_neighbors)
                for n in sample_index:
                    sum_dist = sum_dist + dist[index_order_1[n]][j]
                new_dist[i][j] = sum_dist / k_neighbors
            else:
                new_dist[i][j] = dist[i][j]
    return new_dist


def levenshtein(str1: str, str2: str) -> int:
    """
    calculate the levenshtein distance for two strings
    :param str1: string 1
    :param str2: string 2
    :return: levenshtein distance
    """
    if str1 == str2:
        return 0
    elif len(str1) == 0:
        return len(str2)
    elif len(str2) == 0:
        return len(str1)
    v0 = [None] * (len(str2) + 1)
    v1 = [None] * (len(str2) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(str1)):
        v1[0] = i + 1
        for j in range(len(str2)):
            cost = 0 if str1[i] == str2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
    return v1[len(t)]


def cal_levenshtein(x_str: np.ndarray) -> np.ndarray:
    """
    :param x_str: data in string form
    :return: levenshtein distance matrix
    """
    dist = np.zeros((len(x_str), len(x_str)))
    for i in range(len(x_str)):
        for j in range(len(x_str)):
            dist[i][j] = levenshtein(x_str[i], x_str[j])
    return dist


def cal_probability(dist: np.ndarray, beta: float, idx=0) -> np.ndarray:
    """
    calculate Gaussian probability for each point based on the distance matrix
    :param dist: n x n distance matrix
    :param beta: parameter in gaussian distribution to control the variance
    :param idx:
    :return: n x n matrix
    """
    dist = dist ** 2
    prob = np.exp(-dist * beta)
    # set the
    prob[idx] = 0.0
    sum_prob = np.sum(prob)
    prob /= sum_prob
    return prob


def pca(x: np.ndarray, no_dims: int = 50) -> np.ndarray:
    """
    initialize data with PCA if needed
    :param x: n x d high dimension data with
    :param no_dims: convert data to this dimension
    :return: n x no_dims processed data for tsne
    """
    print("Preprocessing the data using PCA...")
    (n, d) = x.shape
    x = x - np.tile(np.mean(x, 0), (n, 1))
    l, m = np.linalg.eig(np.dot(x.T, x))
    y = np.dot(x, m[:, 0: no_dims])
    return y


def affinity(x: np.array) -> np.ndarray:
    """
    Given a set of kmers, calculate everyone's frequency
    :param x: kmer data
    :return: frequency label
    """
    (n, d) = x.shape
    lis_x = x.tolist()
    fre = []
    for xi in lis_x:
        fre.append(lis_x.count(xi) / n)
    fre = np.array(fre)
    return fre
 
 
def tsne(x: np.ndarray, label: np.ndarray, beta: float, a: float, b: float, k_neigh: int, alpha: float, no_dims: int = 2, max_iter: int = 1000) -> np.ndarray:
    """
    Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    :param x: n x d high dimension data with
    :param label: n x 1 labels of data
    :param beta: the parameter in Gaussian distribution
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
    n = len(x)
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 10
    bound = 1000
    y = np.random.randn(n, no_dims)
    # y = np.random.rand(n, 2) * 2 * bound - bound
    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    fre = affinity(x) * alpha

    # symmetric

    # P = search_prob(x, 1e-5, perplexity)
    dist = cal_pairwise_dist(x, label)

    dist = cal_pairwise_dist_sample(dist, k_neigh)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(dist, cmap=plt.cm.hot_r)
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    plt.title('Hot figure of distance matrix')
    plt.show()
    '''
    h_prob = cal_probability(dist, beta)
    h_prob + np.transpose(h_prob)
    h_prob = h_prob / np.sum(h_prob)
    # early exaggeration
    h_prob = h_prob * 4
    h_prob = np.maximum(h_prob, 1e-12)
 
    # Run iterations

    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        # num = 1 / (1 + a * np.power(np.maximum(np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y), 1e-12), b))
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        l_prob = num / np.sum(num)
        l_prob = np.maximum(l_prob, 1e-12)
 
        # Compute gradient

        diff_prob = h_prob - l_prob
        times_prob = h_prob * l_prob
        diff_times_prob = h_prob - times_prob
        for i in range(n):

            #dy[i, :] = 2 * a * b * np.sum(np.tile(diff_times_prob[:, i] * num[:, i] * (fre[i] + fre), (no_dims, 1)).T * (y[i, :] - y), 0)

            #dy[i, :] = 2 * a * b * np.sum(np.tile(diff_prob[:, i] * num[:, i] * (fre[i] + fre), (no_dims, 1)).T * (y[i, :] - y), 0)
            #test_y = y[i, :] - y
            #test_y = np.where(test_y == 0, 1e-12, test_y)  # notation needs change
            #test_y_sub = np.power(test_y+0j, 2 * b-1)
            #final_y = np.where(test_y == 0, test_y, test_y_sub)
            #dy[i, :] = 2 * a * b*np.sum(np.tile(diff_prob[:, i] * num[:, i], (no_dims, 1)).T * final_y, 0)

            dy[i, :] = np.sum(np.tile(diff_prob[:, i] * num[:, i], (no_dims, 1)).T * (y[i, :] - y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain

        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))

        # Compute current value of cost function

        if (iter + 1) % 100 == 0:
            if iter > 100:
                loss = np.sum(h_prob * np.log(h_prob / l_prob))
            else:
                loss = np.sum(h_prob / 4 * np.log(h_prob / 4 / l_prob))
            print("Iteration ", (iter + 1), ": error is ", loss)

        # Stop lying about P-values

        if iter == 100:
            h_prob = h_prob / 4
    print("finished training!")
    return y
 
 
if __name__ == "__main__":
    # Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.
    # X = np.loadtxt("mnist2500_X.txt")
    # labels = np.loadtxt("mnist2500_labels.txt")
    data, label1, label2 = sim.simulation_run_1()
    X = data.numpy()
    Y = tsne(X, label1, 1 / 2, 1.987, 0.98, k_neigh=20, alpha=3.0)
    fig, ax = plt.subplots()
    title_str = 'Tsne dimensional reduction for 3motif with random sample 20'
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=label2, cmap='tab10')
    ax.legend(*sc.legend_elements(), title="clusters", loc='lower left')
    plt.title(title_str)
    plt.show()
    plt.savefig("./figure/Tsne dimensional reduction for 3motif with random sample 20.png")


    '''
    alpha_set = np.linspace(1, 2.5, 50)
    for alpha_i in alpha_set:
        Y = tsne(X, label1, 1 / 2, 1.987, 0.98, k_neigh = alpha_i)
        fig, ax = plt.subplots()
        title_str = 'Tsne dimensional reduction for 3motif with random(weight-loss) alpha={}' \
            .format(str(k_i))
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=label2, cmap='tab10')
        ax.legend(*sc.legend_elements(), title="clusters")
        plt.title(title_str)
        plt.savefig("./figure/figure_loss_alpha={}.png".format(str(k_i)))


    k = np.linspace(1, 2.5, 50)
    for k_i in k:
        Y = tsne(X, label1, 1 / 2, 1, 1, k_i)
        fig, ax = plt.subplots()
        title_str = 'Tsne dimensional reduction for 3motif with random(weight-loss) alpha={}' \
            .format(str(k_i))
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=label2, cmap='tab10')
        ax.legend(*sc.legend_elements(), title="clusters")
        plt.title(title_str)
        plt.savefig("./figure/figure_loss_alpha={}.png".format(str(k_i)))  
    k = 10
    Y = tsne(X, label1, 1 / 2, 1.987, 0.98, k)
    fig, ax = plt.subplots()
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=label2, cmap='tab10')
    ax.legend(*sc.legend_elements(), title="clusters")
    title_str = 'Tsne dimensional reduction for 3motif with random(weight'
    plt.title(title_str)
    plt.show()
    k = 10
    a_values = np.linspace(1.1, 2, 10)
    b_values = np.linspace(0.1, 1, 10)
    for index_a in a_values:
        for index_b in b_values:
            Y = tsne(X, label1, 1/2, 1, 1, k)
            fig, ax = plt.subplots()
            title_str = 'Tsne dimensional reduction for 3motif with random(different-t a={},b={}' \
                .format(str(index_a), str(index_b))
            sc = ax.scatter(Y[:, 0], Y[:, 1], c=label2, cmap='tab10')
            ax.legend(*sc.legend_elements(), title="clusters")
            plt.title(title_str)
            plt.savefig("./figure/figure_a={},b={}.png".format(str(index_a), str(index_b)))

    #plt.savefig("./figure/figure_sample_k={}.png".format(str(k)))
    # Y = pca(X, 2).real
    # fun = MDS(2)
    # Y = fun.fit_transform(X)

    a_values = np.linspace(1, 2, 5)
    b_values = np.linspace(0.1, 1, 5)
    for index_a in a_values:
        for index_b in b_values:
            Y = tsne(X, label1, 1/2, index_a, index_b)
            fig, ax = plt.subplots()
            title_str = 'Tsne dimensional reduction for 3motif with random(different-t a={},b={}' \
                .format(str(index_a), str(index_b))
            sc = ax.scatter(Y[:, 0], Y[:, 1], c=label2, cmap='tab10')
            ax.legend(*sc.legend_elements(), title="clusters")
            plt.title(title_str)
            plt.savefig("./figure/figure_a={},b={}.png".format(str(index_a), str(index_b)))
    '''
