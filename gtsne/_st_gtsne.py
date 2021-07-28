import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from st_gtsne import ST_GTSNE



def gtsne(
    data,
    pca_d=None,
    D_Z = None,
    d= 2,
    K = None,
    alpha = 1e-2,
    beta = 5e-2,
    perplexity=30.0,
    theta=0.5,
    random_state=None,
    copy_data=False,
    verbose=False,
):
    """
    Run Barnes-Hut T-SNE on _data_.

    @param data         The data.

    @param pca_d        The dimensionality of data is reduced via PCA
                        to this dimensionality. If this is set, then run PCA on X to get Z, and set X = Z,
                        D_Z = pca_d

    @param D_Z          If pca_d is None, this will run PCA on X to get Z. Default min(50,shape(X)[1])

    @param d            The embedding dimensionality. Must be fixed to 2.

    @param K            The number of k-means clusters. if it is None(default), K = max(min(100, N/30), 3)

    @param alpha        the weight of the macro loss, Default 1e-2.

    @param beta         the weight of the k-means loss, Default 1e-5.

    @param perplexity   The perplexity controls the effective number of
                        neighbors.

    @param theta        If set to 0, exact t-SNE is run, which takes
                        very long for dataset > 5000 samples.

    @param random_state A numpy RandomState object; if None, use
                        the numpy.random singleton. Init the RandomState
                        with a fixed seed to obtain consistent results
                        from run to run.

    @param copy_data    Copy the data to prevent it from being modified
                        by the C code

    @param verbose      Verbose output from the training process
    """
    N, _ = data.shape
    if K is None:
        K = max(min(100, int(N*1./30.)),3)
    if pca_d is None:
        if copy_data:
            X = np.copy(data)
        else:
            X = data
        cov = np.dot(data.T, data) / N
        u, s, v = la.svd(cov, full_matrices=False)
        if D_Z is not None:
            D_Z =min(D_Z,X.shape[1])
        else:
            D_Z = min(50, X.shape[1])
        u = u[:, 0:D_Z]
        Z = np.dot(data, u)
    else:
        # do PCA
        data -= data.mean(axis=0) # mean shape D,  x_i -\bar x relative position unchanged

        # working with covariance + (svd on cov.) is
        # much faster than svd on data directly.
        cov = np.dot(data.T, data) / N
        u, s, v = la.svd(cov, full_matrices=False)
        u = u[:, 0:pca_d]
        X = np.dot(data, u)
        Z = X

    kmeans_= KMeans(n_clusters=K)
    kmeans_.fit(X)
    C = kmeans_.cluster_centers_

    if random_state is None:
        seed = np.random.randint(2 ** 32 - 1)
    else:
        seed = random_state.randint(2 ** 32 - 1)

    _st_gtsne = ST_GTSNE()
    def castf64(X_):
        return np.asarray(X_,dtype=np.float64)
    X = castf64(X)
    Z = castf64(Z)
    C = castf64(C)
    # def run(self, X, Z, N, K, D, D_Z, d, alpha, beta,  perplexity, theta, seed, verbose=False):
    Y = _st_gtsne.run(X, Z, C, N, K, X.shape[1], Z.shape[1], d, alpha, beta, perplexity, theta, seed, verbose)
    return Y
