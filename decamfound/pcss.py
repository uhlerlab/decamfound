
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ipdb


def get_confounding_suff_stats(data, K=None, correlation=False, svd_solver='full'):
    """
    Performs PCA to estimate the global confounders. This approach is motivated
    by POET [1].

    [1] https://arxiv.org/pdf/1201.0175.pdf

    Input:
        data: N (# samples) x p (# measurements) matrix
        K: Number of principal components, i.e., latent factor dimension. If None, that estimates K

    Returns:
		confound_suff_stats: N x p matrix equal to E[x | h]

        est_factor_component: p x p matrix estimating the covariance (correlation)
                              corresponding to the latent factors

    """
    N, p = data.shape
    if K is not None:
        assert (p >= K) and (N >= K)
    if correlation:
        data = data / np.std(data, axis=0)
    pca = PCA(n_components=min(N, p), svd_solver=svd_solver)
    pca.fit(data)
    if K is None:
        K = estimate_K(data, pca.components_.T.copy())
    top_K_eigenValues = pca.explained_variance_[:K].copy()
    top_K_eigenVectors = pca.components_.T[:, :K].copy()
    est_factor_component = top_K_eigenVectors.dot(np.diag(top_K_eigenValues).dot(top_K_eigenVectors.T))
    confound_suff_stats = (top_K_eigenVectors.dot(top_K_eigenVectors.T.dot(data.T))).T
    return confound_suff_stats, est_factor_component, K, pca.components_, pca.explained_variance_


def estimate_K(data, evecs):
    N, p = data.shape
    M = min(N, p) # Upperbound on number of PCs
    K_losses = []
    # Pick K as in Bai and Ng (2002)
    for K in range(1, M):
        top_K_eigenVectors = evecs[:, :K].copy()
        confound_suff_stats = (top_K_eigenVectors.dot(top_K_eigenVectors.T.dot(data.T))).T
        loss = np.log(1 / (N*p) * ((data - confound_suff_stats) ** 2).sum()) # log likelihood
        loss +=  K * (p + N) / (p * N) * np.log((p*N) / (p + N)) # penalizer
        K_losses.append(loss)

    return np.argmin(K_losses) + 1



if __name__ == "__main__":
	# Generate random data
	X = np.random.normal(size=(500, 1000))
	confound_suff_stats, est_factor_component, K, evecs, evals = get_confounding_suff_stats(X, None)
